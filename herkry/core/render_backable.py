import torch
import torch.nn as nn
import triton
import triton.language as tl
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

# ==========================================
# 1. 配置类
# ==========================================
@dataclass
class RenderConfig:
    tile_size: int = 2
    max_gauss_chunk: int = 16
    min_sigma_px: float = 0.1
    support_sigma: float = 3.0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

# ==========================================
# 2. 辅助函数
# ==========================================

def get_conic_and_bbox(gs_params, H, W, config: RenderConfig):
    device = gs_params.device
    dtype = gs_params.dtype

    scale_xy = torch.tensor([W, H], device=device, dtype=dtype)
    means = gs_params[..., 0:2] * scale_xy

    scale_axes = torch.tensor([W, H], device=device, dtype=dtype)
    raw_axes = gs_params[..., 2:4] * scale_axes
    axes = torch.clamp(raw_axes, min=config.min_sigma_px)

    theta = gs_params[..., 4]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    inv_a2 = 1.0 / (axes[..., 0] ** 2)
    inv_b2 = 1.0 / (axes[..., 1] ** 2)

    conic_a = inv_a2 * cos_theta**2 + inv_b2 * sin_theta**2
    conic_b = (inv_a2 - inv_b2) * cos_theta * sin_theta
    conic_c = inv_a2 * sin_theta**2 + inv_b2 * cos_theta**2

    conic = torch.stack([conic_a, conic_b, conic_c], dim=-1)
    radii = axes.max(dim=-1).values * float(config.support_sigma)
    return means, conic, radii


def bin_and_sort_gaussians(means, radii, H, W, tile_size):
    B, N, _ = means.shape
    device = means.device

    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    num_tiles_per_batch = tiles_x * tiles_y

    min_x = torch.clamp(means[..., 0] - radii, 0, W - 1)
    max_x = torch.clamp(means[..., 0] + radii, 0, W - 1)
    min_y = torch.clamp(means[..., 1] - radii, 0, H - 1)
    max_y = torch.clamp(means[..., 1] + radii, 0, H - 1)

    min_tx = (min_x / tile_size).floor().int().clamp(0, tiles_x - 1)
    min_ty = (min_y / tile_size).floor().int().clamp(0, tiles_y - 1)
    max_tx = (max_x / tile_size).floor().int().clamp(0, tiles_x - 1)
    max_ty = (max_y / tile_size).floor().int().clamp(0, tiles_y - 1)

    counts_x = (max_tx - min_tx) + 1
    counts_y = (max_ty - min_ty) + 1
    num_overlaps = counts_x * counts_y

    total_interactions = num_overlaps.sum().item()
    if total_interactions == 0:
        return None, None, (tiles_x, tiles_y)

    offsets = torch.zeros((B * N + 1,), device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(num_overlaps.view(-1), dim=0)

    keys = torch.arange(total_interactions, device=device, dtype=torch.int64)

    gauss_idx_flat = torch.searchsorted(offsets, keys, right=True) - 1
    batch_idx = gauss_idx_flat // N

    relative_idx = keys - offsets[gauss_idx_flat]
    width_in_tiles = counts_x.view(-1)[gauss_idx_flat]

    dy = relative_idx // width_in_tiles
    dx = relative_idx % width_in_tiles

    tx = min_tx.view(-1)[gauss_idx_flat] + dx
    ty = min_ty.view(-1)[gauss_idx_flat] + dy

    global_tile_id = batch_idx * num_tiles_per_batch + ty * tiles_x + tx

    sorted_indices = torch.argsort(global_tile_id)
    sorted_gauss_ids = gauss_idx_flat[sorted_indices].int()
    sorted_keys = global_tile_id[sorted_indices]

    total_tiles = B * num_tiles_per_batch
    tile_counts = torch.bincount(sorted_keys, minlength=total_tiles)

    tile_starts = torch.zeros(total_tiles + 1, device=device, dtype=torch.int32)
    tile_starts[1:] = torch.cumsum(tile_counts, dim=0)

    return tile_starts, sorted_gauss_ids, (tiles_x, tiles_y)


# ==========================================
# 3. Triton Kernels
# ==========================================

@triton.jit
def render_forward_hybrid_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    conic_ptr,
    colors_ptr,
    image_ptr,
    H, W, C,
    tiles_x, tiles_y,
    stride_b_means, stride_n_means,
    stride_b_conic, stride_n_conic,
    stride_b_colors, stride_n_colors,
    stride_b_img, stride_c_img, stride_h_img, stride_w_img,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_tile_linear = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    num_tiles_per_batch = tiles_x * tiles_y
    global_tile_id = pid_batch * num_tiles_per_batch + pid_tile_linear

    range_start = tl.load(tile_starts_ptr + global_tile_id)
    range_end = tl.load(tile_starts_ptr + global_tile_id + 1)

    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    if chunk_start >= range_end:
        return

    chunk_end = tl.minimum(chunk_start + GAUSS_CHUNK, range_end)

    tx = pid_tile_linear % tiles_x
    ty = pid_tile_linear // tiles_x
    off_x = tx * TILE_SIZE
    off_y = ty * TILE_SIZE

    px = off_x + tl.arange(0, TILE_SIZE)
    py = off_y + tl.arange(0, TILE_SIZE)

    px_mask_1d = px < W
    py_mask_1d = py < H

    px_mesh = px[None, :]
    py_mesh = py[:, None]

    for g_ptr_idx in range(chunk_start, chunk_end):
        g_flat_idx = tl.load(sorted_gauss_ids_ptr + g_ptr_idx)

        ptr_mean = means_ptr + g_flat_idx * stride_n_means
        mu_x = tl.load(ptr_mean + 0)
        mu_y = tl.load(ptr_mean + 1)

        ptr_conic = conic_ptr + g_flat_idx * stride_n_conic
        A = tl.load(ptr_conic + 0)
        B_coef = tl.load(ptr_conic + 1)
        C_coef = tl.load(ptr_conic + 2)

        dx = (px_mesh + 0.5) - mu_x
        dy = (py_mesh + 0.5) - mu_y
        power = -0.5 * (A * dx * dx + 2.0 * B_coef * dx * dy + C_coef * dy * dy)
        weight = tl.exp(power)

        ptr_color_base = colors_ptr + g_flat_idx * stride_n_colors

        for c_base in range(0, C, BLOCK_C):
            c_offsets = c_base + tl.arange(0, BLOCK_C)
            c_mask_1d = c_offsets < C

            c_vals = tl.load(ptr_color_base + c_offsets, mask=c_mask_1d, other=0.0)
            weighted_color = weight[:, :, None] * c_vals[None, None, :]

            out_ptr_base = image_ptr + pid_batch * stride_b_img
            out_pixel_off = py_mesh[:, :, None] * stride_h_img + px_mesh[:, :, None] * stride_w_img
            out_c_off = c_offsets[None, None, :] * stride_c_img
            out_ptrs = out_ptr_base + out_pixel_off + out_c_off

            mask_x = px_mask_1d[None, :, None]
            mask_y = py_mask_1d[:, None, None]
            mask_c = c_mask_1d[None, None, :]
            write_mask = mask_x & mask_y & mask_c

            tl.atomic_add(out_ptrs, weighted_color, mask=write_mask)


@triton.jit
def render_forward_no_atomic_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    conic_ptr,
    colors_ptr,
    image_ptr,
    H, W, C,
    tiles_x, tiles_y,
    stride_n_means,
    stride_n_conic,
    stride_n_colors,
    stride_b_img, stride_c_img, stride_h_img, stride_w_img,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr
):
    pid_tile_linear = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_cblock = tl.program_id(2)

    num_tiles_per_batch = tiles_x * tiles_y
    global_tile_id = pid_batch * num_tiles_per_batch + pid_tile_linear

    range_start = tl.load(tile_starts_ptr + global_tile_id)
    range_end = tl.load(tile_starts_ptr + global_tile_id + 1)

    tx = pid_tile_linear % tiles_x
    ty = pid_tile_linear // tiles_x
    off_x = tx * TILE_SIZE
    off_y = ty * TILE_SIZE

    px = off_x + tl.arange(0, TILE_SIZE)
    py = off_y + tl.arange(0, TILE_SIZE)
    px_mask_1d = px < W
    py_mask_1d = py < H

    px_mesh = px[None, :]
    py_mesh = py[:, None]

    c_offsets = pid_cblock * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask_1d = c_offsets < C
    acc = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), dtype=tl.float32)

    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            g_ptr_idx = chunk_start + local_g
            valid_g = g_ptr_idx < range_end
            g_flat_idx = tl.load(sorted_gauss_ids_ptr + g_ptr_idx, mask=valid_g, other=0)

            ptr_mean = means_ptr + g_flat_idx * stride_n_means
            mu_x = tl.load(ptr_mean + 0, mask=valid_g, other=0.0)
            mu_y = tl.load(ptr_mean + 1, mask=valid_g, other=0.0)

            ptr_conic = conic_ptr + g_flat_idx * stride_n_conic
            A = tl.load(ptr_conic + 0, mask=valid_g, other=0.0)
            B_coef = tl.load(ptr_conic + 1, mask=valid_g, other=0.0)
            C_coef = tl.load(ptr_conic + 2, mask=valid_g, other=0.0)

            dx = (px_mesh + 0.5) - mu_x
            dy = (py_mesh + 0.5) - mu_y
            power = -0.5 * (A * dx * dx + 2.0 * B_coef * dx * dy + C_coef * dy * dy)
            weight = tl.where(valid_g, tl.exp(power), 0.0)

            ptr_color_base = colors_ptr + g_flat_idx * stride_n_colors
            c_vals = tl.load(ptr_color_base + c_offsets, mask=valid_g & c_mask_1d, other=0.0)
            acc += weight[:, :, None] * c_vals[None, None, :]

    out_ptr_base = image_ptr + pid_batch * stride_b_img
    out_pixel_off = py_mesh[:, :, None] * stride_h_img + px_mesh[:, :, None] * stride_w_img
    out_c_off = c_offsets[None, None, :] * stride_c_img
    out_ptrs = out_ptr_base + out_pixel_off + out_c_off

    write_mask = px_mask_1d[None, :, None] & py_mask_1d[:, None, None] & c_mask_1d[None, None, :]
    tl.store(out_ptrs, acc, mask=write_mask)


@triton.jit
def render_backward_hybrid_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    conic_ptr,
    colors_ptr,
    grad_image_ptr,
    grad_means_ptr,
    grad_conic_ptr,
    grad_colors_ptr,
    H, W, C,
    tiles_x, tiles_y,
    stride_b_means, stride_n_means,
    stride_b_conic, stride_n_conic,
    stride_b_colors, stride_n_colors,
    stride_b_gimg, stride_c_gimg, stride_h_gimg, stride_w_gimg,
    stride_b_gmeans, stride_n_gmeans,
    stride_b_gconic, stride_n_gconic,
    stride_b_gcolors, stride_n_gcolors,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_tile_linear = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    num_tiles_per_batch = tiles_x * tiles_y
    global_tile_id = pid_batch * num_tiles_per_batch + pid_tile_linear

    range_start = tl.load(tile_starts_ptr + global_tile_id)
    range_end = tl.load(tile_starts_ptr + global_tile_id + 1)

    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    if chunk_start >= range_end:
        return

    chunk_end = tl.minimum(chunk_start + GAUSS_CHUNK, range_end)

    tx = pid_tile_linear % tiles_x
    ty = pid_tile_linear // tiles_x
    off_x = tx * TILE_SIZE
    off_y = ty * TILE_SIZE

    px = off_x + tl.arange(0, TILE_SIZE)
    py = off_y + tl.arange(0, TILE_SIZE)

    px_mask_1d = px < W
    py_mask_1d = py < H

    px_mesh = px[None, :]
    py_mesh = py[:, None]

    gimg_ptr_base = grad_image_ptr + pid_batch * stride_b_gimg

    for g_ptr_idx in range(chunk_start, chunk_end):
        g_flat_idx = tl.load(sorted_gauss_ids_ptr + g_ptr_idx)

        ptr_mean = means_ptr + g_flat_idx * stride_n_means
        mu_x = tl.load(ptr_mean + 0)
        mu_y = tl.load(ptr_mean + 1)

        ptr_conic = conic_ptr + g_flat_idx * stride_n_conic
        A = tl.load(ptr_conic + 0)
        B_coef = tl.load(ptr_conic + 1)
        C_coef = tl.load(ptr_conic + 2)

        dx = (px_mesh + 0.5) - mu_x
        dy = (py_mesh + 0.5) - mu_y
        power = -0.5 * (A * dx * dx + 2.0 * B_coef * dx * dy + C_coef * dy * dy)
        weight = tl.exp(power)

        valid_xy = (
            py_mask_1d[:, None]
            & px_mask_1d[None, :]
        )

        grad_mu_x = tl.zeros((), dtype=tl.float32)
        grad_mu_y = tl.zeros((), dtype=tl.float32)
        grad_A = tl.zeros((), dtype=tl.float32)
        grad_B = tl.zeros((), dtype=tl.float32)
        grad_C = tl.zeros((), dtype=tl.float32)

        ptr_color_base = colors_ptr + g_flat_idx * stride_n_colors
        ptr_gcolor_base = grad_colors_ptr + g_flat_idx * stride_n_gcolors

        for c_base in range(0, C, BLOCK_C):
            c_offsets = c_base + tl.arange(0, BLOCK_C)
            c_mask_1d = c_offsets < C

            c_vals = tl.load(ptr_color_base + c_offsets, mask=c_mask_1d, other=0.0)

            gimg_pixel_off = py_mesh[:, :, None] * stride_h_gimg + px_mesh[:, :, None] * stride_w_gimg
            gimg_c_off = c_offsets[None, None, :] * stride_c_gimg
            gimg_ptrs = gimg_ptr_base + gimg_pixel_off + gimg_c_off

            read_mask = py_mask_1d[:, None, None] & px_mask_1d[None, :, None] & c_mask_1d[None, None, :]
            grad_pix = tl.load(gimg_ptrs, mask=read_mask, other=0.0)

            # dL/dcolor = sum_xy grad_out * weight
            grad_c_block = tl.sum(tl.sum(grad_pix * weight[:, :, None], axis=0), axis=0)
            tl.atomic_add(ptr_gcolor_base + c_offsets, grad_c_block, mask=c_mask_1d)

            # dL/dweight = sum_c grad_out * color
            grad_w = tl.sum(grad_pix * c_vals[None, None, :], axis=2)
            grad_w = tl.where(valid_xy, grad_w, 0.0)

            # weight = exp(power) -> dL/dpower = dL/dweight * weight
            grad_power = grad_w * weight

            grad_mu_x += tl.sum(grad_power * (A * dx + B_coef * dy))
            grad_mu_y += tl.sum(grad_power * (B_coef * dx + C_coef * dy))
            grad_A += tl.sum(grad_power * (-0.5 * dx * dx))
            grad_B += tl.sum(grad_power * (-dx * dy))
            grad_C += tl.sum(grad_power * (-0.5 * dy * dy))

        ptr_gmean = grad_means_ptr + g_flat_idx * stride_n_gmeans
        tl.atomic_add(ptr_gmean + 0, grad_mu_x)
        tl.atomic_add(ptr_gmean + 1, grad_mu_y)

        ptr_gconic = grad_conic_ptr + g_flat_idx * stride_n_gconic
        tl.atomic_add(ptr_gconic + 0, grad_A)
        tl.atomic_add(ptr_gconic + 1, grad_B)
        tl.atomic_add(ptr_gconic + 2, grad_C)


@triton.jit
def render_color_adjoint_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    conic_ptr,
    grad_image_ptr,
    grad_colors_ptr,
    H, W, C,
    tiles_x, tiles_y,
    stride_n_means,
    stride_n_conic,
    stride_b_gimg, stride_c_gimg, stride_h_gimg, stride_w_gimg,
    stride_n_gcolors,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_tile_linear = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    num_tiles_per_batch = tiles_x * tiles_y
    global_tile_id = pid_batch * num_tiles_per_batch + pid_tile_linear

    range_start = tl.load(tile_starts_ptr + global_tile_id)
    range_end = tl.load(tile_starts_ptr + global_tile_id + 1)

    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    if chunk_start >= range_end:
        return

    chunk_end = tl.minimum(chunk_start + GAUSS_CHUNK, range_end)

    tx = pid_tile_linear % tiles_x
    ty = pid_tile_linear // tiles_x
    off_x = tx * TILE_SIZE
    off_y = ty * TILE_SIZE

    px = off_x + tl.arange(0, TILE_SIZE)
    py = off_y + tl.arange(0, TILE_SIZE)

    px_mask_1d = px < W
    py_mask_1d = py < H

    px_mesh = px[None, :]
    py_mesh = py[:, None]
    gimg_ptr_base = grad_image_ptr + pid_batch * stride_b_gimg

    for g_ptr_idx in range(chunk_start, chunk_end):
        g_flat_idx = tl.load(sorted_gauss_ids_ptr + g_ptr_idx)

        ptr_mean = means_ptr + g_flat_idx * stride_n_means
        mu_x = tl.load(ptr_mean + 0)
        mu_y = tl.load(ptr_mean + 1)

        ptr_conic = conic_ptr + g_flat_idx * stride_n_conic
        A = tl.load(ptr_conic + 0)
        B_coef = tl.load(ptr_conic + 1)
        C_coef = tl.load(ptr_conic + 2)

        dx = (px_mesh + 0.5) - mu_x
        dy = (py_mesh + 0.5) - mu_y
        power = -0.5 * (A * dx * dx + 2.0 * B_coef * dx * dy + C_coef * dy * dy)
        weight = tl.exp(power)

        ptr_gcolor_base = grad_colors_ptr + g_flat_idx * stride_n_gcolors

        for c_base in range(0, C, BLOCK_C):
            c_offsets = c_base + tl.arange(0, BLOCK_C)
            c_mask_1d = c_offsets < C

            gimg_pixel_off = py_mesh[:, :, None] * stride_h_gimg + px_mesh[:, :, None] * stride_w_gimg
            gimg_c_off = c_offsets[None, None, :] * stride_c_gimg
            gimg_ptrs = gimg_ptr_base + gimg_pixel_off + gimg_c_off

            read_mask = py_mask_1d[:, None, None] & px_mask_1d[None, :, None] & c_mask_1d[None, None, :]
            grad_pix = tl.load(gimg_ptrs, mask=read_mask, other=0.0)
            grad_c_block = tl.sum(tl.sum(grad_pix * weight[:, :, None], axis=0), axis=0)
            tl.atomic_add(ptr_gcolor_base + c_offsets, grad_c_block, mask=c_mask_1d)

# ==========================================
# 4. PyTorch Wrapper
# ==========================================

class HybridGaussianRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gs_params, H, W, config: RenderConfig):
        gs_params = gs_params.contiguous()

        B, N, D = gs_params.shape
        C = D - 5
        device = gs_params.device

        means, conic, radii = get_conic_and_bbox(gs_params, H, W, config)
        colors = gs_params[..., 5:].contiguous()
        means = means.contiguous()
        conic = conic.contiguous()
        radii = radii.contiguous()

        tile_starts, sorted_gauss_ids, (tiles_x, tiles_y) = bin_and_sort_gaussians(
            means, radii, H, W, config.tile_size
        )

        if sorted_gauss_ids is None:
            image = torch.empty(
                (B, C, H, W),
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
            )
            image.zero_()
            ctx.save_for_backward(gs_params)
            ctx.meta = {
                "H": H, "W": W, "C": C,
                "tile_starts": None,
                "sorted_gauss_ids": None,
                "means": None,
                "conic": None,
                "colors": None,
                "tiles_x": 0,
                "tiles_y": 0,
                "config": config,
            }
            return image

        counts = tile_starts[1:] - tile_starts[:-1]
        max_interactions = int(counts.max().item())
        chunks_needed = max(1, (max_interactions + config.max_gauss_chunk - 1) // config.max_gauss_chunk)

        BLOCK_C = 32
        while BLOCK_C > C and BLOCK_C > 4:
            BLOCK_C //= 2
        BLOCK_C = triton.next_power_of_2(min(C, BLOCK_C))
        if BLOCK_C < 1:
            BLOCK_C = 1

        num_cblocks = triton.cdiv(C, BLOCK_C)
        grid = (tiles_x * tiles_y, B, num_cblocks)
        means_flat = means.view(B * N, 2).contiguous()
        conic_flat = conic.view(B * N, 3).contiguous()
        colors_flat = colors.view(B * N, C).contiguous()

        image = torch.empty(
            (B, C, H, W),
            device=device,
            dtype=torch.float32,
            memory_format=torch.channels_last,
        )
        image.zero_()

        render_forward_no_atomic_kernel[grid](
            tile_starts, sorted_gauss_ids,
            means_flat, conic_flat, colors_flat,
            image,
            H, W, C,
            tiles_x, tiles_y,
            means_flat.stride(0),
            conic_flat.stride(0),
            colors_flat.stride(0),
            image.stride(0), image.stride(1), image.stride(2), image.stride(3),
            TILE_SIZE=config.tile_size,
            GAUSS_CHUNK=config.max_gauss_chunk,
            BLOCK_C=BLOCK_C,
            CHUNKS_NEEDED=chunks_needed
        )

        ctx.save_for_backward(gs_params, means, conic, colors, tile_starts, sorted_gauss_ids)
        ctx.meta = {
            "H": H, "W": W, "C": C,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "config": config,
            "BLOCK_C": BLOCK_C,
            "chunks_needed": chunks_needed,
        }
        return image

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        meta = ctx.meta

        if meta["tile_starts"] if "tile_starts" in meta else False:
            pass

        if len(saved) == 1:
            gs_params = saved[0]
            return torch.zeros_like(gs_params), None, None, None

        gs_params, means, conic, colors, tile_starts, sorted_gauss_ids = saved
        grad_out = grad_out.to(torch.float32).contiguous(memory_format=torch.channels_last)

        B, N, D = gs_params.shape
        H = meta["H"]
        W = meta["W"]
        C = meta["C"]
        tiles_x = meta["tiles_x"]
        tiles_y = meta["tiles_y"]
        config = meta["config"]
        BLOCK_C = meta["BLOCK_C"]
        chunks_needed = meta["chunks_needed"]

        grad_means = torch.zeros_like(means, dtype=torch.float32)
        grad_conic = torch.zeros_like(conic, dtype=torch.float32)
        grad_colors = torch.zeros_like(colors, dtype=torch.float32)

        if sorted_gauss_ids is not None:
            grid = (tiles_x * tiles_y, B, chunks_needed)

            render_backward_hybrid_kernel[grid](
                tile_starts, sorted_gauss_ids,
                means, conic, colors,
                grad_out,
                grad_means, grad_conic, grad_colors,
                H, W, C,
                tiles_x, tiles_y,
                means.stride(0), means.stride(1),
                conic.stride(0), conic.stride(1),
                colors.stride(0), colors.stride(1),
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
                grad_means.stride(0), grad_means.stride(1),
                grad_conic.stride(0), grad_conic.stride(1),
                grad_colors.stride(0), grad_colors.stride(1),
                TILE_SIZE=config.tile_size,
                GAUSS_CHUNK=config.max_gauss_chunk,
                BLOCK_C=BLOCK_C
            )

        # ----------------------------------
        # 链式回传：mean/conic -> gs_params
        # gs_params = [x, y, ax, ay, theta, colors...]
        # ----------------------------------
        grad_gs = torch.zeros_like(gs_params, dtype=torch.float32)

        # means = gs_params[...,0:2] * [W, H]
        grad_gs[..., 0] = grad_means[..., 0] * W
        grad_gs[..., 1] = grad_means[..., 1] * H

        # colors 直接回传
        grad_gs[..., 5:] = grad_colors

        # axes / theta
        dtype = gs_params.dtype
        device = gs_params.device

        raw_ax = gs_params[..., 2] * W
        raw_ay = gs_params[..., 3] * H

        ax = torch.clamp(raw_ax, min=config.min_sigma_px)
        ay = torch.clamp(raw_ay, min=config.min_sigma_px)

        theta = gs_params[..., 4]
        c = torch.cos(theta)
        s = torch.sin(theta)

        ia = 1.0 / (ax ** 2)   # inv_a2
        ib = 1.0 / (ay ** 2)   # inv_b2

        gA = grad_conic[..., 0]
        gB = grad_conic[..., 1]
        gC = grad_conic[..., 2]

        # conic:
        # A = ia*c^2 + ib*s^2
        # B = (ia-ib)*c*s
        # C = ia*s^2 + ib*c^2

        grad_ia = gA * (c * c) + gB * (c * s) + gC * (s * s)
        grad_ib = gA * (s * s) - gB * (c * s) + gC * (c * c)

        dA_dtheta = 2.0 * s * c * (ib - ia)
        dB_dtheta = (ia - ib) * (c * c - s * s)
        dC_dtheta = 2.0 * s * c * (ia - ib)

        grad_theta = gA * dA_dtheta + gB * dB_dtheta + gC * dC_dtheta
        grad_gs[..., 4] = grad_theta

        # ia = 1 / ax^2, ib = 1 / ay^2
        grad_ax = grad_ia * (-2.0) / (ax ** 3)
        grad_ay = grad_ib * (-2.0) / (ay ** 3)

        # clamp 反传：仅在 raw_axis > min_sigma_px 时传梯度
        mask_ax = (raw_ax > config.min_sigma_px).to(torch.float32)
        mask_ay = (raw_ay > config.min_sigma_px).to(torch.float32)

        grad_gs[..., 2] = grad_ax * mask_ax * W
        grad_gs[..., 3] = grad_ay * mask_ay * H

        return grad_gs.to(gs_params.dtype), None, None, None


def _choose_block_c(C):
    block_c = 32
    while block_c > C and block_c > 4:
        block_c //= 2
    block_c = triton.next_power_of_2(min(C, block_c))
    return max(1, block_c)


def _build_render_cache(gs_params, H, W, config):
    gs_params = gs_params.contiguous()
    B, N, _ = gs_params.shape
    means, conic, radii = get_conic_and_bbox(gs_params, H, W, config)
    means = means.contiguous()
    conic = conic.contiguous()
    tile_starts, sorted_gauss_ids, (tiles_x, tiles_y) = bin_and_sort_gaussians(
        means, radii, H, W, config.tile_size
    )
    if sorted_gauss_ids is None:
        chunks_needed = 1
    else:
        counts = tile_starts[1:] - tile_starts[:-1]
        max_interactions = int(counts.max().item())
        chunks_needed = max(1, (max_interactions + config.max_gauss_chunk - 1) // config.max_gauss_chunk)
    return {
        "B": B,
        "N": N,
        "H": H,
        "W": W,
        "means": means,
        "conic": conic,
        "tile_starts": tile_starts,
        "sorted_gauss_ids": sorted_gauss_ids,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "chunks_needed": chunks_needed,
        "config": config,
    }


def _render_from_cache(cache, colors):
    colors = colors.contiguous()
    B, N, C = colors.shape
    H = cache["H"]
    W = cache["W"]
    device = colors.device
    image = torch.empty(
        (B, C, H, W),
        device=device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    image.zero_()
    sorted_gauss_ids = cache["sorted_gauss_ids"]
    if sorted_gauss_ids is None:
        return image

    means = cache["means"]
    conic = cache["conic"]
    tile_starts = cache["tile_starts"]
    tiles_x = cache["tiles_x"]
    tiles_y = cache["tiles_y"]
    config = cache["config"]
    chunks_needed = cache["chunks_needed"]
    block_c = _choose_block_c(C)
    num_cblocks = triton.cdiv(C, block_c)
    grid = (tiles_x * tiles_y, B, num_cblocks)
    means_flat = means.view(B * N, 2).contiguous()
    conic_flat = conic.view(B * N, 3).contiguous()
    colors_flat = colors.view(B * N, C).contiguous()

    render_forward_no_atomic_kernel[grid](
        tile_starts, sorted_gauss_ids,
        means_flat, conic_flat, colors_flat,
        image,
        H, W, C,
        tiles_x, tiles_y,
        means_flat.stride(0),
        conic_flat.stride(0),
        colors_flat.stride(0),
        image.stride(0), image.stride(1), image.stride(2), image.stride(3),
        TILE_SIZE=config.tile_size,
        GAUSS_CHUNK=config.max_gauss_chunk,
        BLOCK_C=block_c,
        CHUNKS_NEEDED=chunks_needed,
    )
    return image


def _color_adjoint_from_cache(cache, grad_image):
    grad_image = grad_image.to(torch.float32).contiguous(memory_format=torch.channels_last)
    B, C, H, W = grad_image.shape
    N = cache["N"]
    grad_colors = torch.zeros((B, N, C), device=grad_image.device, dtype=torch.float32)
    sorted_gauss_ids = cache["sorted_gauss_ids"]
    if sorted_gauss_ids is None:
        return grad_colors

    means = cache["means"]
    conic = cache["conic"]
    tile_starts = cache["tile_starts"]
    tiles_x = cache["tiles_x"]
    tiles_y = cache["tiles_y"]
    config = cache["config"]
    chunks_needed = cache["chunks_needed"]
    block_c = _choose_block_c(C)
    grid = (tiles_x * tiles_y, B, chunks_needed)

    render_color_adjoint_kernel[grid](
        tile_starts, sorted_gauss_ids,
        means, conic,
        grad_image,
        grad_colors,
        H, W, C,
        tiles_x, tiles_y,
        means.stride(1),
        conic.stride(1),
        grad_image.stride(0), grad_image.stride(1), grad_image.stride(2), grad_image.stride(3),
        grad_colors.stride(1),
        TILE_SIZE=config.tile_size,
        GAUSS_CHUNK=config.max_gauss_chunk,
        BLOCK_C=block_c,
    )
    return grad_colors


class GaussianRenderer(nn.Module):
    def __init__(self, H=800, W=800, **kwargs):
        super().__init__()
        self.default_H = H
        self.default_W = W
        self.cfg = RenderConfig(**kwargs)

    def forward(self, gs_params, H=None, W=None, **kwargs):
        H = H if H is not None else self.default_H
        W = W if W is not None else self.default_W

        current_cfg = self.cfg
        if kwargs:
            current_cfg = RenderConfig(**asdict(self.cfg))
            current_cfg.update(**kwargs)

        return HybridGaussianRenderFunction.apply(gs_params, H, W, current_cfg)

    def build_cache(self, gs_params, H=None, W=None, **kwargs):
        H = H if H is not None else self.default_H
        W = W if W is not None else self.default_W

        current_cfg = self.cfg
        if kwargs:
            current_cfg = RenderConfig(**asdict(self.cfg))
            current_cfg.update(**kwargs)
        return _build_render_cache(gs_params, H, W, current_cfg)

    def render_cached(self, cache, colors):
        return _render_from_cache(cache, colors)

    def adjoint_cached(self, cache, grad_image):
        return _color_adjoint_from_cache(cache, grad_image)
