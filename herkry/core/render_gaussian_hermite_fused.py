"""Fused normalized tensor-product Gaussian-Hermite rendering."""

import torch
import triton
import triton.language as tl

MAX_BRANE_DEGREE = 3


def hermite_mode_count(degree):
    degree = int(degree)
    if degree < 0 or degree > MAX_BRANE_DEGREE:
        raise ValueError(f"Hermite degree must be in [0, {MAX_BRANE_DEGREE}]")
    return (degree + 1) ** 2


def active_high_mode_indices(max_degree, degree, device=None):
    """Indices selecting an active degree square from max-degree high modes."""
    max_degree = int(max_degree)
    degree = int(degree)
    if degree < 0 or degree > max_degree:
        raise ValueError("Active Hermite degree must not exceed max degree")
    hermite_mode_count(max_degree)
    indices = []
    width = max_degree + 1
    for n in range(degree + 1):
        for m in range(degree + 1):
            if n == 0 and m == 0:
                continue
            indices.append(n * width + m - 1)
    return torch.tensor(indices, device=device, dtype=torch.long)


def pack_active_high_modes(coefficients, max_degree, degree):
    """Pack row-major non-zero max-degree modes for an active degree."""
    degree = int(degree)
    expected = hermite_mode_count(int(max_degree)) - 1
    if coefficients.shape[-1] != expected:
        raise ValueError(f"Expected {expected} max-degree high modes, got {coefficients.shape[-1]}")
    if degree == 0:
        return coefficients[..., :0].contiguous()
    indices = active_high_mode_indices(max_degree, degree, device=coefficients.device)
    return coefficients.index_select(-1, indices).contiguous()


def brane_degree_abs_mean(coefficients, max_degree):
    """Return mean absolute coefficient magnitude for degree shells 1..D."""
    max_degree = int(max_degree)
    if max_degree == 0:
        return coefficients.new_zeros(coefficients.shape[0], 0)
    values = []
    previous = set()
    for degree in range(1, max_degree + 1):
        active = set(active_high_mode_indices(max_degree, degree).cpu().tolist())
        shell = sorted(active - previous)
        previous = active
        index = torch.tensor(shell, device=coefficients.device)
        selected = coefficients.index_select(-1, index)
        values.append(selected.abs().flatten(1).mean(1))
    return torch.stack(values, dim=1)


def augment_hermite_cache(cache, geometry):
    """Add inverse local axes and rotation to a standard Gaussian cache."""
    if "hermite_frames" in cache:
        return cache
    height = cache["H"]
    width = cache["W"]
    minimum = float(cache["config"].min_sigma_px)
    detached = geometry.detach()
    axis_x = (detached[..., 2] * float(width)).clamp_min(minimum)
    axis_y = (detached[..., 3] * float(height)).clamp_min(minimum)
    theta = detached[..., 4]
    cache["hermite_frames"] = (
        torch.stack([axis_x.reciprocal(), axis_y.reciprocal(), theta.cos(), theta.sin()], dim=-1)
        .reshape(-1, 4)
        .contiguous()
    )
    return cache


@triton.jit
def _normalized_hermite(z, DEGREE: tl.constexpr):
    if DEGREE == 0:
        return z * 0.0 + 1.0
    if DEGREE == 1:
        return 1.4142135623730951 * z
    if DEGREE == 2:
        return 1.4142135623730951 * z * z - 0.7071067811865476
    return 1.1547005383792517 * z * z * z - 1.7320508075688772 * z


@triton.jit
def _normalized_hermite_derivative(z, DEGREE: tl.constexpr):
    if DEGREE == 0:
        return z * 0.0
    if DEGREE == 1:
        return z * 0.0 + 1.4142135623730951
    if DEGREE == 2:
        return 2.8284271247461903 * z
    return 3.4641016151377544 * z * z - 1.7320508075688772


@triton.jit
def _hermite_high_forward_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    image_ptr,
    H,
    W,
    C,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_coefficients,
    stride_c_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_cblock = tl.program_id(2)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    channels = pid_cblock * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < C
    acc = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)
    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = px_mesh + 0.5 - mu_x
            dy = py_mesh + 0.5 - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
            coefficient_base = coefficients_ptr + gid * stride_n_coefficients
            for n in tl.static_range(0, DEGREE + 1):
                hx = _normalized_hermite(u, n)
                for m in tl.static_range(0, DEGREE + 1):
                    if n != 0 or m != 0:
                        mode_index = n * (DEGREE + 1) + m - 1
                        values = tl.load(
                            coefficient_base
                            + channels * stride_c_coefficients
                            + mode_index * stride_m_coefficients,
                            mask=valid_g & channel_mask,
                            other=0.0,
                        )
                        hy = _normalized_hermite(v, m)
                        damping = 1.0 / (n + m + 1.0)
                        basis = gaussian * hx * hy * damping
                        acc += basis[:, :, None] * values[None, None, :]
    output_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    mask = (py[:, None, None] < H) & (px[None, :, None] < W) & channel_mask[None, None, :]
    tl.store(output_ptrs, acc, mask=mask)


@triton.jit
def _hermite_high_shell_forward_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    image_ptr,
    base_image_ptr,
    H,
    W,
    C,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_coefficients,
    stride_c_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_b_base,
    stride_c_base,
    stride_h_base,
    stride_w_base,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CNN_READY: tl.constexpr,
):
    """Render all max(m,n) shells in one Gaussian/tile traversal."""
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_cblock = tl.program_id(2)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    channels = pid_cblock * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < C
    acc_1 = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)
    acc_2 = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)
    acc_3 = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)
    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = px_mesh + 0.5 - mu_x
            dy = py_mesh + 0.5 - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
            coefficient_base = coefficients_ptr + gid * stride_n_coefficients
            for n in tl.static_range(0, DEGREE + 1):
                hx = _normalized_hermite(u, n)
                for m in tl.static_range(0, DEGREE + 1):
                    if n != 0 or m != 0:
                        mode_index = n * (DEGREE + 1) + m - 1
                        values = tl.load(
                            coefficient_base
                            + channels * stride_c_coefficients
                            + mode_index * stride_m_coefficients,
                            mask=valid_g & channel_mask,
                            other=0.0,
                        )
                        hy = _normalized_hermite(v, m)
                        damping = 1.0 / (n + m + 1.0)
                        contribution = (
                            gaussian[:, :, None]
                            * hx[:, :, None]
                            * hy[:, :, None]
                            * damping
                            * values[None, None, :]
                        )
                        if n <= 1 and m <= 1:
                            acc_1 += contribution
                        elif n <= 2 and m <= 2:
                            acc_2 += contribution
                        else:
                            acc_3 += contribution
    pixel_mask = (py[:, None, None] < H) & (px[None, :, None] < W) & channel_mask[None, None, :]
    base_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    if CNN_READY:
        base_input_ptrs = (
            base_image_ptr
            + pid_batch * stride_b_base
            + channels[None, None, :] * stride_c_base
            + py_mesh[:, :, None] * stride_h_base
            + px_mesh[:, :, None] * stride_w_base
        )
        base_values = tl.load(base_input_ptrs, mask=pixel_mask, other=0.0)
        composite = base_values + acc_1
        if DEGREE >= 2:
            composite += acc_2
        if DEGREE >= 3:
            composite += acc_3
        tl.store(base_ptrs, composite, mask=pixel_mask)
        tl.store(base_ptrs + C * stride_c_image, acc_1, mask=pixel_mask)
        if DEGREE >= 2:
            tl.store(base_ptrs + 2 * C * stride_c_image, acc_2, mask=pixel_mask)
        if DEGREE >= 3:
            tl.store(base_ptrs + 3 * C * stride_c_image, acc_3, mask=pixel_mask)
    else:
        tl.store(base_ptrs, acc_1, mask=pixel_mask)
        if DEGREE >= 2:
            tl.store(base_ptrs + C * stride_c_image, acc_2, mask=pixel_mask)
        if DEGREE >= 3:
            tl.store(base_ptrs + 2 * C * stride_c_image, acc_3, mask=pixel_mask)


@triton.jit
def _hermite_high_adjoint_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    image_ptr,
    coefficient_grad_ptr,
    H,
    W,
    C,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    stride_c_coefficient_grad,
    stride_m_coefficient_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < C
    pixel_mask = (py[:, None, None] < H) & (px[None, :, None] < W) & channel_mask[None, None, :]
    image_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    image_values = tl.load(image_ptrs, mask=pixel_mask, other=0.0)
    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
        dx = px_mesh + 0.5 - mu_x
        dy = py_mesh + 0.5 - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
        grad_base = coefficient_grad_ptr + gid * stride_n_coefficient_grad
        for n in tl.static_range(0, DEGREE + 1):
            hx = _normalized_hermite(u, n)
            for m in tl.static_range(0, DEGREE + 1):
                if n != 0 or m != 0:
                    mode_index = n * (DEGREE + 1) + m - 1
                    hy = _normalized_hermite(v, m)
                    damping = 1.0 / (n + m + 1.0)
                    basis = gaussian * hx * hy * damping
                    gradients = tl.sum(tl.sum(image_values * basis[:, :, None], axis=0), axis=0)
                    tl.atomic_add(
                        grad_base
                        + channels * stride_c_coefficient_grad
                        + mode_index * stride_m_coefficient_grad,
                        gradients,
                        mask=valid_g & channel_mask,
                    )


@triton.jit
def _hermite_high_full_adjoint_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    image_ptr,
    coefficient_grad_ptr,
    geometry_grad_ptr,
    H,
    W,
    C,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_coefficients,
    stride_c_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    stride_c_coefficient_grad,
    stride_m_coefficient_grad,
    stride_n_geometry_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Coefficient transpose plus analytic geometry VJP in one tile pass."""
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < C
    pixel_mask = (py[:, None, None] < H) & (px[None, :, None] < W) & channel_mask[None, None, :]
    image_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    image_values = tl.load(image_ptrs, mask=pixel_mask, other=0.0)
    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
        dx = px_mesh + 0.5 - mu_x
        dy = py_mesh + 0.5 - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
        coefficient_base = coefficients_ptr + gid * stride_n_coefficients
        coefficient_grad_base = coefficient_grad_ptr + gid * stride_n_coefficient_grad
        weighted_du = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
        weighted_dv = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
        for n in tl.static_range(0, DEGREE + 1):
            hx = _normalized_hermite(u, n)
            dhx = _normalized_hermite_derivative(u, n)
            for m in tl.static_range(0, DEGREE + 1):
                if n != 0 or m != 0:
                    mode_index = n * (DEGREE + 1) + m - 1
                    hy = _normalized_hermite(v, m)
                    dhy = _normalized_hermite_derivative(v, m)
                    damping = 1.0 / (n + m + 1.0)
                    basis = gaussian * hx * hy * damping
                    coefficient = tl.load(
                        coefficient_base
                        + channels * stride_c_coefficients
                        + mode_index * stride_m_coefficients,
                        mask=valid_g & channel_mask,
                        other=0.0,
                    )
                    gradients = tl.sum(tl.sum(image_values * basis[:, :, None], axis=0), axis=0)
                    tl.atomic_add(
                        coefficient_grad_base
                        + channels * stride_c_coefficient_grad
                        + mode_index * stride_m_coefficient_grad,
                        gradients,
                        mask=valid_g & channel_mask,
                    )
                    projected_probe = tl.sum(image_values * coefficient[None, None, :], axis=2)
                    common = projected_probe * gaussian * damping
                    weighted_du += common * hy * (dhx - u * hx)
                    weighted_dv += common * hx * (dhy - v * hy)
        grad_x = (
            tl.sum(
                tl.sum(weighted_du * (-cosine * inv_ax) + weighted_dv * (sine * inv_ay), axis=0),
                axis=0,
            )
            * W
        )
        grad_y = (
            tl.sum(
                tl.sum(weighted_du * (-sine * inv_ax) + weighted_dv * (-cosine * inv_ay), axis=0),
                axis=0,
            )
            * H
        )
        grad_ax = tl.sum(tl.sum(weighted_du * (-u * inv_ax), axis=0), axis=0) * W
        grad_ay = tl.sum(tl.sum(weighted_dv * (-v * inv_ay), axis=0), axis=0) * H
        grad_theta = tl.sum(
            tl.sum(
                weighted_du * (v * inv_ax / inv_ay) + weighted_dv * (-u * inv_ay / inv_ax), axis=0
            ),
            axis=0,
        )
        geometry_grad_base = geometry_grad_ptr + gid * stride_n_geometry_grad
        tl.atomic_add(geometry_grad_base, grad_x, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 1, grad_y, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 2, grad_ax, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 3, grad_ay, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 4, grad_theta, mask=valid_g)


@triton.jit
def _hermite_high_shell_full_adjoint_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    images_ptr,
    coefficient_grad_ptr,
    geometry_grad_ptr,
    H,
    W,
    C,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_coefficients,
    stride_c_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    stride_c_coefficient_grad,
    stride_m_coefficient_grad,
    stride_n_geometry_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CBLOCKS: tl.constexpr,
):
    """VJP for a shell-major [B, DEGREE*C, H, W] fused render."""
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_combined = tl.program_id(2)
    pid_chunk = pid_combined // CBLOCKS
    pid_cblock = pid_combined % CBLOCKS
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    channels = pid_cblock * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < C
    pixel_mask = (py[:, None, None] < H) & (px[None, :, None] < W) & channel_mask[None, None, :]
    image_base_ptrs = (
        images_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    image_1 = tl.load(image_base_ptrs, mask=pixel_mask, other=0.0)
    image_2 = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)
    image_3 = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)
    if DEGREE >= 2:
        image_2 = tl.load(image_base_ptrs + C * stride_c_image, mask=pixel_mask, other=0.0)
    if DEGREE >= 3:
        image_3 = tl.load(image_base_ptrs + 2 * C * stride_c_image, mask=pixel_mask, other=0.0)
    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
        dx = px_mesh + 0.5 - mu_x
        dy = py_mesh + 0.5 - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
        coefficient_base = coefficients_ptr + gid * stride_n_coefficients
        coefficient_grad_base = coefficient_grad_ptr + gid * stride_n_coefficient_grad
        weighted_du = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
        weighted_dv = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
        for n in tl.static_range(0, DEGREE + 1):
            hx = _normalized_hermite(u, n)
            dhx = _normalized_hermite_derivative(u, n)
            for m in tl.static_range(0, DEGREE + 1):
                if n != 0 or m != 0:
                    mode_index = n * (DEGREE + 1) + m - 1
                    hy = _normalized_hermite(v, m)
                    dhy = _normalized_hermite_derivative(v, m)
                    damping = 1.0 / (n + m + 1.0)
                    basis = gaussian * hx * hy * damping
                    if n <= 1 and m <= 1:
                        image_values = image_1
                    elif n <= 2 and m <= 2:
                        image_values = image_2
                    else:
                        image_values = image_3
                    coefficient = tl.load(
                        coefficient_base
                        + channels * stride_c_coefficients
                        + mode_index * stride_m_coefficients,
                        mask=valid_g & channel_mask,
                        other=0.0,
                    )
                    gradients = tl.sum(tl.sum(image_values * basis[:, :, None], axis=0), axis=0)
                    tl.atomic_add(
                        coefficient_grad_base
                        + channels * stride_c_coefficient_grad
                        + mode_index * stride_m_coefficient_grad,
                        gradients,
                        mask=valid_g & channel_mask,
                    )
                    projected_probe = tl.sum(image_values * coefficient[None, None, :], axis=2)
                    common = projected_probe * gaussian * damping
                    weighted_du += common * hy * (dhx - u * hx)
                    weighted_dv += common * hx * (dhy - v * hy)
        grad_x = (
            tl.sum(
                tl.sum(weighted_du * (-cosine * inv_ax) + weighted_dv * (sine * inv_ay), axis=0),
                axis=0,
            )
            * W
        )
        grad_y = (
            tl.sum(
                tl.sum(weighted_du * (-sine * inv_ax) + weighted_dv * (-cosine * inv_ay), axis=0),
                axis=0,
            )
            * H
        )
        grad_ax = tl.sum(tl.sum(weighted_du * (-u * inv_ax), axis=0), axis=0) * W
        grad_ay = tl.sum(tl.sum(weighted_dv * (-v * inv_ay), axis=0), axis=0) * H
        grad_theta = tl.sum(
            tl.sum(
                weighted_du * (v * inv_ax / inv_ay) + weighted_dv * (-u * inv_ay / inv_ax), axis=0
            ),
            axis=0,
        )
        geometry_grad_base = geometry_grad_ptr + gid * stride_n_geometry_grad
        tl.atomic_add(geometry_grad_base, grad_x, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 1, grad_y, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 2, grad_ax, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 3, grad_ay, mask=valid_g)
        tl.atomic_add(geometry_grad_base + 4, grad_theta, mask=valid_g)


@triton.jit
def _hermite_scalar_forward_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    image_ptr,
    H,
    W,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    DEGREE: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    acc = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = px_mesh + 0.5 - mu_x
            dy = py_mesh + 0.5 - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
            coefficient_base = coefficients_ptr + gid * stride_n_coefficients
            for n in tl.static_range(0, DEGREE + 1):
                hx = _normalized_hermite(u, n)
                for m in tl.static_range(0, DEGREE + 1):
                    mode_index = n * (DEGREE + 1) + m
                    value = tl.load(
                        coefficient_base + mode_index * stride_m_coefficients,
                        mask=valid_g,
                        other=0.0,
                    )
                    hy = _normalized_hermite(v, m)
                    damping = 1.0 if n == 0 and m == 0 else 1.0 / (n + m + 1.0)
                    acc += gaussian * hx * hy * damping * value
    output_ptrs = (
        image_ptr + pid_batch * stride_b_image + py_mesh * stride_h_image + px_mesh * stride_w_image
    )
    mask = (py[:, None] < H) & (px[None, :] < W)
    tl.store(output_ptrs, acc, mask=mask)


@triton.jit
def _hermite_scalar_shell_forward_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    image_ptr,
    H,
    W,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    DEGREE: tl.constexpr,
):
    """Render all non-zero max(m,n) scalar shells in one tile pass."""
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    acc_1 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_2 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_3 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = px_mesh + 0.5 - mu_x
            dy = py_mesh + 0.5 - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
            coefficient_base = coefficients_ptr + gid * stride_n_coefficients
            for n in tl.static_range(0, DEGREE + 1):
                hx = _normalized_hermite(u, n)
                for m in tl.static_range(0, DEGREE + 1):
                    if n != 0 or m != 0:
                        mode_index = n * (DEGREE + 1) + m
                        value = tl.load(
                            coefficient_base + mode_index * stride_m_coefficients,
                            mask=valid_g,
                            other=0.0,
                        )
                        hy = _normalized_hermite(v, m)
                        damping = 1.0 / (n + m + 1.0)
                        contribution = gaussian * hx * hy * damping * value
                        if n <= 1 and m <= 1:
                            acc_1 += contribution
                        elif n <= 2 and m <= 2:
                            acc_2 += contribution
                        else:
                            acc_3 += contribution
    mask = (py[:, None] < H) & (px[None, :] < W)
    output_ptrs = (
        image_ptr + pid_batch * stride_b_image + py_mesh * stride_h_image + px_mesh * stride_w_image
    )
    tl.store(output_ptrs, acc_1, mask=mask)
    if DEGREE >= 2:
        tl.store(output_ptrs + stride_c_image, acc_2, mask=mask)
    if DEGREE >= 3:
        tl.store(output_ptrs + 2 * stride_c_image, acc_3, mask=mask)


@triton.jit
def _hermite_scalar_shell_trajectory_forward_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    coefficients_ptr,
    image_ptr,
    H,
    W,
    N,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_b_coefficients,
    stride_t_coefficients,
    stride_n_coefficients,
    stride_m_coefficients,
    stride_b_image,
    stride_t_image,
    stride_s_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    DEGREE: tl.constexpr,
):
    """Render three coefficient states and all shells in one tile pass."""
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    acc_01 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_02 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_03 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_41 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_42 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_43 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_101 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_102 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    acc_103 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=pid_batch * N)
            local_gid = gid - pid_batch * N
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = px_mesh + 0.5 - mu_x
            dy = py_mesh + 0.5 - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
            coefficient_base = (
                coefficients_ptr
                + pid_batch * stride_b_coefficients
                + local_gid * stride_n_coefficients
            )
            for n in tl.static_range(0, DEGREE + 1):
                hx = _normalized_hermite(u, n)
                for m in tl.static_range(0, DEGREE + 1):
                    if n != 0 or m != 0:
                        mode_index = n * (DEGREE + 1) + m
                        coefficient_ptr = coefficient_base + mode_index * stride_m_coefficients
                        value0 = tl.load(coefficient_ptr, mask=valid_g, other=0.0)
                        value4 = tl.load(
                            coefficient_ptr + stride_t_coefficients, mask=valid_g, other=0.0
                        )
                        value10 = tl.load(
                            coefficient_ptr + 2 * stride_t_coefficients, mask=valid_g, other=0.0
                        )
                        hy = _normalized_hermite(v, m)
                        basis = gaussian * hx * hy / (n + m + 1.0)
                        contribution0 = basis * value0
                        contribution4 = basis * value4
                        contribution10 = basis * value10
                        if n <= 1 and m <= 1:
                            acc_01 += contribution0
                            acc_41 += contribution4
                            acc_101 += contribution10
                        elif n <= 2 and m <= 2:
                            acc_02 += contribution0
                            acc_42 += contribution4
                            acc_102 += contribution10
                        else:
                            acc_03 += contribution0
                            acc_43 += contribution4
                            acc_103 += contribution10
    mask = (py[:, None] < H) & (px[None, :] < W)
    output_ptrs = (
        image_ptr + pid_batch * stride_b_image + py_mesh * stride_h_image + px_mesh * stride_w_image
    )
    tl.store(output_ptrs, acc_01, mask=mask)
    tl.store(output_ptrs + stride_t_image, acc_41, mask=mask)
    tl.store(output_ptrs + 2 * stride_t_image, acc_101, mask=mask)
    if DEGREE >= 2:
        tl.store(output_ptrs + stride_s_image, acc_02, mask=mask)
        tl.store(output_ptrs + stride_t_image + stride_s_image, acc_42, mask=mask)
        tl.store(output_ptrs + 2 * stride_t_image + stride_s_image, acc_102, mask=mask)
    if DEGREE >= 3:
        tl.store(output_ptrs + 2 * stride_s_image, acc_03, mask=mask)
        tl.store(output_ptrs + stride_t_image + 2 * stride_s_image, acc_43, mask=mask)
        tl.store(output_ptrs + 2 * stride_t_image + 2 * stride_s_image, acc_103, mask=mask)


@triton.jit
def _hermite_scalar_adjoint_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    image_ptr,
    coefficient_grad_ptr,
    H,
    W,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_b_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    stride_m_coefficient_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    DEGREE: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    pixel_mask = (py[:, None] < H) & (px[None, :] < W)
    image_ptrs = (
        image_ptr + pid_batch * stride_b_image + py_mesh * stride_h_image + px_mesh * stride_w_image
    )
    image_values = tl.load(image_ptrs, mask=pixel_mask, other=0.0)
    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
        dx = px_mesh + 0.5 - mu_x
        dy = py_mesh + 0.5 - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
        grad_base = coefficient_grad_ptr + gid * stride_n_coefficient_grad
        for n in tl.static_range(0, DEGREE + 1):
            hx = _normalized_hermite(u, n)
            for m in tl.static_range(0, DEGREE + 1):
                mode_index = n * (DEGREE + 1) + m
                hy = _normalized_hermite(v, m)
                damping = 1.0 if n == 0 and m == 0 else 1.0 / (n + m + 1.0)
                gradient = tl.sum(
                    tl.sum(image_values * gaussian * hx * hy * damping, axis=0), axis=0
                )
                tl.atomic_add(
                    grad_base + mode_index * stride_m_coefficient_grad, gradient, mask=valid_g
                )


@triton.jit
def _hermite_scalar_shell_adjoint_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    images_ptr,
    coefficient_grad_ptr,
    H,
    W,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    stride_m_coefficient_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    DEGREE: tl.constexpr,
):
    """Adjoint of the shell-major scalar renderer."""
    pid_tile = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    global_tile = pid_batch * tiles_x * tiles_y + pid_tile
    range_start = tl.load(tile_starts_ptr + global_tile)
    range_end = tl.load(tile_starts_ptr + global_tile + 1)
    chunk_start = range_start + pid_chunk * GAUSS_CHUNK
    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    px = tx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    py = ty * TILE_SIZE + tl.arange(0, TILE_SIZE)
    px_mesh = px[None, :]
    py_mesh = py[:, None]
    pixel_mask = (py[:, None] < H) & (px[None, :] < W)
    image_ptrs = (
        images_ptr
        + pid_batch * stride_b_image
        + py_mesh * stride_h_image
        + px_mesh * stride_w_image
    )
    image_1 = tl.load(image_ptrs, mask=pixel_mask, other=0.0)
    image_2 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    image_3 = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
    if DEGREE >= 2:
        image_2 = tl.load(image_ptrs + stride_c_image, mask=pixel_mask, other=0.0)
    if DEGREE >= 3:
        image_3 = tl.load(image_ptrs + 2 * stride_c_image, mask=pixel_mask, other=0.0)
    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(sorted_gauss_ids_ptr + interaction, mask=valid_g, other=0)
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
        dx = px_mesh + 0.5 - mu_x
        dy = py_mesh + 0.5 - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0)
        grad_base = coefficient_grad_ptr + gid * stride_n_coefficient_grad
        for n in tl.static_range(0, DEGREE + 1):
            hx = _normalized_hermite(u, n)
            for m in tl.static_range(0, DEGREE + 1):
                if n != 0 or m != 0:
                    mode_index = n * (DEGREE + 1) + m
                    hy = _normalized_hermite(v, m)
                    damping = 1.0 / (n + m + 1.0)
                    if n <= 1 and m <= 1:
                        image_values = image_1
                    elif n <= 2 and m <= 2:
                        image_values = image_2
                    else:
                        image_values = image_3
                    gradient = tl.sum(
                        tl.sum(image_values * gaussian * hx * hy * damping, axis=0), axis=0
                    )
                    tl.atomic_add(
                        grad_base + mode_index * stride_m_coefficient_grad, gradient, mask=valid_g
                    )


def _cache_tensors(cache):
    means = cache["means"].reshape(-1, 2).contiguous()
    return (cache["tile_starts"], cache["sorted_gauss_ids"], means, cache["hermite_frames"])


def hermite_high_render_cached(cache, coefficients, degree):
    degree = int(degree)
    expected_modes = hermite_mode_count(degree) - 1
    if degree < 1 or coefficients.shape[-1] != expected_modes:
        raise ValueError(f"Degree {degree} expects {expected_modes} high modes")
    coefficients = coefficients.contiguous()
    (batch, _, channels, _) = coefficients.shape
    image = torch.empty(
        (batch, channels, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, channels, expected_modes)
    block_c = min(8, triton.next_power_of_2(channels))
    grid = (cache["tiles_x"] * cache["tiles_y"], batch, triton.cdiv(channels, block_c))
    _hermite_high_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        image,
        cache["H"],
        cache["W"],
        channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        coefficients_flat.stride(2),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        DEGREE=degree,
        BLOCK_C=block_c,
    )
    return image


def hermite_high_shell_cnn_render_cached(cache, coefficients, degree, base_features, output_degree):
    """Render [base+sum(shells), shell1,...] in CNN input layout."""
    degree = int(degree)
    output_degree = int(output_degree)
    expected_modes = hermite_mode_count(degree) - 1
    if degree < 1 or coefficients.shape[-1] != expected_modes:
        raise ValueError(f"Degree {degree} expects {expected_modes} high modes")
    coefficients = coefficients.contiguous()
    base_features = base_features.contiguous(memory_format=torch.channels_last)
    (batch, _, channels, _) = coefficients.shape
    if base_features.shape[:2] != (batch, channels):
        raise ValueError("Base feature shape does not match coefficients")
    if output_degree < degree:
        raise ValueError("CNN output degree cannot be smaller than degree")
    image = torch.empty(
        (batch, (output_degree + 1) * channels, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        image[:, :channels].copy_(base_features)
        return image
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, channels, expected_modes)
    block_c = min(4 if degree == 3 else 8, triton.next_power_of_2(channels))
    grid = (cache["tiles_x"] * cache["tiles_y"], batch, triton.cdiv(channels, block_c))
    _hermite_high_shell_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        image,
        base_features,
        cache["H"],
        cache["W"],
        channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        coefficients_flat.stride(2),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        base_features.stride(0),
        base_features.stride(1),
        base_features.stride(2),
        base_features.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        DEGREE=degree,
        BLOCK_C=block_c,
        CNN_READY=True,
    )
    return image


def hermite_high_shell_render_cached(cache, coefficients, degree):
    """Render [shell1,...,shellD] as shell-major feature channels."""
    degree = int(degree)
    expected_modes = hermite_mode_count(degree) - 1
    if degree < 1 or coefficients.shape[-1] != expected_modes:
        raise ValueError(f"Degree {degree} expects {expected_modes} high modes")
    coefficients = coefficients.contiguous()
    (batch, _, channels, _) = coefficients.shape
    image = torch.empty(
        (batch, degree * channels, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, channels, expected_modes)
    block_c = min(4 if degree == 3 else 8, triton.next_power_of_2(channels))
    grid = (cache["tiles_x"] * cache["tiles_y"], batch, triton.cdiv(channels, block_c))
    _hermite_high_shell_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        image,
        image,
        cache["H"],
        cache["W"],
        channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        coefficients_flat.stride(2),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        DEGREE=degree,
        BLOCK_C=block_c,
        CNN_READY=False,
    )
    return image


def hermite_high_adjoint_cached(cache, images, degree):
    degree = int(degree)
    expected_modes = hermite_mode_count(degree) - 1
    images = images.to(torch.float32).contiguous(memory_format=torch.channels_last)
    (batch, channels, _, _) = images.shape
    gradients = torch.empty(
        (cache["B"], cache["N"], channels, expected_modes),
        device=images.device,
        dtype=torch.float32,
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return gradients
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    gradients_flat = gradients.reshape(-1, channels, expected_modes)
    block_c = min(8, triton.next_power_of_2(channels))
    grid = (cache["tiles_x"] * cache["tiles_y"], batch, cache["chunks_needed"])
    _hermite_high_adjoint_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        images,
        gradients_flat,
        cache["H"],
        cache["W"],
        channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        images.stride(0),
        images.stride(1),
        images.stride(2),
        images.stride(3),
        gradients_flat.stride(0),
        gradients_flat.stride(1),
        gradients_flat.stride(2),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        DEGREE=degree,
        BLOCK_C=block_c,
    )
    return gradients


def hermite_high_full_adjoint_cached(cache, coefficients, images, degree, geometry_shape):
    """Return coefficient and geometry VJPs for the learned Brane branch."""
    degree = int(degree)
    expected_modes = hermite_mode_count(degree) - 1
    coefficients = coefficients.contiguous()
    images = images.to(torch.float32).contiguous(memory_format=torch.channels_last)
    (batch, channels, _, _) = images.shape
    coefficient_gradients = torch.empty_like(coefficients, dtype=torch.float32).zero_()
    geometry_gradients = torch.empty(
        geometry_shape, device=images.device, dtype=torch.float32
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return (coefficient_gradients, geometry_gradients)
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, channels, expected_modes)
    coefficient_gradients_flat = coefficient_gradients.reshape(-1, channels, expected_modes)
    geometry_gradients_flat = geometry_gradients.reshape(-1, 5)
    block_c = min(8, triton.next_power_of_2(channels))
    grid = (cache["tiles_x"] * cache["tiles_y"], batch, cache["chunks_needed"])
    _hermite_high_full_adjoint_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        images,
        coefficient_gradients_flat,
        geometry_gradients_flat,
        cache["H"],
        cache["W"],
        channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        coefficients_flat.stride(2),
        images.stride(0),
        images.stride(1),
        images.stride(2),
        images.stride(3),
        coefficient_gradients_flat.stride(0),
        coefficient_gradients_flat.stride(1),
        coefficient_gradients_flat.stride(2),
        geometry_gradients_flat.stride(0),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        DEGREE=degree,
        BLOCK_C=block_c,
    )
    return (coefficient_gradients, geometry_gradients)


def hermite_high_shell_full_adjoint_cached(cache, coefficients, images, degree, geometry_shape):
    """VJP for the single-pass high-feature shell renderer."""
    degree = int(degree)
    expected_modes = hermite_mode_count(degree) - 1
    coefficients = coefficients.contiguous()
    images = images.to(torch.float32).contiguous(memory_format=torch.channels_last)
    (batch, shell_channels, _, _) = images.shape
    if shell_channels % degree != 0:
        raise ValueError("Shell feature channels must be divisible by degree")
    channels = shell_channels // degree
    coefficient_gradients = torch.empty_like(coefficients, dtype=torch.float32).zero_()
    geometry_gradients = torch.empty(
        geometry_shape, device=images.device, dtype=torch.float32
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return (coefficient_gradients, geometry_gradients)
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, channels, expected_modes)
    coefficient_gradients_flat = coefficient_gradients.reshape(-1, channels, expected_modes)
    geometry_gradients_flat = geometry_gradients.reshape(-1, 5)
    block_c = min(4 if degree == 3 else 8, triton.next_power_of_2(channels))
    cblocks = triton.cdiv(channels, block_c)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch, cache["chunks_needed"] * cblocks)
    _hermite_high_shell_full_adjoint_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        images,
        coefficient_gradients_flat,
        geometry_gradients_flat,
        cache["H"],
        cache["W"],
        channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        coefficients_flat.stride(2),
        images.stride(0),
        images.stride(1),
        images.stride(2),
        images.stride(3),
        coefficient_gradients_flat.stride(0),
        coefficient_gradients_flat.stride(1),
        coefficient_gradients_flat.stride(2),
        geometry_gradients_flat.stride(0),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        DEGREE=degree,
        BLOCK_C=block_c,
        CBLOCKS=cblocks,
    )
    return (coefficient_gradients, geometry_gradients)


def hermite_scalar_render_cached(cache, coefficients, degree):
    degree = int(degree)
    modes = hermite_mode_count(degree)
    if coefficients.shape[-1] != modes:
        raise ValueError(
            f"Degree {degree} expects {modes} scalar modes, got shape={tuple(coefficients.shape)}"
        )
    coefficients = coefficients.contiguous()
    batch = coefficients.shape[0]
    image = torch.empty(
        (batch, 1, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, modes)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch)
    _hermite_scalar_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        image,
        cache["H"],
        cache["W"],
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        image.stride(0),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        DEGREE=degree,
    )
    return image


def hermite_scalar_shell_render_cached(cache, coefficients, degree):
    """Render non-zero scalar shells as [B, D, H, W] in one pass."""
    degree = int(degree)
    modes = hermite_mode_count(degree)
    if degree < 1 or coefficients.shape[-1] != modes:
        raise ValueError(f"Degree {degree} expects {modes} scalar modes")
    coefficients = coefficients.contiguous()
    batch = coefficients.shape[0]
    image = torch.empty(
        (batch, degree, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, modes)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch)
    _hermite_scalar_shell_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients_flat,
        image,
        cache["H"],
        cache["W"],
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients_flat.stride(0),
        coefficients_flat.stride(1),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        DEGREE=degree,
    )
    return image


def hermite_scalar_shell_trajectory_render_cached(cache, coefficients, degree):
    """Render three states as [B,3,D,H,W] in one Triton launch."""
    degree = int(degree)
    modes = hermite_mode_count(degree)
    if coefficients.ndim != 4 or coefficients.shape[1] != 3:
        raise ValueError("Shell trajectory coefficients must have shape [B,3,N,M]")
    if degree < 1 or coefficients.shape[-1] != modes:
        raise ValueError(f"Degree {degree} expects {modes} scalar modes")
    coefficients = coefficients.contiguous()
    (batch, _, gaussian_count, _) = coefficients.shape
    if gaussian_count != cache["N"]:
        raise ValueError("Coefficient lattice does not match render cache")
    image = torch.empty(
        (batch, 3, degree, cache["H"], cache["W"]), device=coefficients.device, dtype=torch.float32
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch)
    _hermite_scalar_shell_trajectory_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        coefficients,
        image,
        cache["H"],
        cache["W"],
        cache["N"],
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        coefficients.stride(0),
        coefficients.stride(1),
        coefficients.stride(2),
        coefficients.stride(3),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        image.stride(4),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        DEGREE=degree,
    )
    return image


def hermite_scalar_adjoint_cached(cache, image, degree):
    degree = int(degree)
    modes = hermite_mode_count(degree)
    if image.shape[1] != 1:
        raise ValueError("Hermite scalar adjoint expects one image channel")
    image = image.to(torch.float32).contiguous(memory_format=torch.channels_last)
    gradients = torch.empty(
        (cache["B"], cache["N"], modes), device=image.device, dtype=torch.float32
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return gradients
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    gradients_flat = gradients.reshape(-1, modes)
    grid = (cache["tiles_x"] * cache["tiles_y"], cache["B"], cache["chunks_needed"])
    _hermite_scalar_adjoint_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        image,
        gradients_flat,
        cache["H"],
        cache["W"],
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        image.stride(0),
        image.stride(2),
        image.stride(3),
        gradients_flat.stride(0),
        gradients_flat.stride(1),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        DEGREE=degree,
    )
    return gradients


def hermite_scalar_shell_adjoint_cached(cache, images, degree):
    """Adjoint of the single-pass scalar shell renderer."""
    degree = int(degree)
    modes = hermite_mode_count(degree)
    if images.shape[1] != degree:
        raise ValueError(f"Degree {degree} expects {degree} shell images")
    images = images.to(torch.float32).contiguous(memory_format=torch.channels_last)
    gradients = torch.empty(
        (cache["B"], cache["N"], modes), device=images.device, dtype=torch.float32
    ).zero_()
    if cache["sorted_gauss_ids"] is None:
        return gradients
    (tile_starts, sorted_ids, means, frames) = _cache_tensors(cache)
    gradients_flat = gradients.reshape(-1, modes)
    grid = (cache["tiles_x"] * cache["tiles_y"], cache["B"], cache["chunks_needed"])
    _hermite_scalar_shell_adjoint_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        images,
        gradients_flat,
        cache["H"],
        cache["W"],
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        images.stride(0),
        images.stride(1),
        images.stride(2),
        images.stride(3),
        gradients_flat.stride(0),
        gradients_flat.stride(1),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        DEGREE=degree,
    )
    return gradients


class _HermiteHighRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, geometry, cache, degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        ctx.geometry_shape = geometry.shape
        ctx.save_for_backward(coefficients)
        return hermite_high_render_cached(cache, coefficients, degree)

    @staticmethod
    def backward(ctx, grad_images):
        (coefficients,) = ctx.saved_tensors
        (coefficient_gradients, geometry_gradients) = hermite_high_full_adjoint_cached(
            ctx.cache, coefficients, grad_images, ctx.degree, ctx.geometry_shape
        )
        return (coefficient_gradients, geometry_gradients, None, None)


class _HermiteHighShellRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, geometry, cache, degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        ctx.geometry_shape = geometry.shape
        ctx.save_for_backward(coefficients)
        return hermite_high_shell_render_cached(cache, coefficients, degree)

    @staticmethod
    def backward(ctx, grad_images):
        (coefficients,) = ctx.saved_tensors
        (coefficient_gradients, geometry_gradients) = hermite_high_shell_full_adjoint_cached(
            ctx.cache, coefficients, grad_images, ctx.degree, ctx.geometry_shape
        )
        return (coefficient_gradients, geometry_gradients, None, None)


class _HermiteHighShellCNNRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, geometry, base_features, cache, degree, output_degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        ctx.output_degree = int(output_degree)
        ctx.channels = int(base_features.shape[1])
        ctx.geometry_shape = geometry.shape
        ctx.save_for_backward(coefficients)
        return hermite_high_shell_cnn_render_cached(
            cache, coefficients, degree, base_features, output_degree
        )

    @staticmethod
    def backward(ctx, grad_images):
        (coefficients,) = ctx.saved_tensors
        channels = ctx.channels
        grad_base = grad_images[:, :channels]
        active_shell_gradients = grad_images[:, channels : (ctx.degree + 1) * channels].reshape(
            grad_images.shape[0], ctx.degree, channels, *grad_images.shape[2:]
        )
        active_shell_gradients = (active_shell_gradients + grad_base[:, None]).reshape(
            grad_images.shape[0], ctx.degree * channels, *grad_images.shape[2:]
        )
        (coefficient_gradients, geometry_gradients) = hermite_high_shell_full_adjoint_cached(
            ctx.cache, coefficients, active_shell_gradients, ctx.degree, ctx.geometry_shape
        )
        return (coefficient_gradients, geometry_gradients, grad_base, None, None, None)


class _HermiteScalarRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, cache, degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        return hermite_scalar_render_cached(cache, coefficients, degree)

    @staticmethod
    def backward(ctx, grad_image):
        return (hermite_scalar_adjoint_cached(ctx.cache, grad_image, ctx.degree), None, None)


class _HermiteScalarShellRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, cache, degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        return hermite_scalar_shell_render_cached(cache, coefficients, degree)

    @staticmethod
    def backward(ctx, grad_images):
        return (hermite_scalar_shell_adjoint_cached(ctx.cache, grad_images, ctx.degree), None, None)


class _HermiteScalarShellTrajectoryRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, cache, degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        return hermite_scalar_shell_trajectory_render_cached(cache, coefficients, degree)

    @staticmethod
    def backward(ctx, grad_images):
        coefficient_gradients = torch.stack(
            [
                hermite_scalar_shell_adjoint_cached(
                    ctx.cache, grad_images[:, state_index], ctx.degree
                )
                for state_index in range(3)
            ],
            dim=1,
        )
        return (coefficient_gradients, None, None)


class _HermiteScalarAdjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, image, cache, degree):
        ctx.cache = cache
        ctx.degree = int(degree)
        return hermite_scalar_adjoint_cached(cache, image, degree)

    @staticmethod
    def backward(ctx, grad_coefficients):
        return (hermite_scalar_render_cached(ctx.cache, grad_coefficients, ctx.degree), None, None)


def differentiable_hermite_high_render(cache, coefficients, degree, geometry=None):
    if geometry is None:
        geometry = coefficients.new_zeros(cache["B"], cache["N"], 5)
    return _HermiteHighRender.apply(coefficients, geometry, cache, int(degree))


def _square_shell_masks(degree, include_zero, device, dtype):
    """Masks for max(m,n)=k shells of the square tensor-product basis."""
    degree = int(degree)
    start = 0 if include_zero else 1
    offset = 0 if include_zero else 1
    masks = []
    for shell in range(1, degree + 1):
        values = []
        for n in range(degree + 1):
            for m in range(degree + 1):
                if n == 0 and m == 0 and (not include_zero):
                    continue
                values.append(float(max(n, m) == shell))
        masks.append(torch.tensor(values, device=device, dtype=dtype))
    if not masks:
        mode_count = (degree + 1) ** 2 - offset
        return torch.empty(0, mode_count, device=device, dtype=dtype)
    return torch.stack(masks, dim=0)


def reference_differentiable_hermite_high_shell_render(cache, coefficients, degree, geometry=None):
    """Render high-order max(m,n) shells as [B,D*C,H,W]."""
    degree = int(degree)
    if degree < 1:
        return coefficients.new_zeros(
            coefficients.shape[0], 0, cache["H"], cache["W"], dtype=torch.float32
        )
    masks = _square_shell_masks(
        degree, include_zero=False, device=coefficients.device, dtype=coefficients.dtype
    )
    rendered = []
    for mask in masks.unbind(dim=0):
        rendered.append(
            differentiable_hermite_high_render(
                cache, coefficients * mask.view(1, 1, 1, -1), degree, geometry
            )
        )
    return torch.cat(rendered, dim=1)


def differentiable_hermite_high_shell_render(cache, coefficients, degree, geometry=None):
    """Single-pass shell-major high-feature render."""
    if geometry is None:
        geometry = coefficients.new_zeros(cache["B"], cache["N"], 5)
    return _HermiteHighShellRender.apply(coefficients, geometry, cache, int(degree))


def differentiable_hermite_high_shell_cnn_render(
    cache, coefficients, degree, base_features, output_degree, geometry=None
):
    """Render full+shell features directly in the CNN input layout."""
    if geometry is None:
        geometry = coefficients.new_zeros(cache["B"], cache["N"], 5)
    return _HermiteHighShellCNNRender.apply(
        coefficients, geometry, base_features, cache, int(degree), int(output_degree)
    )


def reference_differentiable_hermite_scalar_shell_render(cache, coefficients, degree):
    """Render non-zero max(m,n) shells as [B,D,H,W]."""
    degree = int(degree)
    if degree < 1:
        return coefficients.new_zeros(
            coefficients.shape[0], 0, cache["H"], cache["W"], dtype=torch.float32
        )
    masks = _square_shell_masks(
        degree, include_zero=True, device=coefficients.device, dtype=coefficients.dtype
    )
    rendered = []
    for mask in masks.unbind(dim=0):
        rendered.append(
            differentiable_hermite_scalar_render(cache, coefficients * mask.view(1, 1, -1), degree)
        )
    return torch.cat(rendered, dim=1)


def differentiable_hermite_scalar_shell_render(cache, coefficients, degree):
    """Single-pass shell-major scalar render."""
    degree = int(degree)
    if degree < 1:
        return coefficients.new_zeros(
            coefficients.shape[0], 0, cache["H"], cache["W"], dtype=torch.float32
        )
    return _HermiteScalarShellRender.apply(coefficients, cache, degree)


def differentiable_hermite_scalar_shell_trajectory_render(cache, coefficients, degree):
    """Render three trajectory states and all shells in one pass."""
    degree = int(degree)
    if degree < 1:
        return coefficients.new_zeros(
            coefficients.shape[0], 3, 0, cache["H"], cache["W"], dtype=torch.float32
        )
    return _HermiteScalarShellTrajectoryRender.apply(coefficients, cache, degree)


def differentiable_hermite_scalar_render(cache, coefficients, degree):
    return _HermiteScalarRender.apply(coefficients, cache, int(degree))


def differentiable_hermite_scalar_adjoint(cache, image, degree):
    return _HermiteScalarAdjoint.apply(image, cache, int(degree))
