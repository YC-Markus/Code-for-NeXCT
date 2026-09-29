"""Fused analytic Gaussian-derivative forward and coefficient adjoint.

For fixed geometry this module evaluates local, scale-normalized Hermite
bases directly while visiting each Gaussian/pixel interaction.  It therefore
does not materialize G/Gx/Gy/Gxx/Gxy/Gyy images and does not use grouped
convolutions.  The forward and coefficient adjoint kernels share exactly the
same basis and three-sigma tile envelope.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _derivative_forward_kernel(
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
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    ORDER: tl.constexpr,
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
            gid = tl.load(
                sorted_gauss_ids_ptr + interaction,
                mask=valid_g,
                other=0,
            )
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)

            dx = (px_mesh + 0.5) - mu_x
            dy = (py_mesh + 0.5) - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(
                valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0
            )

            coefficient_ptr = coefficients_ptr + gid * stride_n_coefficients
            polynomial = tl.load(
                coefficient_ptr, mask=valid_g, other=0.0
            )
            if ORDER >= 1:
                gx = tl.load(
                    coefficient_ptr + 1, mask=valid_g, other=0.0
                )
                gy = tl.load(
                    coefficient_ptr + 2, mask=valid_g, other=0.0
                )
                polynomial += -gx * u - gy * v
            if ORDER >= 2:
                gxx = tl.load(
                    coefficient_ptr + 3, mask=valid_g, other=0.0
                )
                gxy = tl.load(
                    coefficient_ptr + 4, mask=valid_g, other=0.0
                )
                gyy = tl.load(
                    coefficient_ptr + 5, mask=valid_g, other=0.0
                )
                polynomial += (
                    gxx * (u * u - 1.0)
                    + gxy * u * v
                    + gyy * (v * v - 1.0)
                )
            acc += gaussian * polynomial

    output_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + py_mesh * stride_h_image
        + px_mesh * stride_w_image
    )
    mask = (py[:, None] < H) & (px[None, :] < W)
    tl.store(output_ptrs, acc, mask=mask)


@triton.jit
def _derivative_adjoint_kernel(
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
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    ORDER: tl.constexpr,
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
        image_ptr
        + pid_batch * stride_b_image
        + py_mesh * stride_h_image
        + px_mesh * stride_w_image
    )
    image_values = tl.load(image_ptrs, mask=pixel_mask, other=0.0)

    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(
            sorted_gauss_ids_ptr + interaction,
            mask=valid_g,
            other=0,
        )
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)

        dx = (px_mesh + 0.5) - mu_x
        dy = (py_mesh + 0.5) - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(
            valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0
        )
        weighted_grad = gaussian * image_values
        grad_ptr = coefficient_grad_ptr + gid * stride_n_coefficient_grad
        grad_g = tl.sum(tl.sum(weighted_grad, axis=0), axis=0)
        tl.atomic_add(grad_ptr, grad_g, mask=valid_g)
        if ORDER >= 1:
            grad_gx = tl.sum(tl.sum(-u * weighted_grad, axis=0), axis=0)
            grad_gy = tl.sum(tl.sum(-v * weighted_grad, axis=0), axis=0)
            tl.atomic_add(grad_ptr + 1, grad_gx, mask=valid_g)
            tl.atomic_add(grad_ptr + 2, grad_gy, mask=valid_g)
        if ORDER >= 2:
            grad_gxx = tl.sum(
                tl.sum((u * u - 1.0) * weighted_grad, axis=0), axis=0
            )
            grad_gxy = tl.sum(tl.sum(u * v * weighted_grad, axis=0), axis=0)
            grad_gyy = tl.sum(
                tl.sum((v * v - 1.0) * weighted_grad, axis=0), axis=0
            )
            tl.atomic_add(grad_ptr + 3, grad_gxx, mask=valid_g)
            tl.atomic_add(grad_ptr + 4, grad_gxy, mask=valid_g)
            tl.atomic_add(grad_ptr + 5, grad_gyy, mask=valid_g)


@triton.jit
def _basis_bank_forward_kernel(
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
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK_C: tl.constexpr,
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
    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < 5
    active_mask = channels < (2 + 3 * (ORDER >= 2))
    acc = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)

    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(
                sorted_gauss_ids_ptr + interaction,
                mask=valid_g,
                other=0,
            )
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = (px_mesh + 0.5) - mu_x
            dy = (py_mesh + 0.5) - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(
                valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0
            )
            basis = tl.where(channels[None, None, :] == 0, -u[:, :, None], 0.0)
            basis = tl.where(
                channels[None, None, :] == 1, -v[:, :, None], basis
            )
            if ORDER >= 2:
                basis = tl.where(
                    channels[None, None, :] == 2,
                    (u * u - 1.0)[:, :, None], basis,
                )
                basis = tl.where(
                    channels[None, None, :] == 3,
                    (u * v)[:, :, None], basis,
                )
                basis = tl.where(
                    channels[None, None, :] == 4,
                    (v * v - 1.0)[:, :, None], basis,
                )
            values = tl.load(
                coefficients_ptr + gid * stride_n_coefficients + channels,
                mask=valid_g & active_mask & channel_mask,
                other=0.0,
            )
            acc += gaussian[:, :, None] * basis * values[None, None, :]

    output_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    mask = (
        (py[:, None, None] < H)
        & (px[None, :, None] < W)
        & channel_mask[None, None, :]
    )
    tl.store(output_ptrs, acc, mask=mask)


@triton.jit
def _basis_bank_adjoint_kernel(
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
    stride_c_image,
    stride_h_image,
    stride_w_image,
    stride_n_coefficient_grad,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    ORDER: tl.constexpr,
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
    channel_mask = channels < 5
    active_mask = channels < (2 + 3 * (ORDER >= 2))
    pixel_mask = (py[:, None, None] < H) & (px[None, :, None] < W)
    image_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    image_values = tl.load(
        image_ptrs,
        mask=pixel_mask & channel_mask[None, None, :],
        other=0.0,
    )

    for local_g in tl.range(0, GAUSS_CHUNK):
        interaction = chunk_start + local_g
        valid_g = interaction < range_end
        gid = tl.load(
            sorted_gauss_ids_ptr + interaction,
            mask=valid_g,
            other=0,
        )
        mean_ptr = means_ptr + gid * stride_n_means
        mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
        mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
        frame_ptr = frames_ptr + gid * stride_n_frames
        inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
        inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
        cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
        sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
        dx = (px_mesh + 0.5) - mu_x
        dy = (py_mesh + 0.5) - mu_y
        u = (cosine * dx + sine * dy) * inv_ax
        v = (-sine * dx + cosine * dy) * inv_ay
        gaussian = tl.where(
            valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0
        )
        basis = tl.where(channels[None, None, :] == 0, -u[:, :, None], 0.0)
        basis = tl.where(
            channels[None, None, :] == 1, -v[:, :, None], basis
        )
        if ORDER >= 2:
            basis = tl.where(
                channels[None, None, :] == 2,
                (u * u - 1.0)[:, :, None], basis,
            )
            basis = tl.where(
                channels[None, None, :] == 3,
                (u * v)[:, :, None], basis,
            )
            basis = tl.where(
                channels[None, None, :] == 4,
                (v * v - 1.0)[:, :, None], basis,
            )
        gradients = tl.sum(
            tl.sum(image_values * gaussian[:, :, None] * basis, axis=0),
            axis=0,
        )
        tl.atomic_add(
            coefficient_grad_ptr + gid * stride_n_coefficient_grad + channels,
            gradients,
            mask=valid_g & active_mask & channel_mask,
        )


@triton.jit
def _direct_forward_kernel(
    tile_starts_ptr,
    sorted_gauss_ids_ptr,
    means_ptr,
    frames_ptr,
    zero_ptr,
    higher_ptr,
    scalar_ptr,
    image_ptr,
    H,
    W,
    ZERO_CHANNELS,
    OUTPUT_CHANNELS,
    tiles_x,
    tiles_y,
    stride_n_means,
    stride_n_frames,
    stride_n_zero,
    stride_n_higher,
    stride_n_scalar,
    stride_b_image,
    stride_c_image,
    stride_h_image,
    stride_w_image,
    TILE_SIZE: tl.constexpr,
    GAUSS_CHUNK: tl.constexpr,
    CHUNKS_NEEDED: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INDEPENDENT_BASIS: tl.constexpr,
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
    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < OUTPUT_CHANNELS
    acc = tl.zeros((TILE_SIZE, TILE_SIZE, BLOCK_C), tl.float32)

    for chunk_idx in tl.range(0, CHUNKS_NEEDED):
        chunk_start = range_start + chunk_idx * GAUSS_CHUNK
        for local_g in tl.range(0, GAUSS_CHUNK):
            interaction = chunk_start + local_g
            valid_g = interaction < range_end
            gid = tl.load(
                sorted_gauss_ids_ptr + interaction,
                mask=valid_g,
                other=0,
            )
            mean_ptr = means_ptr + gid * stride_n_means
            mu_x = tl.load(mean_ptr, mask=valid_g, other=0.0)
            mu_y = tl.load(mean_ptr + 1, mask=valid_g, other=0.0)
            frame_ptr = frames_ptr + gid * stride_n_frames
            inv_ax = tl.load(frame_ptr, mask=valid_g, other=1.0)
            inv_ay = tl.load(frame_ptr + 1, mask=valid_g, other=1.0)
            cosine = tl.load(frame_ptr + 2, mask=valid_g, other=1.0)
            sine = tl.load(frame_ptr + 3, mask=valid_g, other=0.0)
            dx = (px_mesh + 0.5) - mu_x
            dy = (py_mesh + 0.5) - mu_y
            u = (cosine * dx + sine * dy) * inv_ax
            v = (-sine * dx + cosine * dy) * inv_ay
            gaussian = tl.where(
                valid_g, tl.exp(-0.5 * (u * u + v * v)), 0.0
            )

            zero_values = tl.load(
                zero_ptr + gid * stride_n_zero + channels,
                mask=valid_g & (channels < ZERO_CHANNELS),
                other=0.0,
            )
            higher_base = higher_ptr + gid * stride_n_higher
            gx = tl.load(higher_base, mask=valid_g, other=0.0)
            gy = tl.load(higher_base + 1, mask=valid_g, other=0.0)
            response_gx = -gx * u
            response_gy = -gy * v
            first = response_gx + response_gy
            second = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
            response_gxx = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
            response_gxy = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
            response_gyy = tl.zeros((TILE_SIZE, TILE_SIZE), tl.float32)
            if ORDER >= 2:
                gxx = tl.load(higher_base + 2, mask=valid_g, other=0.0)
                gxy = tl.load(higher_base + 3, mask=valid_g, other=0.0)
                gyy = tl.load(higher_base + 4, mask=valid_g, other=0.0)
                response_gxx = gxx * (u * u - 1.0)
                response_gxy = gxy * u * v
                response_gyy = gyy * (v * v - 1.0)
                second = response_gxx + response_gxy + response_gyy
            scalar = tl.load(
                scalar_ptr + gid * stride_n_scalar,
                mask=valid_g,
                other=0.0,
            )
            values = tl.where(
                channels[None, None, :] < ZERO_CHANNELS,
                zero_values[None, None, :],
                0.0,
            )
            if INDEPENDENT_BASIS:
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS,
                    response_gx[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 1,
                    response_gy[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 2,
                    response_gxx[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 3,
                    response_gxy[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 4,
                    response_gyy[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 5,
                    scalar, values,
                )
            else:
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS,
                    first[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 1,
                    second[:, :, None], values,
                )
                values = tl.where(
                    channels[None, None, :] == ZERO_CHANNELS + 2,
                    scalar, values,
                )
            acc += gaussian[:, :, None] * values

    output_ptrs = (
        image_ptr
        + pid_batch * stride_b_image
        + channels[None, None, :] * stride_c_image
        + py_mesh[:, :, None] * stride_h_image
        + px_mesh[:, :, None] * stride_w_image
    )
    mask = (
        (py[:, None, None] < H)
        & (px[None, :, None] < W)
        & channel_mask[None, None, :]
    )
    tl.store(output_ptrs, acc, mask=mask)


def augment_derivative_cache(cache, geometry):
    """Add inverse local axes and rotation to a standard renderer cache."""

    if "derivative_frames" in cache:
        return cache
    height = cache["H"]
    width = cache["W"]
    minimum = float(cache["config"].min_sigma_px)
    detached = geometry.detach()
    axis_x = (detached[..., 2] * float(width)).clamp_min(minimum)
    axis_y = (detached[..., 3] * float(height)).clamp_min(minimum)
    theta = detached[..., 4]
    cache["derivative_frames"] = torch.stack(
        [axis_x.reciprocal(), axis_y.reciprocal(), theta.cos(), theta.sin()],
        dim=-1,
    ).reshape(-1, 4).contiguous()
    return cache


def _common_cache_tensors(cache):
    means = cache["means"].reshape(-1, 2).contiguous()
    return (
        cache["tile_starts"],
        cache["sorted_gauss_ids"],
        means,
        cache["derivative_frames"],
    )


def fused_derivative_render_cached(cache, coefficients, order):
    order = int(order)
    expected = 1 + 2 * (order >= 1) + 3 * (order >= 2)
    if order not in (1, 2) or coefficients.shape[-1] != expected:
        raise ValueError(
            f"Order {order} fused render expects {expected} coefficients"
        )
    coefficients = coefficients.contiguous()
    batch = coefficients.shape[0]
    image = torch.empty(
        (batch, 1, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    image.zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    tile_starts, sorted_ids, means, frames = _common_cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, expected)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch)
    _derivative_forward_kernel[grid](
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
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        ORDER=order,
    )
    return image


def fused_derivative_adjoint_cached(cache, image, order):
    order = int(order)
    modes = 1 + 2 * (order >= 1) + 3 * (order >= 2)
    if order not in (1, 2) or image.shape[1] != 1:
        raise ValueError("Fused derivative adjoint expects one image channel")
    image = image.to(torch.float32).contiguous(memory_format=torch.channels_last)
    gradients = torch.zeros(
        (cache["B"], cache["N"], modes),
        device=image.device,
        dtype=torch.float32,
    )
    if cache["sorted_gauss_ids"] is None:
        return gradients
    tile_starts, sorted_ids, means, frames = _common_cache_tensors(cache)
    gradients_flat = gradients.reshape(-1, modes)
    grid = (
        cache["tiles_x"] * cache["tiles_y"],
        cache["B"],
        cache["chunks_needed"],
    )
    _derivative_adjoint_kernel[grid](
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
        image.stride(1),
        image.stride(2),
        image.stride(3),
        gradients_flat.stride(0),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        ORDER=order,
    )
    return gradients


def fused_basis_bank_render_cached(cache, coefficients, order):
    """Render five independent Gx/Gy/Gxx/Gxy/Gyy response maps."""

    order = int(order)
    if order not in (1, 2) or coefficients.shape[-1] != 5:
        raise ValueError("Basis-bank render expects five local coefficients")
    coefficients = coefficients.contiguous()
    batch = coefficients.shape[0]
    image = torch.empty(
        (batch, 5, cache["H"], cache["W"]),
        device=coefficients.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    image.zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    tile_starts, sorted_ids, means, frames = _common_cache_tensors(cache)
    coefficients_flat = coefficients.reshape(-1, 5)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch)
    _basis_bank_forward_kernel[grid](
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
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        ORDER=order,
        BLOCK_C=8,
    )
    return image


def fused_basis_bank_adjoint_cached(cache, images, order):
    """Transpose the independent five-channel basis-bank render."""

    order = int(order)
    if order not in (1, 2) or images.shape[1] != 5:
        raise ValueError("Basis-bank adjoint expects five image channels")
    images = images.to(torch.float32).contiguous(memory_format=torch.channels_last)
    gradients = torch.zeros(
        (cache["B"], cache["N"], 5),
        device=images.device,
        dtype=torch.float32,
    )
    if cache["sorted_gauss_ids"] is None:
        return gradients
    tile_starts, sorted_ids, means, frames = _common_cache_tensors(cache)
    gradients_flat = gradients.reshape(-1, 5)
    grid = (
        cache["tiles_x"] * cache["tiles_y"],
        cache["B"],
        cache["chunks_needed"],
    )
    _basis_bank_adjoint_kernel[grid](
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
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        ORDER=order,
        BLOCK_C=8,
    )
    return gradients


class _FusedBasisBankRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, cache, order):
        ctx.cache = cache
        ctx.order = int(order)
        return fused_basis_bank_render_cached(cache, coefficients, order)

    @staticmethod
    def backward(ctx, grad_images):
        return (
            fused_basis_bank_adjoint_cached(
                ctx.cache, grad_images, ctx.order
            ),
            None,
            None,
        )


def differentiable_fused_basis_bank_render(cache, coefficients, order):
    return _FusedBasisBankRender.apply(coefficients, cache, int(order))


class _FusedDerivativeRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, cache, order):
        ctx.cache = cache
        ctx.order = int(order)
        return fused_derivative_render_cached(cache, coefficients, order)

    @staticmethod
    def backward(ctx, grad_image):
        return (
            fused_derivative_adjoint_cached(ctx.cache, grad_image, ctx.order),
            None,
            None,
        )


class _FusedDerivativeAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, cache, order):
        ctx.cache = cache
        ctx.order = int(order)
        return fused_derivative_adjoint_cached(cache, image, order)

    @staticmethod
    def backward(ctx, grad_coefficients):
        return (
            fused_derivative_render_cached(
                ctx.cache, grad_coefficients, ctx.order
            ),
            None,
            None,
        )


def differentiable_fused_derivative_render(cache, coefficients, order):
    return _FusedDerivativeRender.apply(coefficients, cache, int(order))


def differentiable_fused_derivative_adjoint(cache, image, order):
    return _FusedDerivativeAdjoint.apply(image, cache, int(order))


def fused_direct_render_cached(
    cache, zero_order_coefficients, higher_order_coefficients,
    scalar_coefficients, order, independent_basis=False
):
    """Render zero features, first/second responses and scalar in one pass."""

    order = int(order)
    if order not in (1, 2):
        raise ValueError("Fused direct render is only used for order 1 or 2")
    zero = zero_order_coefficients.contiguous()
    higher = higher_order_coefficients[..., : (2 if order == 1 else 5)].contiguous()
    scalar = scalar_coefficients.contiguous()
    batch, _, zero_channels = zero.shape
    independent_basis = bool(independent_basis)
    output_channels = zero_channels + (6 if independent_basis else 3)
    image = torch.empty(
        (batch, output_channels, cache["H"], cache["W"]),
        device=zero.device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    image.zero_()
    if cache["sorted_gauss_ids"] is None:
        return image
    tile_starts, sorted_ids, means, frames = _common_cache_tensors(cache)
    zero_flat = zero.reshape(-1, zero_channels)
    higher_flat = higher.reshape(-1, higher.shape[-1])
    scalar_flat = scalar.reshape(-1, 1)
    block_c = triton.next_power_of_2(output_channels)
    grid = (cache["tiles_x"] * cache["tiles_y"], batch)
    _direct_forward_kernel[grid](
        tile_starts,
        sorted_ids,
        means,
        frames,
        zero_flat,
        higher_flat,
        scalar_flat,
        image,
        cache["H"],
        cache["W"],
        zero_channels,
        output_channels,
        cache["tiles_x"],
        cache["tiles_y"],
        means.stride(0),
        frames.stride(0),
        zero_flat.stride(0),
        higher_flat.stride(0),
        scalar_flat.stride(0),
        image.stride(0),
        image.stride(1),
        image.stride(2),
        image.stride(3),
        TILE_SIZE=cache["config"].tile_size,
        GAUSS_CHUNK=cache["config"].max_gauss_chunk,
        CHUNKS_NEEDED=cache["chunks_needed"],
        ORDER=order,
        BLOCK_C=block_c,
        INDEPENDENT_BASIS=independent_basis,
    )
    return image
