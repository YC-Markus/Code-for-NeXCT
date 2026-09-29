"""Direct Triton Gaussian transpose gather.

The operator can use either an exact Mahalanobis 3-sigma ellipse or the
renderer-compatible 3-sigma tile envelope.  The latter reproduces the legacy
renderer without rebuilding its bin/sort metadata.  Backward implements
gradients for geometry and the gathered image.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_3sigma_forward_kernel(
    geometry_ptr,
    image_ptr,
    output_ptr,
    num_gaussians: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    support_side: tl.constexpr,
    block_size: tl.constexpr,
    tile_size: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // num_gaussians
    geometry_offset = pid * 5
    center_x = tl.load(geometry_ptr + geometry_offset + 0) * width
    center_y = tl.load(geometry_ptr + geometry_offset + 1) * height
    raw_axis_x = tl.load(geometry_ptr + geometry_offset + 2) * width
    raw_axis_y = tl.load(geometry_ptr + geometry_offset + 3) * height
    theta = tl.load(geometry_ptr + geometry_offset + 4)
    axis_x = tl.maximum(raw_axis_x, 0.1)
    axis_y = tl.maximum(raw_axis_y, 0.1)
    inv_axis_x2 = 1.0 / (axis_x * axis_x)
    inv_axis_y2 = 1.0 / (axis_y * axis_y)
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)
    center_ix = tl.floor(center_x).to(tl.int32)
    center_iy = tl.floor(center_y).to(tl.int32)
    if tile_size > 0:
        bbox_radius = 3.0 * tl.maximum(axis_x, axis_y)
        min_x = tl.maximum(0.0, tl.minimum(center_x - bbox_radius, width - 1.0))
        max_x = tl.maximum(0.0, tl.minimum(center_x + bbox_radius, width - 1.0))
        min_y = tl.maximum(0.0, tl.minimum(center_y - bbox_radius, height - 1.0))
        max_y = tl.maximum(0.0, tl.minimum(center_y + bbox_radius, height - 1.0))
        pixel_min_x = tl.floor(min_x / tile_size).to(tl.int32) * tile_size
        pixel_max_x = (tl.floor(max_x / tile_size).to(tl.int32) + 1) * tile_size
        pixel_min_y = tl.floor(min_y / tile_size).to(tl.int32) * tile_size
        pixel_max_y = (tl.floor(max_y / tile_size).to(tl.int32) + 1) * tile_size
    radius = support_side // 2
    support_pixels: tl.constexpr = support_side * support_side
    offsets = tl.arange(0, block_size)
    accumulator = 0.0

    for start in range(0, support_pixels, block_size):
        linear = start + offsets
        offset_x = linear % support_side - radius
        offset_y = linear // support_side - radius
        pixel_x = center_ix + offset_x
        pixel_y = center_iy + offset_y
        safe_x = tl.maximum(0, tl.minimum(pixel_x, width - 1))
        safe_y = tl.maximum(0, tl.minimum(pixel_y, height - 1))
        dx = pixel_x.to(tl.float32) + 0.5 - center_x
        dy = pixel_y.to(tl.float32) + 0.5 - center_y
        rotated_x = cos_theta * dx + sin_theta * dy
        rotated_y = -sin_theta * dx + cos_theta * dy
        mahalanobis = (
            rotated_x * rotated_x * inv_axis_x2
            + rotated_y * rotated_y * inv_axis_y2
        )
        in_support = mahalanobis <= 9.0
        if tile_size > 0:
            in_support = (
                (pixel_x >= pixel_min_x)
                & (pixel_x < pixel_max_x)
                & (pixel_y >= pixel_min_y)
                & (pixel_y < pixel_max_y)
            )
        valid = (
            (linear < support_pixels)
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
            & in_support
        )
        image_offset = batch_id * height * width + safe_y * width + safe_x
        image_value = tl.load(image_ptr + image_offset, mask=valid, other=0.0)
        weight = tl.exp(-0.5 * mahalanobis)
        accumulator += tl.sum(tl.where(valid, image_value * weight, 0.0), axis=0)

    tl.store(output_ptr + pid, accumulator)


@triton.jit
def _gather_3sigma_backward_kernel(
    geometry_ptr,
    image_ptr,
    grad_output_ptr,
    grad_geometry_ptr,
    grad_image_ptr,
    num_gaussians: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    support_side: tl.constexpr,
    block_size: tl.constexpr,
    tile_size: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // num_gaussians
    geometry_offset = pid * 5
    center_x = tl.load(geometry_ptr + geometry_offset + 0) * width
    center_y = tl.load(geometry_ptr + geometry_offset + 1) * height
    raw_axis_x = tl.load(geometry_ptr + geometry_offset + 2) * width
    raw_axis_y = tl.load(geometry_ptr + geometry_offset + 3) * height
    theta = tl.load(geometry_ptr + geometry_offset + 4)
    axis_x = tl.maximum(raw_axis_x, 0.1)
    axis_y = tl.maximum(raw_axis_y, 0.1)
    inv_axis_x2 = 1.0 / (axis_x * axis_x)
    inv_axis_y2 = 1.0 / (axis_y * axis_y)
    inv_axis_x3 = inv_axis_x2 / axis_x
    inv_axis_y3 = inv_axis_y2 / axis_y
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)
    center_ix = tl.floor(center_x).to(tl.int32)
    center_iy = tl.floor(center_y).to(tl.int32)
    if tile_size > 0:
        bbox_radius = 3.0 * tl.maximum(axis_x, axis_y)
        min_x = tl.maximum(0.0, tl.minimum(center_x - bbox_radius, width - 1.0))
        max_x = tl.maximum(0.0, tl.minimum(center_x + bbox_radius, width - 1.0))
        min_y = tl.maximum(0.0, tl.minimum(center_y - bbox_radius, height - 1.0))
        max_y = tl.maximum(0.0, tl.minimum(center_y + bbox_radius, height - 1.0))
        pixel_min_x = tl.floor(min_x / tile_size).to(tl.int32) * tile_size
        pixel_max_x = (tl.floor(max_x / tile_size).to(tl.int32) + 1) * tile_size
        pixel_min_y = tl.floor(min_y / tile_size).to(tl.int32) * tile_size
        pixel_max_y = (tl.floor(max_y / tile_size).to(tl.int32) + 1) * tile_size
    grad_output = tl.load(grad_output_ptr + pid)
    radius = support_side // 2
    support_pixels: tl.constexpr = support_side * support_side
    offsets = tl.arange(0, block_size)
    grad_center_x = 0.0
    grad_center_y = 0.0
    grad_axis_x = 0.0
    grad_axis_y = 0.0
    grad_theta = 0.0

    for start in range(0, support_pixels, block_size):
        linear = start + offsets
        offset_x = linear % support_side - radius
        offset_y = linear // support_side - radius
        pixel_x = center_ix + offset_x
        pixel_y = center_iy + offset_y
        safe_x = tl.maximum(0, tl.minimum(pixel_x, width - 1))
        safe_y = tl.maximum(0, tl.minimum(pixel_y, height - 1))
        dx = pixel_x.to(tl.float32) + 0.5 - center_x
        dy = pixel_y.to(tl.float32) + 0.5 - center_y
        rotated_x = cos_theta * dx + sin_theta * dy
        rotated_y = -sin_theta * dx + cos_theta * dy
        mahalanobis = (
            rotated_x * rotated_x * inv_axis_x2
            + rotated_y * rotated_y * inv_axis_y2
        )
        in_support = mahalanobis <= 9.0
        if tile_size > 0:
            in_support = (
                (pixel_x >= pixel_min_x)
                & (pixel_x < pixel_max_x)
                & (pixel_y >= pixel_min_y)
                & (pixel_y < pixel_max_y)
            )
        valid = (
            (linear < support_pixels)
            & (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
            & in_support
        )
        image_offset = batch_id * height * width + safe_y * width + safe_x
        image_value = tl.load(image_ptr + image_offset, mask=valid, other=0.0)
        weight = tl.exp(-0.5 * mahalanobis)
        common = tl.where(valid, grad_output * image_value * weight, 0.0)

        dlogw_dcenter_x = (
            rotated_x * cos_theta * inv_axis_x2
            - rotated_y * sin_theta * inv_axis_y2
        )
        dlogw_dcenter_y = (
            rotated_x * sin_theta * inv_axis_x2
            + rotated_y * cos_theta * inv_axis_y2
        )
        grad_center_x += tl.sum(common * dlogw_dcenter_x * width, axis=0)
        grad_center_y += tl.sum(common * dlogw_dcenter_y * height, axis=0)
        grad_axis_x += tl.sum(
            common * rotated_x * rotated_x * inv_axis_x3 * width,
            axis=0,
        )
        grad_axis_y += tl.sum(
            common * rotated_y * rotated_y * inv_axis_y3 * height,
            axis=0,
        )
        grad_theta += tl.sum(
            common
            * rotated_x
            * rotated_y
            * (inv_axis_y2 - inv_axis_x2),
            axis=0,
        )

        grad_image_value = tl.where(valid, grad_output * weight, 0.0)
        tl.atomic_add(
            grad_image_ptr + image_offset,
            grad_image_value,
            mask=valid,
        )

    axis_x_active = raw_axis_x > 0.1
    axis_y_active = raw_axis_y > 0.1
    tl.store(grad_geometry_ptr + geometry_offset + 0, grad_center_x)
    tl.store(grad_geometry_ptr + geometry_offset + 1, grad_center_y)
    tl.store(
        grad_geometry_ptr + geometry_offset + 2,
        tl.where(axis_x_active, grad_axis_x, 0.0),
    )
    tl.store(
        grad_geometry_ptr + geometry_offset + 3,
        tl.where(axis_y_active, grad_axis_y, 0.0),
    )
    tl.store(grad_geometry_ptr + geometry_offset + 4, grad_theta)


def _support_side(num_gaussians, height, width, tile_size):
    resolution = math.isqrt(num_gaussians)
    if resolution * resolution != num_gaussians:
        raise ValueError(f"Expected square Gaussian lattice, got N={num_gaussians}")
    base_sigma_x = max(0.75, 0.75 * width / resolution)
    base_sigma_y = max(0.75, 0.75 * height / resolution)
    max_radius = math.ceil(3.0 * 2.0 * max(base_sigma_x, base_sigma_y))
    if tile_size:
        max_radius += int(tile_size)
    return 2 * max_radius + 1


class GaussianTransposeGather3SigmaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, geometry, image, tile_size):
        if geometry.dtype != torch.float32 or image.dtype != torch.float32:
            raise TypeError("3-sigma gather currently requires float32 tensors")
        if geometry.ndim != 3 or geometry.shape[-1] != 5:
            raise ValueError(f"Expected geometry [B,N,5], got {tuple(geometry.shape)}")
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected image [B,1,H,W], got {tuple(image.shape)}")
        if geometry.shape[0] != image.shape[0]:
            raise ValueError("Geometry and image batch sizes must match")
        geometry = geometry.contiguous()
        image = image.contiguous()
        batch, num_gaussians, _ = geometry.shape
        height, width = image.shape[-2:]
        tile_size = int(tile_size)
        support_side = _support_side(
            num_gaussians, height, width, tile_size
        )
        output = torch.empty(
            (batch, num_gaussians, 1), device=geometry.device, dtype=torch.float32
        )
        grid = (batch * num_gaussians,)
        block_size = 256
        _gather_3sigma_forward_kernel[grid](
            geometry,
            image,
            output,
            num_gaussians=num_gaussians,
            height=height,
            width=width,
            support_side=support_side,
            block_size=block_size,
            tile_size=tile_size,
            num_warps=4,
        )
        ctx.has_backward = geometry.requires_grad or image.requires_grad
        if ctx.has_backward:
            ctx.save_for_backward(geometry, image)
            ctx.support_side = support_side
            ctx.block_size = block_size
            ctx.tile_size = tile_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.has_backward:
            return None, None, None
        geometry, image = ctx.saved_tensors
        batch, num_gaussians, _ = geometry.shape
        height, width = image.shape[-2:]
        grad_geometry = torch.empty_like(geometry)
        grad_image = torch.zeros_like(image)
        grid = (batch * num_gaussians,)
        _gather_3sigma_backward_kernel[grid](
            geometry,
            image,
            grad_output.contiguous(),
            grad_geometry,
            grad_image,
            num_gaussians=num_gaussians,
            height=height,
            width=width,
            support_side=ctx.support_side,
            block_size=ctx.block_size,
            tile_size=ctx.tile_size,
            num_warps=4,
        )
        return grad_geometry, grad_image, None


def gaussian_transpose_gather_3sigma(
    geometry, image, tile_size=0
):
    return GaussianTransposeGather3SigmaFunction.apply(
        geometry, image, int(tile_size)
    )
