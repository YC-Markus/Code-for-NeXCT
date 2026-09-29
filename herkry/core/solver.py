"""Cached zeroth-order CGLS support for the shared block base class."""
import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from .utils import recon

def batch_dot(left, right):
    return (left * right).flatten(1).sum(dim=1, keepdim=True)


@torch.no_grad()
def cached_cgls_amplitudes(
    geometry,
    initial_amplitudes,
    base_image,
    sparse_sino,
    radon,
    scalar_renderer,
    iterations,
    render_size,
    cache_kwargs,
    collect_history=False,
    collect_images=False,
):
    """Solve min_delta ||A(base + Phi(a0 + delta)) - y||_2^2."""

    with record_function("cgls/cache_build"):
        cache = scalar_renderer.build_cache(
            geometry,
            H=render_size,
            W=render_size,
            **cache_kwargs,
        )

    def forward_operator(amplitudes, return_render=False):
        with record_function("cgls/forward_render_project"):
            rendered = scalar_renderer.render_cached(cache, amplitudes)
            projected = radon.forward(rendered.contiguous())
            if return_render:
                return projected, rendered
            return projected

    def adjoint_operator(sinogram):
        with record_function("cgls/backproject_adjoint"):
            backprojected = radon.backprojection(sinogram.contiguous())
            return scalar_renderer.adjoint_cached(
                cache, backprojected.contiguous()
            )

    with record_function("cgls/initial_render_project"):
        initial_render = scalar_renderer.render_cached(
            cache, initial_amplitudes
        )
        base_sino = radon.forward(base_image.contiguous())
        initial_sino = base_sino + radon.forward(
            initial_render.contiguous()
        )
    base_l1 = (
        F.l1_loss(base_sino, sparse_sino, reduction="none")
        .flatten(1)
        .mean(1)
    )
    base_mse = (
        F.mse_loss(base_sino, sparse_sino, reduction="none")
        .flatten(1)
        .mean(1)
    )
    residual = sparse_sino - initial_sino
    pre_l1 = F.l1_loss(initial_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    pre_mse = F.mse_loss(initial_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    history = [pre_mse] if collect_history else None
    image_history = (
        [base_image + initial_render] if collect_images else None
    )
    current_image = image_history[0] if collect_images else None
    sino_history = [initial_sino] if collect_images else None

    delta = torch.zeros_like(initial_amplitudes)
    if int(iterations) == 0:
        return {
            "amplitudes": initial_amplitudes,
            "pre_sino": initial_sino,
            "post_sino": initial_sino,
            "pre_l1": pre_l1,
            "post_l1": pre_l1,
            "base_l1": base_l1,
            "pre_mse": pre_mse,
            "post_mse": pre_mse,
            "base_mse": base_mse,
            "mse_history": (
                torch.stack(history, dim=1) if collect_history else None
            ),
            "image_history": (
                torch.stack(image_history, dim=1)
                if collect_images
                else None
            ),
            "sino_history": (
                torch.stack(sino_history, dim=1)
                if collect_images
                else None
            ),
            "cache": cache,
        }

    gradient = adjoint_operator(residual)
    direction = gradient.clone()
    gradient_norm_sq = batch_dot(gradient, gradient)

    for iteration_idx in range(int(iterations)):
        with record_function(f"cgls/iteration_{iteration_idx + 1}"):
            forward_result = forward_operator(
                direction, return_render=collect_images
            )
            if collect_images:
                projected_direction, rendered_direction = forward_result
            else:
                projected_direction = forward_result
        denominator = batch_dot(
            projected_direction, projected_direction
        ).clamp_min(1e-20)
        alpha = torch.nan_to_num(
            gradient_norm_sq / denominator,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        delta = delta + alpha.view(-1, 1, 1) * direction
        residual = residual - alpha.view(-1, 1, 1, 1) * projected_direction
        if collect_history:
            history.append(residual.square().flatten(1).mean(1))
        if collect_images:
            current_image = (
                current_image
                + alpha.view(-1, 1, 1, 1) * rendered_direction
            )
            image_history.append(current_image)
            sino_history.append(sparse_sino - residual)
        if iteration_idx + 1 == int(iterations):
            break
        next_gradient = adjoint_operator(residual)
        next_norm_sq = batch_dot(next_gradient, next_gradient)
        beta = torch.nan_to_num(
            next_norm_sq / gradient_norm_sq.clamp_min(1e-20),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        direction = next_gradient + beta.view(-1, 1, 1) * direction
        gradient = next_gradient
        gradient_norm_sq = next_norm_sq

    optimized_amplitudes = initial_amplitudes + delta
    final_sino = sparse_sino - residual
    post_l1 = F.l1_loss(final_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    post_mse = F.mse_loss(final_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    return {
        "amplitudes": optimized_amplitudes,
        "pre_sino": initial_sino,
        "post_sino": final_sino,
        "pre_l1": pre_l1,
        "post_l1": post_l1,
        "base_l1": base_l1,
        "pre_mse": pre_mse,
        "post_mse": post_mse,
        "base_mse": base_mse,
        "mse_history": (
            torch.stack(history, dim=1) if collect_history else None
        ),
        "image_history": (
            torch.stack(image_history, dim=1)
            if collect_images
            else None
        ),
        "sino_history": (
            torch.stack(sino_history, dim=1)
            if collect_images
            else None
        ),
        "cache": cache,
    }
