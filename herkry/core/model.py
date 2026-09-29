"""Fully unrolled differentiable CGLS trajectory-context model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from herkry.core.blocks import (
    GCTCGLSTrajectoryContext,
    SVCTFactorizedRouter,
    TrajectoryContextGaussianBlock,
)
from herkry.core.trajectory import (
    KrylovDeltaGRU,
    prune_inactive_hermite_heads as _prune_inactive_hermite_heads,
)
from herkry.core.render_gaussian_hermite_fused import (
    augment_hermite_cache,
    differentiable_hermite_high_render,
    differentiable_hermite_high_shell_cnn_render,
    differentiable_hermite_scalar_adjoint,
    differentiable_hermite_scalar_render,
    differentiable_hermite_scalar_shell_trajectory_render,
    pack_active_high_modes,
)
from herkry.core.utils import recon


def batch_dot(left, right):
    return (left * right).flatten(1).sum(dim=1, keepdim=True)


class CachedLinearRender(torch.autograd.Function):

    @staticmethod
    def forward(ctx, colors, scalar_renderer, cache):
        ctx.scalar_renderer = scalar_renderer
        ctx.cache = cache
        return scalar_renderer.render_cached(cache, colors)

    @staticmethod
    def backward(ctx, grad_image):
        grad_colors = ctx.scalar_renderer.adjoint_cached(ctx.cache, grad_image.contiguous())
        return (grad_colors, None, None)


class CachedLinearAdjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, image, scalar_renderer, cache):
        ctx.scalar_renderer = scalar_renderer
        ctx.cache = cache
        return scalar_renderer.adjoint_cached(cache, image.contiguous())

    @staticmethod
    def backward(ctx, grad_colors):
        grad_image = ctx.scalar_renderer.render_cached(ctx.cache, grad_colors.contiguous())
        return (grad_image, None, None)


def differentiable_cached_render(scalar_renderer, cache, colors):
    return CachedLinearRender.apply(colors, scalar_renderer, cache)


def differentiable_cached_adjoint(scalar_renderer, cache, image):
    return CachedLinearAdjoint.apply(image, scalar_renderer, cache)


def unrolled_cached_cgls(
    geometry,
    initial_amplitudes,
    base_image,
    sparse_sino,
    radon,
    scalar_renderer,
    iterations,
    render_size,
    cache_kwargs,
    prebuilt_cache=None,
    precomputed_initial_render=None,
    precomputed_initial_sino=None,
    precomputed_base_sino=None,
    gaussian_order=0,
):
    """Unroll CGLS with gradients through every amplitude-space operation."""
    if prebuilt_cache is None:
        with torch.no_grad():
            cache = scalar_renderer.build_cache(
                geometry.detach(), H=render_size, W=render_size, **cache_kwargs
            )
    else:
        cache = prebuilt_cache
    gaussian_order = int(gaussian_order)
    operator_geometry = geometry.detach()
    if gaussian_order > 0:
        augment_hermite_cache(cache, operator_geometry)

    def render(amplitudes):
        if gaussian_order > 0:
            return differentiable_hermite_scalar_render(cache, amplitudes, gaussian_order)
        return differentiable_cached_render(scalar_renderer, cache, amplitudes.contiguous())

    def forward_operator(amplitudes):
        rendered = render(amplitudes)
        projected = radon.forward(rendered.contiguous())
        return (projected, rendered)

    def adjoint_operator(sinogram):
        backprojected = radon.backprojection(sinogram.contiguous())
        if gaussian_order == 0:
            return differentiable_cached_adjoint(scalar_renderer, cache, backprojected)
        return differentiable_hermite_scalar_adjoint(cache, backprojected, gaussian_order)

    if precomputed_initial_render is None:
        initial_render = render(initial_amplitudes)
    else:
        initial_render = precomputed_initial_render
    if precomputed_base_sino is None:
        base_sino = radon.forward(base_image.contiguous())
    else:
        base_sino = precomputed_base_sino
    if precomputed_initial_sino is None:
        initial_render_sino = radon.forward(initial_render.contiguous())
        initial_sino = base_sino + initial_render_sino
    else:
        initial_sino = precomputed_initial_sino
    residual = sparse_sino - initial_sino
    base_l1 = F.l1_loss(base_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    base_mse = F.mse_loss(base_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    pre_l1 = F.l1_loss(initial_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    pre_mse = F.mse_loss(initial_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    delta = torch.zeros_like(initial_amplitudes)
    gradient = adjoint_operator(residual)
    direction = gradient
    gradient_norm_sq = batch_dot(gradient, gradient)
    current_image = base_image + initial_render
    image_history = [current_image]
    sino_history = [initial_sino]
    mse_history = [pre_mse]
    amplitude_history = [initial_amplitudes]
    for iteration_idx in range(int(iterations)):
        (projected_direction, rendered_direction) = forward_operator(direction)
        denominator = batch_dot(projected_direction, projected_direction).clamp_min(1e-20)
        alpha = torch.nan_to_num(gradient_norm_sq / denominator, nan=0.0, posinf=0.0, neginf=0.0)
        delta = delta + alpha.view(-1, 1, 1) * direction
        residual = residual - alpha.view(-1, 1, 1, 1) * projected_direction
        current_image = current_image + alpha.view(-1, 1, 1, 1) * rendered_direction
        image_history.append(current_image)
        sino_history.append(sparse_sino - residual)
        amplitude_history.append(initial_amplitudes + delta)
        mse_history.append(residual.square().flatten(1).mean(1))
        if iteration_idx + 1 < int(iterations):
            next_gradient = adjoint_operator(residual)
            next_norm_sq = batch_dot(next_gradient, next_gradient)
            beta = torch.nan_to_num(
                next_norm_sq / gradient_norm_sq.clamp_min(1e-20), nan=0.0, posinf=0.0, neginf=0.0
            )
            direction = next_gradient + beta.view(-1, 1, 1) * direction
            gradient = next_gradient
            gradient_norm_sq = next_norm_sq
    final_sino = sparse_sino - residual
    post_l1 = F.l1_loss(final_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    post_mse = F.mse_loss(final_sino, sparse_sino, reduction="none").flatten(1).mean(1)
    return {
        "amplitudes": initial_amplitudes + delta,
        "base_l1": base_l1,
        "base_mse": base_mse,
        "pre_l1": pre_l1,
        "pre_mse": pre_mse,
        "post_l1": post_l1,
        "post_mse": post_mse,
        "mse_history": torch.stack(mse_history, dim=1),
        "image_history": torch.stack(image_history, dim=1),
        "sino_history": torch.stack(sino_history, dim=1),
        "amplitude_history": torch.stack(amplitude_history, dim=1),
        "render_cache": cache,
    }


class UnrolledTrajectoryContextGaussianBlock(TrajectoryContextGaussianBlock):
    reuse_inference_cache = True
    vectorized_residual_fbp = True

    def forward(
        self,
        context_feature,
        stage_code,
        direct_base_image,
        sparse_sino,
        radon,
        renderer,
        decoder,
        cache_kwargs,
        routing=None,
        sigma_logit_bias=None,
        return_aux=False,
        cgls_base_image=None,
        stage_base_image=None,
        own_iterations=None,
        run_cgls=True,
        gaussian_order=0,
    ):
        if "direct" in cache_kwargs or "cgls" in cache_kwargs:
            direct_cache_kwargs = cache_kwargs["direct"]
            if not torch.is_grad_enabled():
                direct_cache_kwargs = cache_kwargs.get("direct_inference", direct_cache_kwargs)
            cgls_cache_kwargs = cache_kwargs.get("cgls", cache_kwargs["direct"])
            brane_cache_kwargs = cache_kwargs.get("brane", direct_cache_kwargs)
        else:
            direct_cache_kwargs = cache_kwargs
            cgls_cache_kwargs = cache_kwargs
            brane_cache_kwargs = cache_kwargs
        condition = (
            stage_code
            if self.condition is None
            else self.condition(torch.cat([stage_code, context_feature.mean(dim=(2, 3))], dim=1))
        )
        feature = F.gelu(self.context_fuse(context_feature, condition, routing=routing))
        feature = self.process_feature(feature, condition, routing=routing)
        spatial_hidden = F.gelu(self.to_raw_spatial_hidden(feature, condition, routing=routing))
        feature_hidden = (
            feature
            if self.to_raw_feature_hidden is None
            else F.gelu(self.to_raw_feature_hidden(feature, condition, routing=routing))
        )
        raw_spatial = self.to_raw_spatial_out(spatial_hidden, condition, routing=routing)
        if sigma_logit_bias is None:
            sigma_logit_bias = getattr(self, "sigma_logit_bias", 0.0)
        if sigma_logit_bias != 0.0:
            raw_spatial = raw_spatial.clone()
            raw_spatial[:, 2:4] = raw_spatial[:, 2:4] + sigma_logit_bias
        raw_feature = self.to_raw_feature_out(feature_hidden, condition, routing=routing)
        gaussian = decoder(raw_spatial, raw_feature)
        geometry = gaussian[..., :5]
        features = gaussian[..., 5:]
        if self.direct_amplitude_channels > 0:
            predicted_amplitudes = features[:, :, -1:].contiguous()
            image_amplitudes = features[:, :, : self.nonlinear_head_channels].contiguous()
        else:
            mixer = self.linear_feature_to_scalar.weight.reshape(1, self.token_channels, 1)
            predicted_amplitudes = torch.matmul(features, mixer).contiguous()
            image_amplitudes = None
        if self.image_feature_mixer is not None:
            image_mixer = self.image_feature_mixer.weight[:, :, 0, 0].transpose(0, 1).unsqueeze(0)
            image_amplitudes = torch.matmul(features, image_mixer).contiguous()
        brane_feature_coefficients = None
        brane_solver_coefficients = None
        if self.to_raw_higher_order_out is not None and int(gaussian_order) > 0:
            raw_higher_order = self.to_raw_higher_order_out(
                feature_hidden, condition, routing=routing
            )
            decoded_higher = decoder.decode_bounded_feature_map(raw_higher_order).contiguous()
            brane_vectors = self.brane_feature_channels + 1
            decoded_higher = decoded_higher.reshape(
                decoded_higher.shape[0],
                decoded_higher.shape[1],
                brane_vectors,
                self.brane_high_mode_count,
            )
            brane_feature_coefficients = decoded_higher[
                :, :, : self.brane_feature_channels
            ].contiguous()
            brane_solver_coefficients = decoded_higher[
                :, :, self.brane_feature_channels
            ].contiguous()
        render_amplitudes = (
            predicted_amplitudes
            if image_amplitudes is None
            else torch.cat([image_amplitudes, predicted_amplitudes], dim=-1)
        )
        inference_cache = None
        if not torch.is_grad_enabled() and self.reuse_inference_cache:
            inference_cache = renderer.renderer.build_cache(
                geometry, H=self.render_size, W=self.render_size, **direct_cache_kwargs
            )
            predicted_residual_image = renderer.renderer.render_cached(
                inference_cache, render_amplitudes
            )
        else:
            direct_gaussian = torch.cat([geometry, render_amplitudes], dim=-1)
            predicted_residual_image = renderer(
                direct_gaussian, H=self.render_size, W=self.render_size, **direct_cache_kwargs
            )
        brane_render_cache = None
        active_brane_features = None
        brane_feature_shell_images = None
        brane_cnn_features = None
        if brane_feature_coefficients is not None and int(gaussian_order) > 0:
            active_brane_features = pack_active_high_modes(
                brane_feature_coefficients, self.brane_max_degree, gaussian_order
            )
            with torch.no_grad():
                if inference_cache is not None and direct_cache_kwargs == brane_cache_kwargs:
                    brane_render_cache = inference_cache
                else:
                    brane_render_cache = renderer.renderer.build_cache(
                        geometry.detach(),
                        H=self.render_size,
                        W=self.render_size,
                        **brane_cache_kwargs,
                    )
                augment_hermite_cache(brane_render_cache, geometry.detach())
            brane_cnn_features = differentiable_hermite_high_shell_cnn_render(
                brane_render_cache,
                active_brane_features,
                gaussian_order,
                predicted_residual_image[:, : self.nonlinear_head_channels],
                self.shell_feature_max_degree,
                geometry,
            )
            brane_feature_shell_images = brane_cnn_features[
                :,
                self.nonlinear_head_channels : (int(gaussian_order) + 1)
                * self.nonlinear_head_channels,
            ]
            brane_feature_images = None
        rendered_image_features = predicted_residual_image[:, : self.nonlinear_head_channels]
        if brane_cnn_features is not None:
            rendered_image_features = brane_cnn_features
        else:
            brane_feature_shell_images = rendered_image_features.new_zeros(
                rendered_image_features.shape[0], 0, *rendered_image_features.shape[2:]
            )
            missing_shells = self.shell_feature_max_degree - int(gaussian_order)
            if missing_shells > 0:
                padding = rendered_image_features.new_zeros(
                    rendered_image_features.shape[0],
                    missing_shells * self.nonlinear_head_channels,
                    *rendered_image_features.shape[2:],
                )
                brane_feature_shell_images = torch.cat([brane_feature_shell_images, padding], dim=1)
            rendered_image_features = torch.cat(
                [rendered_image_features, brane_feature_shell_images], dim=1
            )
        linear_residual_image = predicted_residual_image[:, -1:]
        nonlinear_residual_image = self.image_residual_head(rendered_image_features)
        predicted_residual_image = nonlinear_residual_image
        predicted_residual_image = predicted_residual_image + linear_residual_image
        direct_image = direct_base_image + predicted_residual_image
        solver_order = int(gaussian_order) if brane_solver_coefficients is not None else 0
        solver_initial_amplitudes = predicted_amplitudes
        if solver_order > 0:
            active_solver_modes = pack_active_high_modes(
                brane_solver_coefficients, self.brane_max_degree, solver_order
            )
            solver_initial_amplitudes = torch.cat(
                [predicted_amplitudes, active_solver_modes], dim=-1
            ).contiguous()
        solver_initial_render = linear_residual_image
        direct_sino = radon.forward(direct_image.contiguous())
        direct_residual_sino = sparse_sino - direct_sino
        direct_residual_fbp = recon(direct_residual_sino.contiguous(), radon=radon)
        if not run_cgls:
            history_slots = len(self.context_step_indices)
            context_cgls_images = direct_image[:, None].expand(-1, history_slots, -1, -1, -1)
            cgls_residual_fbps = direct_residual_fbp[:, None].expand(-1, history_slots, -1, -1, -1)
            context_parts = [
                direct_image,
                direct_residual_fbp,
                context_cgls_images.flatten(1, 2),
                cgls_residual_fbps.flatten(1, 2),
            ]
            context_parts.append(
                direct_image.new_zeros(
                    direct_image.shape[0],
                    5 * self.shell_feature_max_degree,
                    *direct_image.shape[2:],
                )
            )
            raw_next_context = torch.cat(context_parts, dim=1)
            next_context = self.adapt_trajectory_context(raw_next_context)
            stage_state = {
                "geometry": geometry,
                "amplitudes": solver_initial_amplitudes,
                "image": direct_image,
                "coefficient_trajectory": None,
            }
            if not return_aux:
                result = (direct_image, next_context, stage_state)
                return result
            aux = {}
            result = (direct_image, next_context, stage_state, aux)
            return result
        if cgls_base_image is None:
            cgls_base_image = direct_base_image
        if stage_base_image is None:
            stage_base_image = cgls_base_image
        if own_iterations is None:
            own_iterations = self.cgls_iterations
        own_iterations = int(own_iterations)
        precomputed_linear_sino = None
        if inference_cache is not None:
            if image_amplitudes is None:
                precomputed_linear_sino = direct_sino
            else:
                precomputed_linear_sino = radon.forward(
                    (cgls_base_image + solver_initial_render).contiguous()
                )
        shared_solver_cache = None
        if inference_cache is not None and direct_cache_kwargs == cgls_cache_kwargs:
            shared_solver_cache = inference_cache
        elif brane_render_cache is not None and brane_cache_kwargs == cgls_cache_kwargs:
            shared_solver_cache = brane_render_cache
        own_solver = unrolled_cached_cgls(
            geometry=geometry,
            initial_amplitudes=solver_initial_amplitudes,
            base_image=cgls_base_image,
            sparse_sino=sparse_sino,
            radon=radon,
            scalar_renderer=renderer.renderer,
            iterations=own_iterations,
            render_size=self.render_size,
            cache_kwargs=cgls_cache_kwargs,
            prebuilt_cache=shared_solver_cache,
            precomputed_initial_render=(
                solver_initial_render
                if inference_cache is not None and (not solver_order > 0)
                else None
            ),
            precomputed_initial_sino=(
                precomputed_linear_sino
                if inference_cache is not None and (not solver_order > 0)
                else None
            ),
            precomputed_base_sino=None,
            gaussian_order=solver_order,
        )
        joint_solver = None
        final_solver = own_solver
        stage_geometry = geometry
        stage_amplitudes = own_solver["amplitudes"]
        own_images = own_solver["image_history"][:, 1:]
        own_sinos = own_solver["sino_history"][:, 1:]
        own_step_indices = tuple((min(index, own_iterations - 1) for index in (1, 3, 5, 7)))
        context_image_parts = [own_images[:, index] for index in own_step_indices]
        context_sino_parts = [own_sinos[:, index] for index in own_step_indices]
        context_image_parts.append(own_images[:, -1])
        context_sino_parts.append(own_sinos[:, -1])
        context_cgls_images = torch.stack(context_image_parts, dim=1)
        selected_sinos = torch.stack(context_sino_parts, dim=1)
        if self.vectorized_residual_fbp:
            residual_sinos = sparse_sino[:, None] - selected_sinos
            residual_shape = residual_sinos.shape
            flat_residual_sinos = residual_sinos.reshape(-1, *residual_shape[2:])
            flat_residual_fbps = recon(flat_residual_sinos.contiguous(), radon=radon)
            cgls_residual_fbps = flat_residual_fbps.reshape(
                residual_shape[0], residual_shape[1], *flat_residual_fbps.shape[1:]
            )
        else:
            residual_fbp_steps = []
            for selected_sino in selected_sinos.unbind(dim=1):
                residual_sino = sparse_sino - selected_sino
                residual_fbp_steps.append(recon(residual_sino.contiguous(), radon=radon))
            cgls_residual_fbps = torch.stack(residual_fbp_steps, dim=1)
        shell_context = direct_image.new_zeros(
            direct_image.shape[0], 5 * self.shell_feature_max_degree, *direct_image.shape[2:]
        )
        if solver_order > 0:
            shell_middle_step = own_iterations // 2
            shell_states = own_solver["amplitude_history"][
                :, [0, shell_middle_step, own_iterations]
            ]
            shell_images = differentiable_hermite_scalar_shell_trajectory_render(
                own_solver["render_cache"], shell_states, solver_order
            )
            shell_initial = shell_images[:, 0]
            active_shell_context = torch.cat(
                [
                    shell_initial,
                    shell_images[:, 1],
                    shell_images[:, 2],
                    shell_images[:, 1] - shell_initial,
                    shell_images[:, 2] - shell_initial,
                ],
                dim=1,
            )
            shell_context[:, : active_shell_context.shape[1]] = active_shell_context
        context_parts = [
            direct_image,
            direct_residual_fbp,
            context_cgls_images.flatten(1, 2),
            cgls_residual_fbps.flatten(1, 2),
        ]
        context_parts.append(shell_context)
        raw_next_context = torch.cat(context_parts, dim=1)
        next_context = self.adapt_trajectory_context(raw_next_context)
        selected_coefficient_indices = tuple(range(int(own_iterations) + 1))
        coefficient_trajectory = own_solver["amplitude_history"][
            :, list(selected_coefficient_indices)
        ]
        stage_state = {
            "geometry": stage_geometry,
            "amplitudes": stage_amplitudes,
            "image": final_solver["image_history"][:, -1],
            "fixed_base_image": stage_base_image,
            "coefficient_trajectory": coefficient_trajectory,
        }
        if not return_aux:
            result = (direct_image, next_context, stage_state)
            return result
        loss_initial_image = cgls_base_image + solver_initial_render
        if getattr(self, "full_hg_auxiliary_render", True):
            complete_coefficients = solver_initial_amplitudes
            loss_initial_render = renderer(
                torch.cat([geometry, complete_coefficients[..., :1]], dim=-1),
                H=self.render_size,
                W=self.render_size,
                **cgls_cache_kwargs,
            )
            if solver_order > 0:
                loss_initial_render = loss_initial_render + differentiable_hermite_high_render(
                    own_solver["render_cache"],
                    complete_coefficients[..., 1:].unsqueeze(2).contiguous(),
                    solver_order,
                    geometry,
                )
            loss_initial_image = cgls_base_image + loss_initial_render
        aux = {"cgls_initial_image_for_loss": loss_initial_image}
        result = (direct_image, next_context, stage_state, aux)
        return result


class LightweightPersistentEncoder(nn.Module):
    """Small image stem with an efficient grouped spatial mixer."""

    def __init__(self, channels, resolution, groups=2, fbp_only=False):
        super().__init__()
        self.resolution = int(resolution)
        self.fbp_only = bool(fbp_only)
        self.input_conv = nn.Conv2d(1 if self.fbp_only else 2, channels, kernel_size=3, padding=1)
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, direct_image, fbp_image=None):
        if self.fbp_only:
            value = F.interpolate(
                fbp_image,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
        else:
            value = F.interpolate(
                torch.cat([direct_image, fbp_image], dim=1),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
        value = F.gelu(self.input_conv(value))
        return value + F.gelu(self.spatial(value))


class LightweightPersistentUpdate(nn.Module):
    """Residual update using a grouped spatial mixer."""

    def __init__(self, channels, groups=2):
        super().__init__()
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, persistent, context_feature):
        update = F.gelu(self.fuse(torch.cat([persistent, context_feature], dim=1)))
        return persistent + F.gelu(self.spatial(update))


class GCTCGLSTrajectoryContextUnrolled(GCTCGLSTrajectoryContext):

    def __init__(
        self,
        render_size=256,
        stage_resolutions=(32, 64, 128, 256),
        channels=48,
        blocks_per_stage=3,
        token_channels=24,
        num_experts=6,
        stage_embed_dim=32,
        cond_dim=96,
        cgls_iterations=10,
        context_channels=12,
        trajectory_sample_stride=2,
        persistent_groups=2,
        image_head_channels=7,
        direct_amplitude_channels=8,
        max_offset_cells=0.75,
        anchor_offset_cells=0.5,
        stage_sigma_fractions=(0.5, 0.45, 0.4, 0.35),
        run_final_block_cgls=False,
        block_sigma_multipliers=None,
        gaussian_stage_orders=(3, 2, 1, 0),
        basis_mixer_channels=4,
        kdelta_width=16,
    ):
        super().__init__(
            render_size=render_size,
            stage_resolutions=stage_resolutions,
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            token_channels=token_channels,
            num_experts=num_experts,
            stage_embed_dim=stage_embed_dim,
            cond_dim=cond_dim,
            cgls_iterations=cgls_iterations,
            context_channels=context_channels,
            gradient_checkpointing=False,
            trajectory_sample_stride=trajectory_sample_stride,
            image_head_channels=image_head_channels,
            direct_amplitude_channels=direct_amplitude_channels,
            cgls_delta_context="none",
            max_offset_cells=max_offset_cells,
            anchor_offset_cells=anchor_offset_cells,
            stage_sigma_fractions=stage_sigma_fractions,
        )
        self.anchor_offset_cells = float(anchor_offset_cells)
        self.reuse_solver_channel_in_nonlinear = False
        self.gaussian_stage_orders = tuple((int(value) for value in gaussian_stage_orders))
        self.hermite_brane = True
        self.brane_shared_shape = False
        self.brane_kernel_cgls = False
        self.hermite_shell_context = True
        self.hermite_shell_nonlinear = True
        if len(self.gaussian_stage_orders) != len(self.stage_resolutions):
            raise ValueError("Gaussian stage orders must match stage resolutions")
        allowed_orders = (0, 1, 2, 3)
        if any((value not in allowed_orders for value in self.gaussian_stage_orders)):
            raise ValueError("Gaussian stage orders must be compatible with the selected renderer")
        if max(self.gaussian_stage_orders) > 0 and direct_amplitude_channels < 2:
            raise ValueError("Higher-order modes require direct nonlinear amplitudes")
        self.higher_order_cgls = True
        self.mixed_order_head = False
        self.independent_basis_head = False
        self.basis_mixer_channels = int(basis_mixer_channels)
        higher_order_mode_channels = 0
        brane_max_degree = max(self.gaussian_stage_orders)
        self.brane_max_degree = brane_max_degree
        self.shared_blocks = torch.nn.ModuleList(
            [
                UnrolledTrajectoryContextGaussianBlock(
                    channels=channels,
                    token_channels=token_channels,
                    render_size=render_size,
                    stage_embed_dim=stage_embed_dim,
                    cond_dim=cond_dim,
                    num_experts=num_experts,
                    cgls_iterations=cgls_iterations,
                    context_output_channels=self.context_channels,
                    trajectory_sample_stride=self.trajectory_sample_stride,
                    image_head_channels=self.image_head_channels,
                    direct_amplitude_channels=self.direct_amplitude_channels,
                    external_routing=True,
                    solver_only_scalar=False,
                    reuse_solver_channel_in_nonlinear=False,
                    parallel_cgls_base=True,
                    projection_safe_initialization=False,
                    zero_cgls_initial_amplitudes=False,
                    cgls_geometry_gradient=False,
                    cgls_delta_context="none",
                    higher_order_mode_channels=higher_order_mode_channels,
                    higher_order_cgls=True,
                    mixed_order_head=False,
                    fused_derivative_renderer=False,
                    independent_basis_head=False,
                    basis_mixer_channels=self.basis_mixer_channels,
                    hermite_brane=True,
                    brane_max_degree=brane_max_degree,
                    brane_shared_shape=False,
                    brane_kernel_cgls=False,
                    hermite_shell_context=True,
                    hermite_shell_nonlinear=True,
                )
                for _ in range(blocks_per_stage)
            ]
        )
        self.persistent_groups = int(persistent_groups)
        self.kdelta_width = int(kdelta_width)
        for block in self.shared_blocks:
            block.variable_kdelta_trajectory = True
        self.run_final_block_cgls = bool(run_final_block_cgls)
        if channels % self.persistent_groups != 0:
            raise ValueError("persistent groups must divide channels")
        if self.kdelta_width < 1:
            raise ValueError("KDelta-GRU width must be positive")
        if cgls_iterations != 10 or trajectory_sample_stride != 2:
            raise ValueError("Coefficient GRU requires CGLS steps 2/4/6/8/10")
        if self.stage_sigma_fractions is not None and block_sigma_multipliers is not None:
            raise ValueError("stage sigma fractions replace block sigma guidance")
        if block_sigma_multipliers is None:
            block_sigma_multipliers = (1.0,) * blocks_per_stage
        if len(block_sigma_multipliers) != blocks_per_stage:
            raise ValueError("block_sigma_multipliers must match blocks_per_stage")
        self.block_sigma_multipliers = tuple((float(value) for value in block_sigma_multipliers))
        sigma_logit_biases = []
        for multiplier in self.block_sigma_multipliers:
            target = 1.25 * multiplier
            if not 0.5 < target < 2.0:
                raise ValueError("guided initial sigma multiplier must stay in (0.5, 2.0)")
            probability = (target - 0.5) / 1.5
            sigma_logit_biases.append(math.log(probability / (1.0 - probability)))
        self.block_sigma_logit_biases = tuple(sigma_logit_biases)
        for block, bias in zip(self.shared_blocks, self.block_sigma_logit_biases):
            block.sigma_logit_bias = bias
        self.persistent_encoder = LightweightPersistentEncoder(
            channels=channels,
            resolution=self.stage_resolutions[0],
            groups=self.persistent_groups,
            fbp_only=False,
        )
        self.persistent_stage_adapters = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1) for _ in self.stage_resolutions[1:]]
        )
        for adapter in self.persistent_stage_adapters:
            nn.init.dirac_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        self.persistent_updates = None
        self.persistent_updates = nn.ModuleList(
            [
                LightweightPersistentUpdate(channels, groups=self.persistent_groups)
                for _ in range(blocks_per_stage)
            ]
        )
        self.persistent_fusions = None
        self.persistent_fusions = nn.ModuleList(
            [nn.Conv2d(2 * channels, channels, kernel_size=1) for _ in range(blocks_per_stage)]
        )
        scale = 2.0 ** (-0.5)
        for fusion in self.persistent_fusions:
            with torch.no_grad():
                fusion.weight.zero_()
                fusion.bias.zero_()
                for channel in range(channels):
                    fusion.weight[channel, channel, 0, 0] = scale
                    fusion.weight[channel, channels + channel, 0, 0] = scale
        self.svct_routers = None
        self.svct_routers = nn.ModuleList(
            [
                SVCTFactorizedRouter(
                    stage_embed_dim, channels, num_experts, view_anchor_logits=False
                )
                for _ in range(blocks_per_stage)
            ]
        )
        self.stage_blocks = None
        self.stage_persistent_updates = None
        self.stage_persistent_fusions = None
        self.stage_svct_routers = None
        stage_count = len(self.stage_resolutions)

        def expand(modules):
            if modules is None:
                return None
            return nn.ModuleList([copy.deepcopy(modules) for _ in range(stage_count)])

        self.stage_blocks = expand(self.shared_blocks)
        self.stage_persistent_updates = expand(getattr(self, "persistent_updates", None))
        self.stage_persistent_fusions = expand(getattr(self, "persistent_fusions", None))
        self.stage_svct_routers = expand(getattr(self, "svct_routers", None))
        self.shared_blocks = None
        self.persistent_updates = None
        self.persistent_fusions = None
        self.svct_routers = None
        self.pruned_inactive_hermite_parameters = _prune_inactive_hermite_heads(
            self, cond_dim=cond_dim, num_experts=num_experts
        )
        self.krylov_coefficient_gru = None
        self.krylov_coefficient_gru = KrylovDeltaGRU(
            self.gaussian_stage_orders, persistent_channels=channels, width=self.kdelta_width
        )

    def _forward_with_persistent(self, sparse_sino, RADON, return_aux=False, view_counts=None):
        radon = RADON[self.render_size]
        batch = sparse_sino.shape[0]
        if view_counts is None:
            view_counts = sparse_sino.new_full((batch,), float(sparse_sino.shape[-2]))
        view_condition = None
        fbp_image = torch.nan_to_num(
            recon(sparse_sino, radon=radon), nan=0.0, posinf=1.0, neginf=0.0
        )
        direct_image = F.interpolate(
            F.adaptive_avg_pool2d(fbp_image, (16, 16)),
            size=(self.render_size, self.render_size),
            mode="bilinear",
            align_corners=False,
        )
        context = self._initial_context(direct_image, sparse_sino, radon)
        persistent = self.persistent_encoder(direct_image, fbp_image)
        image_outputs = []
        aux_outputs = []
        for stage_idx, resolution in enumerate(self.stage_resolutions):
            if stage_idx > 0:
                flat_persistent = persistent
                flat_persistent = F.interpolate(
                    flat_persistent,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                flat_persistent = self.persistent_stage_adapters[stage_idx - 1](flat_persistent)
                persistent = flat_persistent
            stage_ids = torch.full((batch,), stage_idx, device=sparse_sino.device, dtype=torch.long)
            stage_code = self.stage_embedding(stage_ids)
            decoder = self.image_decoders[str(resolution)]
            cache_kwargs = self.CACHE_CONFIGS[resolution]
            stage_base_image = direct_image
            blocks = self.stage_blocks[stage_idx]
            persistent_updates = (
                self.stage_persistent_updates[stage_idx]
                if self.stage_persistent_updates is not None
                else self.persistent_updates
            )
            persistent_fusions = (
                self.stage_persistent_fusions[stage_idx]
                if self.stage_persistent_fusions is not None
                else self.persistent_fusions
            )
            svct_routers = (
                self.stage_svct_routers[stage_idx]
                if self.stage_svct_routers is not None
                else self.svct_routers
            )
            for block_idx, block in enumerate(blocks):
                encoder = self.context_encoders[stage_idx][block_idx]
                persistent_update = persistent_updates[block_idx]
                persistent_fusion = persistent_fusions[block_idx]
                context_feature = encoder(context, view_condition)
                persistent = persistent_update(persistent, context_feature)
                persistent_read = persistent
                fused_feature = persistent_fusion(
                    torch.cat([context_feature, persistent_read], dim=1)
                )
                routing = None
                routing_diagnostics = None
                (routing, routing_diagnostics) = svct_routers[block_idx](
                    stage_code, view_counts, context, fused_feature
                )
                is_final_cell = (
                    stage_idx == len(self.stage_resolutions) - 1 and block_idx == len(blocks) - 1
                )
                block_kwargs = {"run_cgls": self.run_final_block_cgls or not is_final_cell}
                result = block(
                    context_feature=fused_feature,
                    stage_code=stage_code,
                    direct_base_image=direct_image,
                    sparse_sino=sparse_sino,
                    radon=radon,
                    renderer=self.renderer,
                    decoder=decoder,
                    cache_kwargs=cache_kwargs,
                    routing=routing,
                    sigma_logit_bias=self.block_sigma_logit_biases[block_idx],
                    return_aux=return_aux,
                    gaussian_order=self.gaussian_stage_orders[stage_idx],
                    **block_kwargs,
                )
                if return_aux:
                    (direct_image, context, stage_state, aux) = result
                else:
                    (direct_image, context, stage_state) = result
                coefficient_trajectory = stage_state.get("coefficient_trajectory")
                if coefficient_trajectory is not None:
                    persistent = self.krylov_coefficient_gru(
                        coefficient_trajectory, persistent_read, stage_idx
                    )
                if return_aux:
                    if routing_diagnostics is not None:
                        aux.update(
                            {key: value.detach() for (key, value) in routing_diagnostics.items()}
                        )
                    aux_outputs.append(
                        {
                            **aux,
                            "cell": len(image_outputs) + 1,
                            "resolution": resolution,
                            "stage_idx": stage_idx,
                            "block_idx": block_idx,
                        }
                    )
                image_outputs.append(direct_image)
        if return_aux:
            return (image_outputs, aux_outputs)
        return image_outputs

    def forward(self, sparse_sino, RADON, return_aux=False, view_counts=None):
        return self._forward_with_persistent(
            sparse_sino, RADON, return_aux=return_aux, view_counts=view_counts
        )


GCT = GCTCGLSTrajectoryContextUnrolled
