"""Fully unrolled differentiable CGLS trajectory-context model.

Geometry is fixed inside each linear solve, while all amplitude-space CGLS
iterations remain in autograd. Cached scalar render and color-adjoint
operations are exposed as an explicit differentiable linear-operator pair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd.profiler import record_function
from torch.utils.checkpoint import checkpoint

from herkry.core.solver import batch_dot
from herkry.core.blocks import (
    GCTCGLSTrajectoryContext,
    SVCTFactorizedRouter,
    TrajectoryContextGaussianBlock,
)
from herkry.core.gaussian import FeatureGaussianDecoder
from herkry.core.solverdna import (
    KrylovCoefficientGRU,
    KrylovDeltaGRU,
    SolverDNAFactorizedRouter,
    prune_inactive_hermite_heads as _prune_inactive_hermite_heads,
    replace_condconv_with_kernel_subspace,
    replace_condconv_with_solverdna,
)
from herkry.core.gaussian_derivative import (
    coefficient_abs_mean,
    combine_derivative_render,
    combine_global_derivative_components,
    derivative_component_adjoint,
    prepare_derivative_render_amplitudes,
    prepare_cgls_render_amplitudes,
    unpack_cgls_render_adjoint,
)
from herkry.core.render_gaussian_derivative_fused import (
    augment_derivative_cache,
    differentiable_fused_basis_bank_render,
    differentiable_fused_derivative_adjoint,
    differentiable_fused_derivative_render,
    fused_direct_render_cached,
)
from herkry.core.render_gaussian_hermite_fused import (
    augment_hermite_cache,
    brane_degree_abs_mean,
    differentiable_hermite_high_render,
    differentiable_hermite_high_shell_cnn_render,
    differentiable_hermite_high_shell_render,
    differentiable_hermite_scalar_adjoint,
    differentiable_hermite_scalar_render,
    differentiable_hermite_scalar_shell_render,
    differentiable_hermite_scalar_shell_trajectory_render,
    hermite_mode_count,
    pack_active_high_modes,
)
from herkry.core.gaussian_transpose_gather_3sigma import (
    gaussian_transpose_gather_3sigma,
)
from herkry.core.utils import recon

# The exact renderer-compatible tile-envelope gather is substantially slower
# at the 128/256 lattice sizes, so production uses the shared cached adjoint.
DIRECT_GATHER_MIN_SIDE = 10**9


class CachedLinearRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, scalar_renderer, cache):
        ctx.scalar_renderer = scalar_renderer
        ctx.cache = cache
        return scalar_renderer.render_cached(cache, colors)

    @staticmethod
    def backward(ctx, grad_image):
        grad_colors = ctx.scalar_renderer.adjoint_cached(
            ctx.cache, grad_image.contiguous()
        )
        return grad_colors, None, None


class CachedLinearAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, scalar_renderer, cache):
        ctx.scalar_renderer = scalar_renderer
        ctx.cache = cache
        return scalar_renderer.adjoint_cached(
            cache, image.contiguous()
        )

    @staticmethod
    def backward(ctx, grad_colors):
        grad_image = ctx.scalar_renderer.render_cached(
            ctx.cache, grad_colors.contiguous()
        )
        return grad_image, None, None


def differentiable_cached_render(scalar_renderer, cache, colors):
    return CachedLinearRender.apply(colors, scalar_renderer, cache)


def differentiable_cached_adjoint(
    scalar_renderer, cache, image
):
    return CachedLinearAdjoint.apply(
        image, scalar_renderer, cache
    )


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
    geometry_gradient=False,
    gaussian_order=0,
    fused_derivative_renderer=False,
    hermite_brane=False,
    hermite_kernel_modes=None,
):
    """Unroll CGLS with gradients through every amplitude-space operation."""

    # Geometry defines the fixed linear operator for this solve.  Its direct
    # image-regression path remains differentiable elsewhere.
    if prebuilt_cache is None:
        with torch.no_grad(), record_function(
            "cgls_unrolled/cache_build"
        ):
            cache = scalar_renderer.build_cache(
                geometry.detach(),
                H=render_size,
                W=render_size,
                **cache_kwargs,
            )
    else:
        cache = prebuilt_cache

    gaussian_order = int(gaussian_order)
    operator_geometry = geometry.detach()
    fused_derivative_renderer = bool(
        fused_derivative_renderer and gaussian_order > 0
    )
    hermite_brane = bool(hermite_brane and gaussian_order > 0)
    if hermite_kernel_modes is not None and not hermite_brane:
        raise ValueError("Hermite kernel modes require the Hermite operator")
    if fused_derivative_renderer and hermite_brane:
        raise ValueError(
            "CGLS cannot use derivative and Hermite operators together"
        )
    if fused_derivative_renderer:
        augment_derivative_cache(cache, operator_geometry)
    if hermite_brane:
        augment_hermite_cache(cache, operator_geometry)
    if gaussian_order > 0 and geometry_gradient:
        raise ValueError(
            "Higher-order CGLS uses fixed geometry; geometry gradients are "
            "provided by the direct rendering branch"
        )

    def render(amplitudes):
        if hermite_brane:
            if hermite_kernel_modes is not None:
                return differentiable_hermite_scalar_render(
                    cache,
                    amplitudes * hermite_kernel_modes,
                    gaussian_order,
                )
            return differentiable_hermite_scalar_render(
                cache, amplitudes, gaussian_order
            )
        if fused_derivative_renderer:
            return differentiable_fused_derivative_render(
                cache, amplitudes, gaussian_order
            )
        packed = prepare_cgls_render_amplitudes(
            operator_geometry, amplitudes, gaussian_order, render_size
        )
        if geometry_gradient and torch.is_grad_enabled():
            gaussian = torch.cat([geometry, packed], dim=-1)
            return scalar_renderer(
                gaussian,
                H=render_size,
                W=render_size,
                **cache_kwargs,
            )
        rendered = differentiable_cached_render(
            scalar_renderer, cache, packed
        )
        if gaussian_order == 0:
            return rendered
        return rendered[:, :1] + combine_global_derivative_components(
            rendered[:, 1:], gaussian_order
        )

    def forward_operator(amplitudes):
        with record_function(
            "cgls_unrolled/forward_render_project"
        ):
            with record_function(
                "cgls_unrolled/render_cached"
            ):
                rendered = render(amplitudes)
            with record_function(
                "cgls_unrolled/radon_forward"
            ):
                projected = radon.forward(
                    rendered.contiguous()
                )
            return projected, rendered

    def adjoint_operator(sinogram):
        with record_function(
            "cgls_unrolled/backproject_adjoint"
        ):
            with record_function(
                "cgls_unrolled/radon_backprojection"
            ):
                backprojected = radon.backprojection(
                    sinogram.contiguous()
                )
            with record_function(
                "cgls_unrolled/gaussian_adjoint"
            ):
                gaussian_count = geometry.shape[1]
                lattice_side = math.isqrt(gaussian_count)
                if (
                    lattice_side * lattice_side
                    == gaussian_count
                    and lattice_side >= DIRECT_GATHER_MIN_SIDE
                ):
                    return gaussian_transpose_gather_3sigma(
                        geometry,
                        backprojected.contiguous(),
                        tile_size=cache_kwargs["tile_size"],
                    )
                if gaussian_order == 0:
                    return differentiable_cached_adjoint(
                        scalar_renderer, cache, backprojected
                    )
                if hermite_brane:
                    mode_adjoint = differentiable_hermite_scalar_adjoint(
                        cache, backprojected, gaussian_order
                    )
                    if hermite_kernel_modes is not None:
                        return (
                            mode_adjoint * hermite_kernel_modes
                        ).sum(dim=-1, keepdim=True)
                    return mode_adjoint
                if fused_derivative_renderer:
                    return differentiable_fused_derivative_adjoint(
                        cache, backprojected, gaussian_order
                    )
                component_images = derivative_component_adjoint(
                    backprojected, gaussian_order
                )
                packed_images = torch.cat(
                    [backprojected, component_images], dim=1
                )
                packed_adjoint = differentiable_cached_adjoint(
                    scalar_renderer, cache, packed_images
                )
                return unpack_cgls_render_adjoint(
                    operator_geometry,
                    packed_adjoint,
                    gaussian_order,
                    render_size,
                )

    with record_function("cgls_unrolled/initial_state"):
        if precomputed_initial_render is None:
            with record_function(
                "cgls_unrolled/initial_render_cached"
            ):
                initial_render = render(initial_amplitudes)
        else:
            initial_render = precomputed_initial_render
        if precomputed_base_sino is None:
            with record_function(
                "cgls_unrolled/initial_base_radon_forward"
            ):
                base_sino = radon.forward(base_image.contiguous())
        else:
            base_sino = precomputed_base_sino
        if precomputed_initial_sino is None:
            with record_function(
                "cgls_unrolled/initial_render_radon_forward"
            ):
                initial_render_sino = radon.forward(
                    initial_render.contiguous()
                )
            initial_sino = base_sino + initial_render_sino
        else:
            initial_sino = precomputed_initial_sino
    residual = sparse_sino - initial_sino
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
    pre_l1 = (
        F.l1_loss(initial_sino, sparse_sino, reduction="none")
        .flatten(1)
        .mean(1)
    )
    pre_mse = (
        F.mse_loss(initial_sino, sparse_sino, reduction="none")
        .flatten(1)
        .mean(1)
    )

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
        with record_function(
            f"cgls_unrolled/iteration_{iteration_idx + 1}"
        ):
            projected_direction, rendered_direction = (
                forward_operator(direction)
            )
            denominator = batch_dot(
                projected_direction, projected_direction
            ).clamp_min(1e-20)
            alpha = torch.nan_to_num(
                gradient_norm_sq / denominator,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            delta = (
                delta
                + alpha.view(-1, 1, 1) * direction
            )
            residual = (
                residual
                - alpha.view(-1, 1, 1, 1)
                * projected_direction
            )
            current_image = (
                current_image
                + alpha.view(-1, 1, 1, 1)
                * rendered_direction
            )
            image_history.append(current_image)
            sino_history.append(sparse_sino - residual)
            amplitude_history.append(initial_amplitudes + delta)
            mse_history.append(
                residual.square().flatten(1).mean(1)
            )
            if iteration_idx + 1 < int(iterations):
                next_gradient = adjoint_operator(residual)
                next_norm_sq = batch_dot(
                    next_gradient, next_gradient
                )
                beta = torch.nan_to_num(
                    next_norm_sq
                    / gradient_norm_sq.clamp_min(1e-20),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                direction = (
                    next_gradient
                    + beta.view(-1, 1, 1) * direction
                )
                gradient = next_gradient
                gradient_norm_sq = next_norm_sq

    final_sino = sparse_sino - residual
    post_l1 = (
        F.l1_loss(final_sino, sparse_sino, reduction="none")
        .flatten(1)
        .mean(1)
    )
    post_mse = (
        F.mse_loss(final_sino, sparse_sino, reduction="none")
        .flatten(1)
        .mean(1)
    )
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


class UnrolledTrajectoryContextGaussianBlock(
    TrajectoryContextGaussianBlock
):
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
        return_debug=False,
        cgls_base_image=None,
        stage_base_image=None,
        prior_geometry=None,
        prior_amplitudes=None,
        own_iterations=None,
        joint_refine_iterations=0,
        run_cgls=True,
        return_stage_state=False,
        return_network_feature=False,
        gaussian_order=0,
    ):
        if "direct" in cache_kwargs or "cgls" in cache_kwargs:
            direct_cache_kwargs = cache_kwargs["direct"]
            if not torch.is_grad_enabled():
                direct_cache_kwargs = cache_kwargs.get(
                    "direct_inference", direct_cache_kwargs
                )
            cgls_cache_kwargs = cache_kwargs.get(
                "cgls", cache_kwargs["direct"]
            )
            brane_cache_kwargs = cache_kwargs.get(
                "brane", direct_cache_kwargs
            )
        else:
            direct_cache_kwargs = cache_kwargs
            cgls_cache_kwargs = cache_kwargs
            brane_cache_kwargs = cache_kwargs
        with record_function("network/direct_gaussian_prediction"):
            condition = (
                stage_code
                if self.condition is None
                else self.condition(
                    torch.cat(
                        [
                            stage_code,
                            context_feature.mean(dim=(2, 3)),
                        ],
                        dim=1,
                    )
                )
            )
            feature = F.gelu(
                self.context_fuse(
                    context_feature, condition, routing=routing
                )
            )
            feature = self.process_feature(
                feature, condition, routing=routing
            )
            spatial_hidden = F.gelu(
                self.to_raw_spatial_hidden(
                    feature, condition, routing=routing
                )
            )
            feature_hidden = (
                feature
                if self.to_raw_feature_hidden is None
                else F.gelu(
                    self.to_raw_feature_hidden(
                        feature, condition, routing=routing
                    )
                )
            )
            raw_spatial = self.to_raw_spatial_out(
                spatial_hidden, condition, routing=routing
            )
            # A fixed logit offset only biases the decoder's starting scale.
            # The predicted logits can freely cancel it, so this remains soft
            # guidance rather than a hard Gaussian-size constraint.
            if sigma_logit_bias is None:
                sigma_logit_bias = getattr(
                    self, "sigma_logit_bias", 0.0
                )
            if sigma_logit_bias != 0.0:
                raw_spatial = raw_spatial.clone()
                raw_spatial[:, 2:4] = (
                    raw_spatial[:, 2:4] + sigma_logit_bias
                )
            raw_feature = self.to_raw_feature_out(
                feature_hidden, condition, routing=routing
            )
            gaussian = decoder(raw_spatial, raw_feature)
            geometry = gaussian[..., :5]
            features = gaussian[..., 5:]
            if self.direct_amplitude_channels > 0:
                predicted_amplitudes = features[:, :, -1:].contiguous()
                image_amplitudes = features[
                    :, :, : self.nonlinear_head_channels
                ].contiguous()
            else:
                mixer = self.linear_feature_to_scalar.weight.reshape(
                    1, self.token_channels, 1
                )
                predicted_amplitudes = torch.matmul(
                    features, mixer
                ).contiguous()
                image_amplitudes = None
            if self.image_feature_mixer is not None:
                image_mixer = (
                    self.image_feature_mixer.weight[:, :, 0, 0]
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                image_amplitudes = torch.matmul(
                    features, image_mixer
                ).contiguous()

            higher_order_coefficients = None
            brane_feature_coefficients = None
            brane_solver_coefficients = None
            brane_shape_coefficients = None
            if (
                self.to_raw_higher_order_out is not None
                and int(gaussian_order) > 0
            ):
                raw_higher_order = self.to_raw_higher_order_out(
                    feature_hidden, condition, routing=routing
                )
                decoded_higher = (
                    decoder.decode_bounded_feature_map(raw_higher_order)
                    if self.hermite_brane
                    else decoder.decode_feature_map(raw_higher_order)
                ).contiguous()
                if self.hermite_brane:
                    brane_vectors = (
                        1
                        if self.brane_shared_shape
                        else self.brane_feature_channels + int(
                            self.higher_order_cgls
                        )
                    )
                    decoded_higher = decoded_higher.reshape(
                        decoded_higher.shape[0],
                        decoded_higher.shape[1],
                        brane_vectors,
                        self.brane_high_mode_count,
                    )
                    if self.brane_shared_shape:
                        brane_shape_coefficients = decoded_higher[
                            :, :, 0
                        ].contiguous()
                        brane_feature_coefficients = (
                            image_amplitudes.unsqueeze(-1)
                            * brane_shape_coefficients.unsqueeze(2)
                        ).contiguous()
                        if self.higher_order_cgls:
                            brane_solver_coefficients = (
                                predicted_amplitudes
                                * brane_shape_coefficients
                            ).contiguous()
                    else:
                        brane_feature_coefficients = decoded_higher[
                            :, :, : self.brane_feature_channels
                        ].contiguous()
                        if self.higher_order_cgls:
                            brane_solver_coefficients = decoded_higher[
                                :, :, self.brane_feature_channels
                            ].contiguous()
                else:
                    higher_order_coefficients = decoded_higher

        render_amplitudes = (
            predicted_amplitudes
            if image_amplitudes is None
            else (
                features
                if self.reuse_solver_channel_in_nonlinear
                else torch.cat(
                    [image_amplitudes, predicted_amplitudes], dim=-1
                )
            )
        )
        if (
            higher_order_coefficients is not None
            and not self.independent_basis_head
        ):
            render_amplitudes = prepare_derivative_render_amplitudes(
                geometry=geometry,
                zero_order_coefficients=image_amplitudes,
                higher_order_coefficients=higher_order_coefficients,
                scalar_coefficients=predicted_amplitudes,
                order=gaussian_order,
                render_size=self.render_size,
            )

        inference_cache = None
        fused_direct_output = False
        fused_independent_output = False
        if (
            not torch.is_grad_enabled()
            and self.reuse_inference_cache
        ):
            with record_function(
                "inference/shared_cache_build"
            ):
                inference_cache = (
                    renderer.renderer.build_cache(
                        geometry,
                        H=self.render_size,
                        W=self.render_size,
                        **direct_cache_kwargs,
                    )
                )
                if (
                    self.fused_derivative_renderer
                    and higher_order_coefficients is not None
                    and int(gaussian_order) > 0
                ):
                    augment_derivative_cache(
                        inference_cache, geometry
                    )
            with record_function(
                "inference/shared_cache_direct_render"
            ):
                if (
                    self.fused_derivative_renderer
                    and higher_order_coefficients is not None
                    and int(gaussian_order) > 0
                ):
                    predicted_residual_image = fused_direct_render_cached(
                        inference_cache,
                        image_amplitudes,
                        higher_order_coefficients,
                        predicted_amplitudes,
                        gaussian_order,
                        independent_basis=(
                            self.independent_basis_head
                        ),
                    )
                    fused_direct_output = True
                    fused_independent_output = (
                        self.independent_basis_head
                    )
                else:
                    predicted_residual_image = (
                        renderer.renderer.render_cached(
                            inference_cache,
                            render_amplitudes,
                        )
                    )
        else:
            direct_gaussian = torch.cat(
                [geometry, render_amplitudes], dim=-1
            )
            with record_function(
                "network/direct_graph_render"
            ):
                predicted_residual_image = renderer(
                    direct_gaussian,
                    H=self.render_size,
                    W=self.render_size,
                    **direct_cache_kwargs,
                )
        brane_render_cache = None
        active_brane_features = None
        brane_feature_shell_images = None
        brane_cnn_features = None
        if (
            self.hermite_brane
            and brane_feature_coefficients is not None
            and int(gaussian_order) > 0
        ):
            active_brane_features = pack_active_high_modes(
                brane_feature_coefficients,
                self.brane_max_degree,
                gaussian_order,
            )
            with torch.no_grad(), record_function(
                "network/hermite_cache_build"
            ):
                if (
                    inference_cache is not None
                    and direct_cache_kwargs == brane_cache_kwargs
                ):
                    brane_render_cache = inference_cache
                else:
                    brane_render_cache = renderer.renderer.build_cache(
                        geometry.detach(),
                        H=self.render_size,
                        W=self.render_size,
                        **brane_cache_kwargs,
                    )
                augment_hermite_cache(
                    brane_render_cache, geometry.detach()
                )
            with record_function("network/hermite_direct_render"):
                if self.hermite_shell_nonlinear:
                    brane_cnn_features = (
                        differentiable_hermite_high_shell_cnn_render(
                            brane_render_cache,
                            active_brane_features,
                            gaussian_order,
                            predicted_residual_image[
                                :, : self.nonlinear_head_channels
                            ],
                            self.shell_feature_max_degree,
                            geometry,
                        )
                    )
                    brane_feature_shell_images = brane_cnn_features[
                        :,
                        self.nonlinear_head_channels : (
                            int(gaussian_order) + 1
                        )
                        * self.nonlinear_head_channels,
                    ]
                    brane_feature_images = None
                else:
                    brane_feature_images = (
                        differentiable_hermite_high_render(
                            brane_render_cache,
                            active_brane_features,
                            gaussian_order,
                            geometry,
                        )
                    )
            if not self.hermite_shell_nonlinear:
                predicted_residual_image = torch.cat(
                    [
                        predicted_residual_image[
                            :, : self.nonlinear_head_channels
                        ]
                        + brane_feature_images,
                        predicted_residual_image[
                            :, self.nonlinear_head_channels :
                        ],
                    ],
                    dim=1,
                )
        solver_derivative_image = None
        basis_response_images = None
        if self.hermite_brane:
            rendered_image_features = predicted_residual_image[
                :, : self.nonlinear_head_channels
            ]
            if self.hermite_shell_nonlinear:
                if brane_cnn_features is not None:
                    rendered_image_features = brane_cnn_features
                else:
                    brane_feature_shell_images = (
                        rendered_image_features.new_zeros(
                            rendered_image_features.shape[0],
                            0,
                            *rendered_image_features.shape[2:],
                        )
                    )
                    missing_shells = (
                        self.shell_feature_max_degree - int(gaussian_order)
                    )
                    if missing_shells > 0:
                        padding = rendered_image_features.new_zeros(
                            rendered_image_features.shape[0],
                            missing_shells * self.nonlinear_head_channels,
                            *rendered_image_features.shape[2:],
                        )
                        brane_feature_shell_images = torch.cat(
                            [brane_feature_shell_images, padding], dim=1
                        )
                    rendered_image_features = torch.cat(
                        [
                            rendered_image_features,
                            brane_feature_shell_images,
                        ],
                        dim=1,
                    )
            linear_residual_image = predicted_residual_image[:, -1:]
            nonlinear_residual_image = self.image_residual_head(
                rendered_image_features
            )
            predicted_residual_image = nonlinear_residual_image
            if not self.solver_only_scalar:
                predicted_residual_image = (
                    predicted_residual_image + linear_residual_image
                )
        elif (
            higher_order_coefficients is not None
            and self.independent_basis_head
        ):
            if fused_independent_output:
                zero_features = predicted_residual_image[
                    :, : self.nonlinear_head_channels
                ]
                basis_response_images = predicted_residual_image[
                    :, self.nonlinear_head_channels : self.nonlinear_head_channels + 5
                ]
                linear_residual_image = predicted_residual_image[:, -1:]
            else:
                zero_features = predicted_residual_image[
                    :, : self.nonlinear_head_channels
                ]
                linear_residual_image = predicted_residual_image[:, -1:]
                if int(gaussian_order) > 0:
                    with torch.no_grad(), record_function(
                        "network/fused_basis_cache_build"
                    ):
                        basis_cache = renderer.renderer.build_cache(
                            geometry.detach(),
                            H=self.render_size,
                            W=self.render_size,
                            **direct_cache_kwargs,
                        )
                        augment_derivative_cache(
                            basis_cache, geometry.detach()
                        )
                    basis_response_images = (
                        differentiable_fused_basis_bank_render(
                            basis_cache,
                            higher_order_coefficients,
                            gaussian_order,
                        )
                    )
                else:
                    basis_response_images = zero_features.new_zeros(
                        zero_features.shape[0],
                        5,
                        *zero_features.shape[2:],
                    )
            compact_basis_response = basis_response_images.sum(
                dim=1, keepdim=True
            )
            mixed_basis_responses = self.basis_response_mixer(
                basis_response_images
            )
            rendered_image_features = torch.cat(
                [
                    zero_features,
                    basis_response_images,
                    compact_basis_response,
                    mixed_basis_responses,
                ],
                dim=1,
            )
            solver_derivative_image = compact_basis_response
            nonlinear_residual_image = self.image_residual_head(
                rendered_image_features
            )
            predicted_residual_image = nonlinear_residual_image
            if not self.solver_only_scalar:
                predicted_residual_image = (
                    predicted_residual_image + linear_residual_image
                )
        elif higher_order_coefficients is not None:
            if fused_direct_output:
                zero_features = predicted_residual_image[
                    :, : self.nonlinear_head_channels
                ]
                first_order = predicted_residual_image[
                    :, self.nonlinear_head_channels : self.nonlinear_head_channels + 1
                ]
                second_order = predicted_residual_image[
                    :, self.nonlinear_head_channels + 1 : self.nonlinear_head_channels + 2
                ]
                derivative_features = (
                    torch.cat(
                        [first_order, second_order, first_order * second_order],
                        dim=1,
                    )
                    if self.mixed_order_head
                    else first_order + second_order
                )
                rendered_image_features = torch.cat(
                    [zero_features, derivative_features], dim=1
                )
                linear_residual_image = predicted_residual_image[:, -1:]
            else:
                (
                    rendered_image_features,
                    linear_residual_image,
                ) = combine_derivative_render(
                    predicted_residual_image,
                    nonlinear_channels=self.nonlinear_head_channels,
                    order=gaussian_order,
                    mode_channels=1,
                    mixed_order_features=self.mixed_order_head,
                )
            nonlinear_residual_image = self.image_residual_head(
                rendered_image_features
            )
            predicted_residual_image = nonlinear_residual_image
            if not self.solver_only_scalar:
                predicted_residual_image = (
                    predicted_residual_image + linear_residual_image
                )
        elif image_amplitudes is None:
            linear_residual_image = predicted_residual_image
            nonlinear_residual_image = torch.zeros_like(
                linear_residual_image
            )
        else:
            rendered_image_features = predicted_residual_image[
                :, : self.nonlinear_head_channels
            ]
            linear_residual_image = predicted_residual_image[:, -1:]
            nonlinear_residual_image = self.image_residual_head(
                rendered_image_features
            )
            predicted_residual_image = nonlinear_residual_image
            if not self.solver_only_scalar:
                predicted_residual_image = (
                    predicted_residual_image + linear_residual_image
                )
        direct_image = direct_base_image + predicted_residual_image
        solver_order = (
            int(gaussian_order)
            if (self.higher_order_cgls or self.brane_kernel_cgls)
            and (
                higher_order_coefficients is not None
                or brane_solver_coefficients is not None
                or brane_shape_coefficients is not None
            )
            else 0
        )
        solver_initial_amplitudes = predicted_amplitudes
        solver_kernel_modes = None
        if self.hermite_brane and self.brane_kernel_cgls and solver_order > 0:
            active_shape_modes = pack_active_high_modes(
                brane_shape_coefficients,
                self.brane_max_degree,
                solver_order,
            )
            solver_kernel_modes = torch.cat(
                [torch.ones_like(predicted_amplitudes), active_shape_modes],
                dim=-1,
            ).contiguous()
        if self.hermite_brane and solver_order > 0:
            if not self.brane_kernel_cgls:
                active_solver_modes = pack_active_high_modes(
                    brane_solver_coefficients,
                    self.brane_max_degree,
                    solver_order,
                )
                solver_initial_amplitudes = torch.cat(
                    [predicted_amplitudes, active_solver_modes], dim=-1
                ).contiguous()
        elif solver_order > 0:
            active_higher_modes = 2 if solver_order == 1 else 5
            solver_initial_amplitudes = torch.cat(
                [
                    predicted_amplitudes,
                    higher_order_coefficients[
                        :, :, :active_higher_modes
                    ],
                ],
                dim=-1,
            ).contiguous()
        if self.zero_cgls_initial_amplitudes:
            solver_initial_amplitudes = torch.zeros_like(
                predicted_amplitudes
            )
        solver_initial_render = linear_residual_image
        if solver_order > 0 and not self.hermite_brane:
            # The compact nonlinear feature's last channel is the same
            # derivative-mode image used by the multi-mode linear operator.
            if solver_derivative_image is not None:
                derivative_linear_image = solver_derivative_image
            else:
                derivative_features = rendered_image_features[
                    :, self.nonlinear_head_channels :
                ]
                derivative_linear_image = (
                    derivative_features[:, :2].sum(dim=1, keepdim=True)
                    if self.mixed_order_head
                    else derivative_features[:, -1:]
                )
            solver_initial_render = linear_residual_image + derivative_linear_image
        if self.zero_cgls_initial_amplitudes:
            solver_initial_render = torch.zeros_like(
                linear_residual_image
            )
        projection_safe_alpha = solver_initial_amplitudes.new_ones(
            solver_initial_amplitudes.shape[0]
        )
        projection_safe_raw_mse = None
        projection_safe_base_sino = None
        projection_safe_initial_sino = None
        if self.projection_safe_initialization:
            if not self.parallel_cgls_base:
                raise ValueError(
                    "Projection-safe initialization requires the parallel "
                    "CGLS base"
                )
            with record_function("cgls_unrolled/projection_safe_gate"):
                projection_safe_base_sino = radon.forward(
                    direct_base_image.contiguous()
                )
                raw_update_sino = radon.forward(
                    solver_initial_render.contiguous()
                )
                base_residual_sino = (
                    sparse_sino - projection_safe_base_sino
                )
                numerator = (
                    raw_update_sino * base_residual_sino
                ).flatten(1).sum(dim=1)
                denominator = raw_update_sino.square().flatten(1).sum(dim=1)
                raw_alpha = numerator / denominator.clamp_min(1e-20)
                bounded_alpha = raw_alpha.clamp(-1.0, 1.0)
                projection_safe_alpha = torch.where(
                    denominator > 1e-20,
                    bounded_alpha,
                    torch.ones_like(bounded_alpha),
                )
                alpha_image = projection_safe_alpha.view(-1, 1, 1, 1)
                alpha_coeff = projection_safe_alpha.view(-1, 1, 1)
                projection_safe_raw_mse = (
                    base_residual_sino - raw_update_sino
                ).square().flatten(1).mean(dim=1)
                solver_initial_amplitudes = (
                    solver_initial_amplitudes * alpha_coeff
                )
                solver_initial_render = (
                    solver_initial_render * alpha_image
                )
                projection_safe_initial_sino = (
                    projection_safe_base_sino
                    + raw_update_sino * alpha_image
                )
            # Keep the nonlinear branch parallel and untouched. Only the
            # Gaussian/Hermite linear contribution uses the analytic gate.
            predicted_residual_image = nonlinear_residual_image
            if not self.solver_only_scalar:
                predicted_residual_image = (
                    predicted_residual_image + solver_initial_render
                )
            direct_image = direct_base_image + predicted_residual_image
        brane_degree_norms = None
        if self.hermite_brane and brane_feature_coefficients is not None:
            brane_degree_norms = brane_degree_abs_mean(
                brane_feature_coefficients,
                self.brane_max_degree,
            )
            higher_order_mode_norms = predicted_amplitudes.new_zeros(
                predicted_amplitudes.shape[0], 5
            )
        elif higher_order_coefficients is None:
            higher_order_mode_norms = predicted_amplitudes.new_zeros(
                predicted_amplitudes.shape[0], 5
            )
        else:
            higher_order_mode_norms = coefficient_abs_mean(
                higher_order_coefficients,
                nonlinear_channels=1,
                order=gaussian_order,
            )

        with record_function("context/direct_residual_fbp"):
            direct_sino = radon.forward(direct_image.contiguous())
            direct_residual_sino = sparse_sino - direct_sino
            direct_residual_fbp = recon(
                direct_residual_sino.contiguous(), radon=radon
            )

        if not run_cgls:
            history_slots = len(self.context_step_indices)
            context_cgls_images = direct_image[:, None].expand(
                -1, history_slots, -1, -1, -1
            )
            cgls_residual_fbps = (
                direct_residual_fbp[:, None].expand(
                    -1, history_slots, -1, -1, -1
                )
            )
            context_parts = [
                direct_image,
                direct_residual_fbp,
                context_cgls_images.flatten(1, 2),
                cgls_residual_fbps.flatten(1, 2),
            ]
            context_delta_images = torch.zeros_like(
                context_cgls_images
            )
            if self.cgls_delta_context != "none":
                context_parts.append(
                    context_delta_images.flatten(1, 2)
                )
            if self.hermite_shell_context:
                context_parts.append(
                    direct_image.new_zeros(
                        direct_image.shape[0],
                        5 * self.shell_feature_max_degree,
                        *direct_image.shape[2:],
                    )
                )
            raw_next_context = torch.cat(context_parts, dim=1)
            next_context = self.adapt_trajectory_context(
                raw_next_context
            )
            stage_state = {
                "geometry": geometry,
                "amplitudes": solver_initial_amplitudes,
                "image": direct_image,
                "coefficient_trajectory": None,
            }
            if not return_aux:
                result = (
                    (direct_image, next_context, stage_state)
                    if return_stage_state
                    else (direct_image, next_context)
                )
                if return_network_feature:
                    return (*result, feature)
                return result

            metric_entry_sino = (
                projection_safe_base_sino
                if projection_safe_base_sino is not None
                else direct_sino
            )
            metric_direct_sino = (
                projection_safe_initial_sino
                if projection_safe_initial_sino is not None
                else direct_sino
            )
            entry_l1 = (
                F.l1_loss(
                    metric_entry_sino,
                    sparse_sino,
                    reduction="none",
                )
                .flatten(1)
                .mean(1)
            )
            entry_mse = (
                F.mse_loss(
                    metric_entry_sino,
                    sparse_sino,
                    reduction="none",
                )
                .flatten(1)
                .mean(1)
            )
            direct_l1 = (
                F.l1_loss(
                    metric_direct_sino,
                    sparse_sino,
                    reduction="none",
                )
                .flatten(1)
                .mean(1)
            )
            direct_mse = (
                F.mse_loss(
                    metric_direct_sino,
                    sparse_sino,
                    reduction="none",
                )
                .flatten(1)
                .mean(1)
            )
            amplitude_norm = (
                solver_initial_amplitudes.detach()
                .flatten(1)
                .norm(dim=1)
            )
            aux = {
                "entry_sino_l1": entry_l1.detach(),
                "direct_sino_l1": direct_l1.detach(),
                "post_cgls_sino_l1": direct_l1.detach(),
                "entry_sino_mse": entry_mse.detach(),
                "direct_sino_mse": direct_mse.detach(),
                "post_cgls_sino_mse": direct_mse.detach(),
                "cgls_mse_history": direct_mse[:, None].detach(),
                "initial_amplitude_norm": amplitude_norm,
                "optimized_amplitude_norm": amplitude_norm,
                "higher_order_mode_abs_mean": (
                    higher_order_mode_norms.detach()
                ),
                "brane_degree_abs_mean": (
                    None
                    if brane_degree_norms is None
                    else brane_degree_norms.detach()
                ),
                "gaussian_order": int(gaussian_order),
                "geometry": geometry.detach(),
                "raw_context_channels": self.raw_context_channels,
                "network_context_channels": (
                    self.context_output_channels
                ),
                "trajectory_steps": (),
                "own_cgls_iterations": 0,
                "joint_refine_iterations": 0,
                "cgls_skipped": True,
                "projection_safe_alpha": projection_safe_alpha.detach(),
                "projection_safe_raw_mse": (
                    direct_mse.detach()
                    if projection_safe_raw_mse is None
                    else projection_safe_raw_mse.detach()
                ),
                "cgls_initial_image_for_sino_loss": None,
            }
            if return_debug:
                empty_images = direct_image[:, None, :0]
                aux.update(
                    {
                        "direct_base_image": (
                            direct_base_image.detach()
                        ),
                        "predicted_residual_image": (
                            predicted_residual_image.detach()
                        ),
                        "linear_residual_image": (
                            linear_residual_image.detach()
                        ),
                        "nonlinear_residual_image": (
                            nonlinear_residual_image.detach()
                        ),
                        "predicted_amplitudes": (
                            predicted_amplitudes.detach()
                        ),
                        "higher_order_coefficients": (
                            None
                            if higher_order_coefficients is None
                            else higher_order_coefficients.detach()
                        ),
                        "brane_feature_coefficients": (
                            None
                            if brane_feature_coefficients is None
                            else brane_feature_coefficients.detach()
                        ),
                        "brane_solver_coefficients": (
                            None
                            if brane_solver_coefficients is None
                            else brane_solver_coefficients.detach()
                        ),
                        "basis_response_images": (
                            None
                            if basis_response_images is None
                            else basis_response_images.detach()
                        ),
                        "direct_image": direct_image.detach(),
                        "direct_residual_fbp": (
                            direct_residual_fbp.detach()
                        ),
                        "cgls_images": empty_images.detach(),
                        "context_cgls_images": (
                            context_cgls_images.detach()
                        ),
                        "cgls_residual_fbps": (
                            cgls_residual_fbps.detach()
                        ),
                        "context_delta_images": (
                            context_delta_images.detach()
                        ),
                    }
                )
            result = (
                (direct_image, next_context, stage_state, aux)
                if return_stage_state
                else (direct_image, next_context, aux)
            )
            if return_network_feature:
                return (*result, feature)
            return result

        if cgls_base_image is None:
            cgls_base_image = direct_base_image
        if not self.parallel_cgls_base:
            # Serial mode places the learned residual inside the solver's
            # fixed image. Parallel mode keeps the current learned and
            # physics estimates independent.
            cgls_base_image = cgls_base_image + nonlinear_residual_image
        if stage_base_image is None:
            stage_base_image = cgls_base_image
        elif not self.parallel_cgls_base:
            stage_base_image = stage_base_image + nonlinear_residual_image
        if own_iterations is None:
            own_iterations = self.cgls_iterations
        own_iterations = int(own_iterations)
        joint_refine_iterations = int(
            joint_refine_iterations
        )

        with record_function("cgls_unrolled/full_trajectory"):
            precomputed_linear_sino = None
            if inference_cache is not None:
                if image_amplitudes is None:
                    precomputed_linear_sino = direct_sino
                else:
                    precomputed_linear_sino = radon.forward(
                        (
                            cgls_base_image
                            + solver_initial_render
                        ).contiguous()
                    )
            shared_solver_cache = None
            if (
                inference_cache is not None
                and direct_cache_kwargs == cgls_cache_kwargs
            ):
                shared_solver_cache = inference_cache
            elif (
                brane_render_cache is not None
                and brane_cache_kwargs == cgls_cache_kwargs
            ):
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
                    if self.projection_safe_initialization
                    or (
                        inference_cache is not None
                        and not (self.hermite_brane and solver_order > 0)
                    )
                    else None
                ),
                precomputed_initial_sino=(
                    projection_safe_initial_sino
                    if self.projection_safe_initialization
                    else precomputed_linear_sino
                    if inference_cache is not None
                    and not (self.hermite_brane and solver_order > 0)
                    else None
                ),
                precomputed_base_sino=(
                    projection_safe_base_sino
                    if self.projection_safe_initialization
                    else None
                ),
                geometry_gradient=self.cgls_geometry_gradient,
                gaussian_order=solver_order,
                fused_derivative_renderer=(
                    self.fused_derivative_renderer
                ),
                hermite_brane=(
                    self.hermite_brane and solver_order > 0
                ),
                hermite_kernel_modes=solver_kernel_modes,
            )
            has_prior = (
                prior_geometry is not None
                and prior_amplitudes is not None
            )
            if has_prior and joint_refine_iterations > 0:
                joint_geometry = torch.cat(
                    [prior_geometry, geometry], dim=1
                )
                joint_initial_amplitudes = torch.cat(
                    [
                        prior_amplitudes,
                        own_solver["amplitudes"],
                    ],
                    dim=1,
                )
                with record_function(
                    "cgls_unrolled/stage_joint_refine"
                ):
                    joint_solver = unrolled_cached_cgls(
                        geometry=joint_geometry,
                        initial_amplitudes=(
                            joint_initial_amplitudes
                        ),
                        base_image=stage_base_image,
                        sparse_sino=sparse_sino,
                        radon=radon,
                        scalar_renderer=renderer.renderer,
                        iterations=joint_refine_iterations,
                        render_size=self.render_size,
                        cache_kwargs=cgls_cache_kwargs,
                        gaussian_order=solver_order,
                        fused_derivative_renderer=(
                            self.fused_derivative_renderer
                        ),
                        hermite_brane=(
                            self.hermite_brane and solver_order > 0
                        ),
                    )
                final_solver = joint_solver
                stage_geometry = joint_geometry
                stage_amplitudes = joint_solver["amplitudes"]
            else:
                joint_solver = None
                final_solver = own_solver
                stage_geometry = geometry
                stage_amplitudes = own_solver["amplitudes"]

            own_images = own_solver["image_history"][:, 1:]
            own_sinos = own_solver["sino_history"][:, 1:]
            # Keep five checkpoint-compatible CT/residual slots.  A shorter
            # inference solve repeats its final state, exactly like halting
            # the original trajectory early and holding the state constant.
            own_step_indices = tuple(
                min(index, own_iterations - 1)
                for index in (1, 3, 5, 7)
            )
            context_image_parts = [
                own_images[:, index]
                for index in own_step_indices
            ]
            context_sino_parts = [
                own_sinos[:, index]
                for index in own_step_indices
            ]
            if joint_solver is None:
                context_image_parts.append(own_images[:, -1])
                context_sino_parts.append(own_sinos[:, -1])
            else:
                context_image_parts.append(
                    joint_solver["image_history"][:, -1]
                )
                context_sino_parts.append(
                    joint_solver["sino_history"][:, -1]
                )
            context_cgls_images = torch.stack(
                context_image_parts, dim=1
            )
            selected_sinos = torch.stack(
                context_sino_parts, dim=1
            )
            with record_function(
                "context/unrolled_cgls_residual_fbp"
            ):
                if self.vectorized_residual_fbp:
                    residual_sinos = (
                        sparse_sino[:, None] - selected_sinos
                    )
                    residual_shape = residual_sinos.shape
                    flat_residual_sinos = (
                        residual_sinos.reshape(
                            -1, *residual_shape[2:]
                        )
                    )
                    flat_residual_fbps = recon(
                        flat_residual_sinos.contiguous(),
                        radon=radon,
                    )
                    cgls_residual_fbps = (
                        flat_residual_fbps.reshape(
                            residual_shape[0],
                            residual_shape[1],
                            *flat_residual_fbps.shape[1:],
                        )
                    )
                else:
                    residual_fbp_steps = []
                    for selected_sino in selected_sinos.unbind(
                        dim=1
                    ):
                        residual_sino = (
                            sparse_sino
                            - selected_sino
                        )
                        residual_fbp_steps.append(
                            recon(
                                residual_sino.contiguous(),
                                radon=radon,
                            )
                        )
                    cgls_residual_fbps = torch.stack(
                        residual_fbp_steps, dim=1
                    )

        shell_context = direct_image.new_zeros(
            direct_image.shape[0],
            5 * self.shell_feature_max_degree,
            *direct_image.shape[2:],
        )
        if self.hermite_shell_context and solver_order > 0:
            # Sample a stage/block-relative midpoint. Integer division is
            # intentional: for odd K, prefer the earlier Krylov iterate where
            # CGLS normally makes the larger, more informative correction.
            shell_middle_step = own_iterations // 2
            shell_states = own_solver["amplitude_history"][
                :, [0, shell_middle_step, own_iterations]
            ]
            with record_function("context/hermite_shell_render"):
                shell_images = (
                    differentiable_hermite_scalar_shell_trajectory_render(
                        own_solver["render_cache"],
                        shell_states,
                        solver_order,
                    )
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
            shell_context[:, : active_shell_context.shape[1]] = (
                active_shell_context
            )

        context_parts = [
            direct_image,
            direct_residual_fbp,
            context_cgls_images.flatten(1, 2),
            cgls_residual_fbps.flatten(1, 2),
        ]
        context_delta_images = context_cgls_images[:, :0]
        if self.cgls_delta_context == "branch_disagreement":
            context_delta_images = (
                context_cgls_images - direct_image[:, None]
            )
        elif self.cgls_delta_context == "solver_update":
            physics_initial = own_solver["image_history"][:, :1]
            context_delta_images = (
                context_cgls_images - physics_initial
            )
        if self.cgls_delta_context != "none":
            context_parts.append(context_delta_images.flatten(1, 2))
        if self.hermite_shell_context:
            context_parts.append(shell_context)
        raw_next_context = torch.cat(context_parts, dim=1)
        next_context = self.adapt_trajectory_context(
            raw_next_context
        )
        if getattr(self, "variable_kdelta_trajectory", False):
            selected_coefficient_indices = tuple(
                range(int(own_iterations) + 1)
            )
        else:
            selected_coefficient_indices = (
                0,
                *tuple(
                    min(index + 1, own_iterations)
                    for index in self.context_step_indices
                ),
            )
        coefficient_trajectory = own_solver[
            "amplitude_history"
        ][:, list(selected_coefficient_indices)]
        stage_state = {
            "geometry": stage_geometry,
            "amplitudes": stage_amplitudes,
            "image": final_solver["image_history"][:, -1],
            "fixed_base_image": stage_base_image,
            "coefficient_trajectory": coefficient_trajectory,
        }
        if not return_aux:
            if return_stage_state:
                result = (
                    direct_image,
                    next_context,
                    stage_state,
                )
            else:
                result = (direct_image, next_context)
            if return_network_feature:
                return (*result, feature)
            return result
        # Supervise the complete actual CGLS initial image, not the old
        # zero-order-only proxy. Keep live geometry gradients in this loss
        # render; the original solver/trajectory and direct path are unchanged.
        loss_initial_image = cgls_base_image + solver_initial_render
        if self.hermite_brane and not self.projection_safe_initialization and getattr(self, "full_hg_auxiliary_render", True):
            complete_coefficients = solver_initial_amplitudes
            if solver_kernel_modes is not None:
                complete_coefficients = complete_coefficients * solver_kernel_modes
            loss_initial_render = renderer(
                torch.cat([geometry, complete_coefficients[..., :1]], dim=-1),
                H=self.render_size, W=self.render_size, **cgls_cache_kwargs,
            )
            if solver_order > 0:
                loss_initial_render = loss_initial_render + differentiable_hermite_high_render(
                    own_solver["render_cache"],
                    complete_coefficients[..., 1:].unsqueeze(2).contiguous(),
                    solver_order, geometry,
                )
            loss_initial_image = cgls_base_image + loss_initial_render
        aux = {
            "entry_sino_l1": own_solver["base_l1"].detach(),
            "direct_sino_l1": own_solver["pre_l1"].detach(),
            "post_cgls_sino_l1": final_solver["post_l1"].detach(),
            "entry_sino_mse": own_solver["base_mse"].detach(),
            "direct_sino_mse": own_solver["pre_mse"].detach(),
            "post_cgls_sino_mse": final_solver["post_mse"].detach(),
            "cgls_mse_history": torch.cat(
                [
                    own_solver["mse_history"],
                    (
                        joint_solver["mse_history"][:, 1:]
                        if joint_solver is not None
                        else own_solver["mse_history"][:, :0]
                    ),
                ],
                dim=1,
            ).detach(),
            "initial_amplitude_norm": (
                solver_initial_amplitudes.detach().flatten(1).norm(dim=1)
            ),
            "optimized_amplitude_norm": (
                stage_amplitudes.detach().flatten(1).norm(dim=1)
            ),
            "higher_order_mode_abs_mean": (
                higher_order_mode_norms.detach()
            ),
            "brane_degree_abs_mean": (
                None
                if brane_degree_norms is None
                else brane_degree_norms.detach()
            ),
            "gaussian_order": int(gaussian_order),
            "geometry": geometry.detach(),
            "raw_context_channels": self.raw_context_channels,
            "network_context_channels": self.context_output_channels,
            "trajectory_steps": tuple(
                [2, 4, 6, 8, "joint4_or_repeat8"]
            ),
            "own_cgls_iterations": own_iterations,
            "joint_refine_iterations": (
                joint_refine_iterations if has_prior else 0
            ),
            # Kept in graph only when the training loop explicitly requests
            # auxiliary supervision.
            "cgls_initial_image_for_loss": loss_initial_image,
            "projection_safe_alpha": projection_safe_alpha.detach(),
            "projection_safe_raw_mse": (
                own_solver["pre_mse"].detach()
                if projection_safe_raw_mse is None
                else projection_safe_raw_mse.detach()
            ),
            # Keep this differentiable.  Training may impose an auxiliary
            # dense-view data-fidelity loss on the raw linear initialization.
            "cgls_initial_image_for_sino_loss": (
                cgls_base_image + solver_initial_render
            ),
        }
        if return_debug:
            aux.update(
                {
                    "direct_base_image": direct_base_image.detach(),
                    "predicted_residual_image": (
                        predicted_residual_image.detach()
                    ),
                    "linear_residual_image": (
                        linear_residual_image.detach()
                    ),
                    "nonlinear_residual_image": (
                        nonlinear_residual_image.detach()
                    ),
                    "predicted_amplitudes": (
                        predicted_amplitudes.detach()
                    ),
                    "higher_order_coefficients": (
                        None
                        if higher_order_coefficients is None
                        else higher_order_coefficients.detach()
                    ),
                    "brane_feature_coefficients": (
                        None
                        if brane_feature_coefficients is None
                        else brane_feature_coefficients.detach()
                    ),
                    "brane_solver_coefficients": (
                        None
                        if brane_solver_coefficients is None
                        else brane_solver_coefficients.detach()
                    ),
                    "basis_response_images": (
                        None
                        if basis_response_images is None
                        else basis_response_images.detach()
                    ),
                    "direct_image": direct_image.detach(),
                    "direct_residual_fbp": (
                        direct_residual_fbp.detach()
                    ),
                    "cgls_images": own_images.detach(),
                    "context_cgls_images": (
                        context_cgls_images.detach()
                    ),
                    "cgls_residual_fbps": (
                        cgls_residual_fbps.detach()
                    ),
                    "context_delta_images": (
                        context_delta_images.detach()
                    ),
                }
            )
        if return_stage_state:
            result = (
                direct_image,
                next_context,
                stage_state,
                aux,
            )
        else:
            result = (direct_image, next_context, aux)
        if return_network_feature:
            return (*result, feature)
        return result


class LightweightPersistentEncoder(nn.Module):
    """Small image stem with an efficient grouped spatial mixer."""

    def __init__(self, channels, resolution, groups=2, fbp_only=False):
        super().__init__()
        self.resolution = int(resolution)
        self.fbp_only = bool(fbp_only)
        self.input_conv = nn.Conv2d(
            1 if self.fbp_only else 2, channels, kernel_size=3, padding=1
        )
        self.spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=groups,
        )

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
        self.fuse = nn.Conv2d(
            2 * channels, channels, kernel_size=1
        )
        self.spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=groups,
        )

    def forward(self, persistent, context_feature):
        update = F.gelu(
            self.fuse(
                torch.cat([persistent, context_feature], dim=1)
            )
        )
        return persistent + F.gelu(self.spatial(update))


class PhysicsResidualEncoder(nn.Module):
    """Encode the residual-FBP progress that should control memory writes."""

    def __init__(self, channels, groups=2):
        super().__init__()
        self.input_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.spatial = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=groups,
        )

    def forward(self, raw_context, output_size):
        direct_residual = raw_context[:, 1:2]
        final_residual = raw_context[:, -1:]
        physics = torch.cat(
            [
                direct_residual,
                final_residual,
                direct_residual - final_residual,
            ],
            dim=1,
        )
        physics = F.interpolate(
            physics,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        physics = F.gelu(self.input_conv(physics))
        return physics + F.gelu(self.spatial(physics))


class PhysicsGatedPersistentUpdate(nn.Module):
    """Single-slot residual memory with stage-aware spatial write gates."""

    def __init__(
        self,
        channels,
        num_stages,
        groups=2,
    ):
        super().__init__()
        merged_channels = 3 * channels
        self.candidate_fuse = nn.Conv2d(
            merged_channels, channels, 1
        )
        self.candidate_spatial = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=groups,
        )
        self.gate_fuse = nn.Conv2d(
            merged_channels, channels, 1
        )
        self.gate_spatial = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=groups,
        )
        self.stage_gate_bias = nn.Parameter(
            torch.zeros(num_stages, channels)
        )
        nn.init.zeros_(self.gate_spatial.weight)
        nn.init.constant_(self.gate_spatial.bias, -1.0)

    def forward(
        self,
        persistent,
        context_feature,
        physics_feature,
        stage_idx,
    ):
        merged = torch.cat(
            [persistent, context_feature, physics_feature], dim=1
        )
        candidate = self.candidate_spatial(
            F.gelu(self.candidate_fuse(merged))
        )
        gate_logits = self.gate_spatial(
            F.gelu(self.gate_fuse(merged))
        )
        gate_logits = gate_logits + self.stage_gate_bias[
            stage_idx
        ][None, :, None, None]
        gate = torch.sigmoid(gate_logits)
        updated = persistent + gate * candidate
        diagnostics = {
            "memory_write_gate": gate.mean(dim=(2, 3)),
            "memory_update_l1": candidate.abs().mean(
                dim=(1, 2, 3), keepdim=True
            ),
            "memory_physics_l1": physics_feature.abs().mean(
                dim=(1, 2, 3), keepdim=True
            ),
        }
        return updated, diagnostics


class PhysicsGuidedMultiScaleMemoryFlow(nn.Module):
    """Three coarse/mid/fine slots gated by residual-FBP physics."""

    def __init__(self, channels, blocks_per_stage, groups=2):
        super().__init__()
        if blocks_per_stage != 3:
            raise ValueError(
                "Physics memory requires three blocks per stage"
            )
        self.channels = int(channels)
        self.read_query = nn.Linear(channels + 7, 3)
        self.write_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(7, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                )
                for _ in range(3)
            ]
        )
        self.write_fuse = nn.ModuleList(
            [
                nn.Conv2d(2 * channels, channels, 1)
                for _ in range(3)
            ]
        )
        self.write_spatial = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=1,
                    groups=groups,
                )
                for _ in range(3)
            ]
        )
        self.slot_bias = nn.Parameter(
            torch.zeros(3, channels, 1, 1)
        )
        for gate in self.write_gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, -1.0)

    @staticmethod
    def initialize(base):
        coarse = F.avg_pool2d(base, 5, stride=1, padding=2)
        smooth = F.avg_pool2d(base, 3, stride=1, padding=1)
        mid = smooth - coarse
        fine = base - smooth
        return torch.stack([coarse, mid, fine], dim=1)

    @staticmethod
    def physics_vector(raw_context):
        residual_maps = torch.cat(
            [raw_context[:, 1:2], raw_context[:, -5:]], dim=1
        )
        energies = residual_maps.abs().mean(dim=(2, 3))
        relative_drop = (
            energies[:, :1] - energies[:, -1:]
        ) / energies[:, :1].clamp_min(1.0e-6)
        return torch.cat(
            [torch.log1p(10.0 * energies), relative_drop], dim=1
        )

    @staticmethod
    def filter_write(candidate, slot_idx):
        if slot_idx == 0:
            return F.avg_pool2d(
                candidate, 5, stride=1, padding=2
            )
        if slot_idx == 1:
            smooth3 = F.avg_pool2d(
                candidate, 3, stride=1, padding=1
            )
            smooth5 = F.avg_pool2d(
                candidate, 5, stride=1, padding=2
            )
            return smooth3 - smooth5
        return candidate - F.avg_pool2d(
            candidate, 3, stride=1, padding=1
        )

    def forward(
        self, memory, context_feature, raw_context, block_idx
    ):
        physics = self.physics_vector(raw_context)
        query = torch.cat(
            [context_feature.mean(dim=(2, 3)), physics], dim=1
        )
        read_weights = torch.softmax(
            self.read_query(query), dim=1
        )
        biased_memory = memory + self.slot_bias.unsqueeze(0)
        read = (
            biased_memory
            * read_weights[:, :, None, None, None]
        ).sum(dim=1)
        candidate = F.gelu(
            self.write_fuse[block_idx](
                torch.cat([read, context_feature], dim=1)
            )
        )
        candidate = F.gelu(
            self.write_spatial[block_idx](candidate)
        )
        candidate = self.filter_write(candidate, block_idx)
        gate = torch.sigmoid(
            self.write_gates[block_idx](physics)
        )[:, :, None, None]
        slots = list(memory.unbind(dim=1))
        slots[block_idx] = slots[block_idx] + gate * candidate
        memory = torch.stack(slots, dim=1)
        diagnostics = {
            "memory_read_weights": read_weights,
            "memory_write_gate": gate.flatten(1),
            "memory_physics": physics,
        }
        return memory, read, diagnostics


class GCTCGLSTrajectoryContextUnrolled(
    GCTCGLSTrajectoryContext
):
    def __init__(
        self,
        render_size=256,
        stage_resolutions=(32, 64, 128, 256),
        channels=48,
        blocks_per_stage=2,
        token_channels=24,
        num_experts=6,
        stage_embed_dim=32,
        cond_dim=96,
        cgls_iterations=5,
        context_channels=None,
        gradient_checkpointing=False,
        trajectory_sample_stride=1,
        use_persistent_feature=False,
        learned_persistent_fusion=False,
        propagate_block_feature=False,
        persistent_groups=2,
        physics_memory_flow=False,
        physics_gated_memory=False,
        svct_factorized_moe=False,
        view_conditioned_context_encoder=False,
        svct_view_anchor_router=False,
        image_head_channels=0,
        direct_amplitude_channels=0,
        solver_only_scalar=False,
        reuse_solver_channel_in_nonlinear=False,
        parallel_cgls_base=False,
        projection_safe_initialization=False,
        zero_cgls_initial_amplitudes=False,
        cgls_geometry_gradient=False,
        cgls_delta_context="none",
        max_offset_cells=0.5,
        anchor_offset_cells=0.5,
        stage_sigma_fractions=None,
        run_final_block_cgls=True,
        block_sigma_multipliers=None,
        stage_own_iterations=None,
        stage_joint_refine_iterations=0,
        zero_initial_image=False,
        gaussian_stage_orders=(0, 0, 0, 0),
        stage_independent_blocks=False,
        higher_order_cgls=False,
        mixed_order_head=False,
        fused_derivative_renderer=False,
        independent_basis_head=False,
        basis_mixer_channels=4,
        hermite_brane=False,
        brane_shared_shape=False,
        brane_kernel_cgls=False,
        coefficient_gru=False,
        kdelta_gru=False,
        kdelta_width=16,
        variable_kdelta_trajectory=False,
        prune_inactive_hermite_heads=False,
        solver_dna=False,
        solver_dna_rank=16,
        solver_dna_router_dim=16,
        dynamic_kernel_subspace=False,
        kernel_subspace_bases=4,
        kernel_subspace_rank=4,
        hermite_shell_context=False,
        hermite_shell_nonlinear=False,
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
            gradient_checkpointing=gradient_checkpointing,
            trajectory_sample_stride=trajectory_sample_stride,
            image_head_channels=image_head_channels,
            direct_amplitude_channels=direct_amplitude_channels,
            cgls_delta_context=cgls_delta_context,
            max_offset_cells=max_offset_cells,
            anchor_offset_cells=anchor_offset_cells,
            stage_sigma_fractions=stage_sigma_fractions,
        )
        self.anchor_offset_cells = float(anchor_offset_cells)
        self.zero_initial_image = bool(zero_initial_image)
        self.solver_only_scalar = bool(solver_only_scalar)
        self.reuse_solver_channel_in_nonlinear = bool(
            reuse_solver_channel_in_nonlinear
        )
        self.parallel_cgls_base = bool(parallel_cgls_base)
        self.projection_safe_initialization = bool(
            projection_safe_initialization
        )
        if self.projection_safe_initialization and not self.parallel_cgls_base:
            raise ValueError(
                "Projection-safe initialization requires parallel CGLS"
            )
        self.zero_cgls_initial_amplitudes = bool(
            zero_cgls_initial_amplitudes
        )
        self.cgls_geometry_gradient = bool(
            cgls_geometry_gradient
        )
        self.gaussian_stage_orders = tuple(
            int(value) for value in gaussian_stage_orders
        )
        self.hermite_brane = bool(hermite_brane)
        self.brane_shared_shape = bool(brane_shared_shape)
        self.brane_kernel_cgls = bool(brane_kernel_cgls)
        self.hermite_shell_context = bool(hermite_shell_context)
        self.hermite_shell_nonlinear = bool(
            hermite_shell_nonlinear
        )
        if (
            self.hermite_shell_context
            or self.hermite_shell_nonlinear
        ) and not self.hermite_brane:
            raise ValueError(
                "Hermite shell features require the Hermite Brane"
            )
        if len(self.gaussian_stage_orders) != len(
            self.stage_resolutions
        ):
            raise ValueError(
                "Gaussian stage orders must match stage resolutions"
            )
        allowed_orders = (0, 1, 2, 3) if self.hermite_brane else (0, 1, 2)
        if any(value not in allowed_orders for value in self.gaussian_stage_orders):
            raise ValueError(
                "Gaussian stage orders must be compatible with the selected renderer"
            )
        if max(self.gaussian_stage_orders) > 0 and direct_amplitude_channels < 2:
            raise ValueError(
                "Higher-order modes require direct nonlinear amplitudes"
            )
        if (
            max(self.gaussian_stage_orders) > 0
            and reuse_solver_channel_in_nonlinear
        ):
            raise ValueError(
                "Higher-order modes keep the scalar solver channel separate"
            )
        self.stage_independent_blocks = bool(stage_independent_blocks)
        self.higher_order_cgls = bool(higher_order_cgls)
        self.mixed_order_head = bool(mixed_order_head)
        self.fused_derivative_renderer = bool(
            fused_derivative_renderer
        )
        self.independent_basis_head = bool(independent_basis_head)
        self.basis_mixer_channels = int(basis_mixer_channels)
        if self.hermite_brane and (
            self.mixed_order_head
            or self.independent_basis_head
            or self.fused_derivative_renderer
        ):
            raise ValueError(
                "Hermite Brane replaces legacy derivative/mixed basis paths"
            )
        if self.independent_basis_head and self.mixed_order_head:
            raise ValueError(
                "Independent basis head replaces the R1R2 mixed-order head"
            )
        if self.mixed_order_head and max(self.gaussian_stage_orders) < 2:
            raise ValueError("Mixed-order head requires at least one order-2 stage")
        higher_order_mode_channels = (
            1
            if max(self.gaussian_stage_orders) > 0 and not self.hermite_brane
            else 0
        )
        brane_max_degree = (
            max(self.gaussian_stage_orders) if self.hermite_brane else 0
        )
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
                    trajectory_sample_stride=(
                        self.trajectory_sample_stride
                    ),
                    image_head_channels=self.image_head_channels,
                    direct_amplitude_channels=(
                        self.direct_amplitude_channels
                    ),
                    external_routing=svct_factorized_moe,
                    solver_only_scalar=solver_only_scalar,
                    reuse_solver_channel_in_nonlinear=(
                        reuse_solver_channel_in_nonlinear
                    ),
                    parallel_cgls_base=parallel_cgls_base,
                    projection_safe_initialization=(
                        projection_safe_initialization
                    ),
                    zero_cgls_initial_amplitudes=(
                        zero_cgls_initial_amplitudes
                    ),
                    cgls_geometry_gradient=cgls_geometry_gradient,
                    cgls_delta_context=cgls_delta_context,
                    higher_order_mode_channels=(
                        higher_order_mode_channels
                    ),
                    higher_order_cgls=self.higher_order_cgls,
                    mixed_order_head=self.mixed_order_head,
                    fused_derivative_renderer=(
                        self.fused_derivative_renderer
                    ),
                    independent_basis_head=(
                        self.independent_basis_head
                    ),
                    basis_mixer_channels=self.basis_mixer_channels,
                    hermite_brane=self.hermite_brane,
                    brane_max_degree=brane_max_degree,
                    brane_shared_shape=self.brane_shared_shape,
                    brane_kernel_cgls=self.brane_kernel_cgls,
                    hermite_shell_context=self.hermite_shell_context,
                    hermite_shell_nonlinear=(
                        self.hermite_shell_nonlinear
                    ),
                )
                for _ in range(blocks_per_stage)
            ]
        )
        self.use_persistent_feature = bool(
            use_persistent_feature
        )
        self.learned_persistent_fusion = bool(
            learned_persistent_fusion
        )
        self.propagate_block_feature = bool(
            propagate_block_feature
        )
        self.persistent_groups = int(persistent_groups)
        self.physics_memory_flow = bool(physics_memory_flow)
        self.physics_gated_memory = bool(physics_gated_memory)
        self.svct_factorized_moe = bool(svct_factorized_moe)
        self.view_conditioned_context_encoder = bool(
            view_conditioned_context_encoder
        )
        self.svct_view_anchor_router = bool(svct_view_anchor_router)
        self.use_coefficient_gru = bool(coefficient_gru)
        self.use_kdelta_gru = bool(kdelta_gru)
        self.kdelta_width = int(kdelta_width)
        self.variable_kdelta_trajectory = bool(
            variable_kdelta_trajectory
        )
        self.use_coefficient_memory = (
            self.use_coefficient_gru or self.use_kdelta_gru
        )
        for block in self.shared_blocks:
            block.variable_kdelta_trajectory = (
                self.variable_kdelta_trajectory
            )
        self.prune_inactive_hermite_heads = bool(
            prune_inactive_hermite_heads
        )
        self.solver_dna = bool(solver_dna)
        self.solver_dna_rank = int(solver_dna_rank)
        self.solver_dna_router_dim = int(solver_dna_router_dim)
        self.dynamic_kernel_subspace = bool(dynamic_kernel_subspace)
        self.kernel_subspace_bases = int(kernel_subspace_bases)
        self.kernel_subspace_rank = int(kernel_subspace_rank)
        self.run_final_block_cgls = bool(
            run_final_block_cgls
        )
        if channels % self.persistent_groups != 0:
            raise ValueError(
                "persistent groups must divide channels"
            )
        if (
            (
                self.learned_persistent_fusion
                or self.propagate_block_feature
            )
            and not self.use_persistent_feature
        ):
            raise ValueError(
                "Persistent fusion/propagation requires persistent features"
            )
        if self.physics_memory_flow and (
            not self.use_persistent_feature
            or self.propagate_block_feature
        ):
            raise ValueError(
                "Physics memory requires persistent features and cannot "
                "use direct block-feature propagation"
            )
        if self.physics_gated_memory and (
            not self.use_persistent_feature
            or self.propagate_block_feature
            or self.physics_memory_flow
        ):
            raise ValueError(
                "Physics-gated memory requires one persistent feature slot "
                "and is mutually exclusive with other memory modes"
            )
        if self.svct_factorized_moe and not self.use_persistent_feature:
            raise ValueError(
                "SVCT factorized routing currently requires the persistent "
                "forward path"
            )
        if (
            self.view_conditioned_context_encoder
            and not self.svct_factorized_moe
        ):
            raise ValueError(
                "View-conditioned context encoders require SVCT routing"
            )
        if self.svct_view_anchor_router and not self.svct_factorized_moe:
            raise ValueError("View-anchor routing requires SVCT routing")
        if self.use_coefficient_gru and self.use_kdelta_gru:
            raise ValueError(
                "Coefficient GRU and KDelta-GRU are mutually exclusive"
            )
        if self.kdelta_width < 1:
            raise ValueError("KDelta-GRU width must be positive")
        if self.use_coefficient_memory and (
            not self.use_persistent_feature
            or self.propagate_block_feature
            or self.physics_memory_flow
            or self.physics_gated_memory
        ):
            raise ValueError(
                "Coefficient GRU requires the single-slot persistent path"
            )
        if self.use_coefficient_memory and (
            cgls_iterations != 10 or trajectory_sample_stride != 2
        ):
            raise ValueError(
                "Coefficient GRU requires CGLS steps 2/4/6/8/10"
            )
        if self.solver_dna and (
            not self.use_coefficient_memory
            or not self.stage_independent_blocks
            or not self.svct_factorized_moe
        ):
            raise ValueError(
                "SolverDNA requires coefficient GRU, stage-independent "
                "blocks, and factorized routing"
            )
        if self.dynamic_kernel_subspace and (
            not self.use_kdelta_gru
            or not self.stage_independent_blocks
            or not self.svct_factorized_moe
        ):
            raise ValueError(
                "Dynamic kernel subspace requires KDelta-GRU, "
                "stage-independent blocks, and factorized routing"
            )
        if self.dynamic_kernel_subspace and self.solver_dna:
            raise ValueError(
                "Dynamic kernel subspace and SolverDNA are mutually exclusive"
            )
        if self.kernel_subspace_bases < 1 or self.kernel_subspace_rank < 1:
            raise ValueError(
                "Kernel-subspace basis count and residual rank must be positive"
            )
        self.stage_own_iterations = int(
            cgls_iterations
            if stage_own_iterations is None
            else stage_own_iterations
        )
        self.stage_joint_refine_iterations = int(
            stage_joint_refine_iterations
        )
        self.stage_joint_schedule = (
            self.stage_joint_refine_iterations > 0
        )
        if self.stage_joint_schedule:
            if blocks_per_stage != 3:
                raise ValueError(
                    "Stage-joint schedule currently requires 3 blocks"
                )
            if self.stage_own_iterations != 8:
                raise ValueError(
                    "Stage-joint schedule is defined as own8+joint4"
                )
            if self.stage_joint_refine_iterations != 4:
                raise ValueError(
                    "Stage-joint schedule is defined as own8+joint4"
                )
            if not self.use_persistent_feature:
                raise ValueError(
                    "Stage-joint schedule currently uses the persistent "
                    "forward path"
                )
        if self.use_coefficient_memory and self.stage_joint_schedule:
            raise ValueError(
                "Coefficient GRU currently uses the fixed 10-step schedule"
            )
        if (
            self.stage_sigma_fractions is not None
            and block_sigma_multipliers is not None
        ):
            raise ValueError(
                "stage sigma fractions replace block sigma guidance"
            )
        if block_sigma_multipliers is None:
            block_sigma_multipliers = (1.0,) * blocks_per_stage
        if len(block_sigma_multipliers) != blocks_per_stage:
            raise ValueError(
                "block_sigma_multipliers must match blocks_per_stage"
            )
        self.block_sigma_multipliers = tuple(
            float(value) for value in block_sigma_multipliers
        )
        sigma_logit_biases = []
        for multiplier in self.block_sigma_multipliers:
            target = 1.25 * multiplier
            if not 0.5 < target < 2.0:
                raise ValueError(
                    "guided initial sigma multiplier must stay in (0.5, 2.0)"
                )
            probability = (target - 0.5) / 1.5
            sigma_logit_biases.append(
                math.log(probability / (1.0 - probability))
            )
        self.block_sigma_logit_biases = tuple(sigma_logit_biases)
        for block, bias in zip(
            self.shared_blocks, self.block_sigma_logit_biases
        ):
            block.sigma_logit_bias = bias
        if self.use_persistent_feature:
            self.persistent_encoder = LightweightPersistentEncoder(
                channels=channels,
                resolution=self.stage_resolutions[0],
                groups=self.persistent_groups,
                fbp_only=self.zero_initial_image,
            )
            self.persistent_stage_adapters = nn.ModuleList(
                [
                    nn.Conv2d(
                        channels, channels, kernel_size=1
                    )
                    for _ in self.stage_resolutions[1:]
                ]
            )
            for adapter in self.persistent_stage_adapters:
                nn.init.dirac_(adapter.weight)
                nn.init.zeros_(adapter.bias)
            self.persistent_updates = None
            if (
                not self.propagate_block_feature
                and not self.physics_memory_flow
                and not self.physics_gated_memory
            ):
                self.persistent_updates = nn.ModuleList(
                    [
                        LightweightPersistentUpdate(
                            channels,
                            groups=self.persistent_groups,
                        )
                        for _ in range(blocks_per_stage)
                    ]
                )
            self.physics_memory = None
            if self.physics_memory_flow:
                self.physics_memory = (
                    PhysicsGuidedMultiScaleMemoryFlow(
                        channels,
                        blocks_per_stage,
                        self.persistent_groups,
                    )
                )
            self.physics_residual_encoder = None
            self.physics_gated_updates = None
            if self.physics_gated_memory:
                self.physics_residual_encoder = PhysicsResidualEncoder(
                    channels,
                    groups=self.persistent_groups,
                )
                self.physics_gated_updates = nn.ModuleList(
                    [
                        PhysicsGatedPersistentUpdate(
                            channels,
                            len(self.stage_resolutions),
                            groups=self.persistent_groups,
                        )
                        for _ in range(blocks_per_stage)
                    ]
                )
            self.persistent_fusions = None
            if self.learned_persistent_fusion:
                self.persistent_fusions = nn.ModuleList(
                    [
                        nn.Conv2d(
                            2 * channels,
                            channels,
                            kernel_size=1,
                        )
                        for _ in range(blocks_per_stage)
                    ]
                )
                scale = 2.0**-0.5
                for fusion in self.persistent_fusions:
                    with torch.no_grad():
                        fusion.weight.zero_()
                        fusion.bias.zero_()
                        for channel in range(channels):
                            fusion.weight[
                                channel, channel, 0, 0
                            ] = scale
                            fusion.weight[
                                channel,
                                channels + channel,
                                0,
                                0,
                            ] = scale
        self.svct_routers = None
        if self.svct_factorized_moe:
            self.svct_routers = nn.ModuleList(
                [
                    SVCTFactorizedRouter(
                        stage_embed_dim,
                        channels,
                        num_experts,
                        view_anchor_logits=self.svct_view_anchor_router,
                    )
                    for _ in range(blocks_per_stage)
                ]
            )
        # Add zero-initialized FiLM only after all baseline modules have been
        # constructed, preserving their seeded initialization exactly.
        if self.view_conditioned_context_encoder:
            for stage_encoders in self.context_encoders:
                for encoder in stage_encoders:
                    encoder.enable_view_conditioning(condition_dim=2)

        self.stage_blocks = None
        self.stage_persistent_updates = None
        self.stage_physics_gated_updates = None
        self.stage_persistent_fusions = None
        self.stage_svct_routers = None
        if self.stage_independent_blocks:
            stage_count = len(self.stage_resolutions)

            def expand(modules):
                if modules is None:
                    return None
                return nn.ModuleList(
                    [copy.deepcopy(modules) for _ in range(stage_count)]
                )

            # Deep copies begin from exactly the same seeded function as the
            # shared model while registering independent parameters per stage.
            self.stage_blocks = expand(self.shared_blocks)
            self.stage_persistent_updates = expand(
                getattr(self, "persistent_updates", None)
            )
            self.stage_physics_gated_updates = expand(
                getattr(self, "physics_gated_updates", None)
            )
            self.stage_persistent_fusions = expand(
                getattr(self, "persistent_fusions", None)
            )
            self.stage_svct_routers = expand(
                getattr(self, "svct_routers", None)
            )
            self.shared_blocks = None
            self.persistent_updates = None
            self.physics_gated_updates = None
            self.persistent_fusions = None
            self.svct_routers = None

        self.pruned_inactive_hermite_parameters = 0
        if self.prune_inactive_hermite_heads:
            self.pruned_inactive_hermite_parameters = (
                _prune_inactive_hermite_heads(
                    self,
                    cond_dim=cond_dim,
                    num_experts=num_experts,
                )
            )

        self.krylov_coefficient_gru = None
        if self.use_coefficient_gru:
            self.krylov_coefficient_gru = KrylovCoefficientGRU(
                self.gaussian_stage_orders,
                hidden_channels=channels,
                width=16,
            )
        elif self.use_kdelta_gru:
            self.krylov_coefficient_gru = KrylovDeltaGRU(
                self.gaussian_stage_orders,
                persistent_channels=channels,
                width=self.kdelta_width,
            )

        self.solver_dna_conv_count = 0
        self.solver_dna_parent_groups = 0
        if self.solver_dna:
            self.stage_svct_routers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            SolverDNAFactorizedRouter(
                                stage_embed_dim,
                                channels,
                                code_dim=self.solver_dna_router_dim,
                                block_idx=block_idx,
                            )
                            for block_idx in range(blocks_per_stage)
                        ]
                    )
                    for _ in self.stage_resolutions
                ]
            )
            (
                self.solver_dna_conv_count,
                self.solver_dna_parent_groups,
            ) = replace_condconv_with_solverdna(
                self,
                rank=self.solver_dna_rank,
                router_dim=self.solver_dna_router_dim,
                maximum_groups=2,
            )

        self.kernel_subspace_conv_count = 0
        self.kernel_subspace_shared_groups = 0
        if self.dynamic_kernel_subspace:
            (
                self.kernel_subspace_conv_count,
                self.kernel_subspace_shared_groups,
            ) = replace_condconv_with_kernel_subspace(
                self,
                num_bases=self.kernel_subspace_bases,
                residual_rank=self.kernel_subspace_rank,
            )

    def _forward_cell_checkpointed(
        self,
        context,
        direct_image,
        encoder,
        block,
        stage_code,
        sparse_sino,
        radon,
        decoder,
        cache_kwargs,
        sigma_logit_bias=None,
        gaussian_order=0,
    ):
        def cell_forward(context_input, direct_image_input):
            context_feature = encoder(context_input)
            return block(
                context_feature=context_feature,
                stage_code=stage_code,
                direct_base_image=direct_image_input,
                sparse_sino=sparse_sino,
                radon=radon,
                renderer=self.renderer,
                decoder=decoder,
                cache_kwargs=cache_kwargs,
                sigma_logit_bias=sigma_logit_bias,
                gaussian_order=gaussian_order,
                return_aux=False,
                return_debug=False,
            )

        return checkpoint(
            cell_forward,
            context,
            direct_image,
            use_reentrant=False,
            # The 3-sigma gather cache has data-dependent internal lengths.
            # Tiny floating-point differences at the support boundary can
            # change that length without changing any public tensor shape.
            determinism_check="none",
        )

    def _forward_persistent_cell_checkpointed(
        self,
        context,
        direct_image,
        persistent,
        encoder,
        persistent_update,
        persistent_fusion,
        block,
        stage_code,
        sparse_sino,
        radon,
        decoder,
        cache_kwargs,
        sigma_logit_bias,
        gaussian_order=0,
    ):
        def cell_forward(
            context_input,
            direct_image_input,
            persistent_input,
        ):
            context_feature = encoder(context_input)
            if self.propagate_block_feature:
                persistent_output = persistent_input
            else:
                persistent_output = persistent_update(
                    persistent_input, context_feature
                )
            if self.learned_persistent_fusion:
                fused_feature = persistent_fusion(
                    torch.cat(
                        [context_feature, persistent_output],
                        dim=1,
                    )
                )
            else:
                fused_feature = (
                    context_feature + persistent_output
                ) * (2.0**-0.5)
            block_output = block(
                context_feature=fused_feature,
                stage_code=stage_code,
                direct_base_image=direct_image_input,
                sparse_sino=sparse_sino,
                radon=radon,
                renderer=self.renderer,
                decoder=decoder,
                cache_kwargs=cache_kwargs,
                sigma_logit_bias=sigma_logit_bias,
                gaussian_order=gaussian_order,
                return_aux=False,
                return_debug=False,
                return_network_feature=(
                    self.propagate_block_feature
                ),
            )
            if self.propagate_block_feature:
                (
                    image_output,
                    context_output,
                    persistent_output,
                ) = block_output
            else:
                image_output, context_output = block_output
            return (
                image_output,
                context_output,
                persistent_output,
            )

        return checkpoint(
            cell_forward,
            context,
            direct_image,
            persistent,
            use_reentrant=False,
            determinism_check="none",
        )

    def _forward_with_persistent(
        self,
        sparse_sino,
        RADON,
        return_aux=False,
        return_debug=False,
        view_counts=None,
    ):
        if return_debug:
            return_aux = True
        radon = RADON[self.render_size]
        batch = sparse_sino.shape[0]
        if view_counts is None:
            view_counts = sparse_sino.new_full(
                (batch,), float(sparse_sino.shape[-2])
            )
        view_condition = (
            SVCTFactorizedRouter.acquisition_vector(
                view_counts, sparse_sino
            )
            if self.view_conditioned_context_encoder
            else None
        )
        fbp_image = torch.nan_to_num(
            recon(sparse_sino, radon=radon),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if self.zero_initial_image:
            direct_image = torch.zeros_like(fbp_image)
        else:
            direct_image = F.interpolate(
                F.adaptive_avg_pool2d(fbp_image, (16, 16)),
                size=(self.render_size, self.render_size),
                mode="bilinear",
                align_corners=False,
            )
        context = self._initial_context(
            direct_image, sparse_sino, radon
        )
        persistent = self.persistent_encoder(
            direct_image, fbp_image
        )
        if self.physics_memory_flow:
            persistent = self.physics_memory.initialize(persistent)
        image_outputs = []
        aux_outputs = []
        checkpoint_cells = (
            self.gradient_checkpointing
            and self.training
            and not return_aux
            and not self.stage_joint_schedule
            and not self.physics_memory_flow
            and not self.physics_gated_memory
            and not self.svct_factorized_moe
        )
        for stage_idx, resolution in enumerate(
            self.stage_resolutions
        ):
            if stage_idx > 0:
                slot_count = (
                    persistent.shape[1]
                    if self.physics_memory_flow
                    else 1
                )
                flat_persistent = (
                    persistent.flatten(0, 1)
                    if self.physics_memory_flow
                    else persistent
                )
                flat_persistent = F.interpolate(
                    flat_persistent,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                flat_persistent = self.persistent_stage_adapters[
                    stage_idx - 1
                ](flat_persistent)
                persistent = (
                    flat_persistent.reshape(
                        batch,
                        slot_count,
                        self.channels,
                        resolution,
                        resolution,
                    )
                    if self.physics_memory_flow
                    else flat_persistent
                )
            stage_ids = torch.full(
                (batch,),
                stage_idx,
                device=sparse_sino.device,
                dtype=torch.long,
            )
            stage_code = self.stage_embedding(stage_ids)
            decoder = self.image_decoders[str(resolution)]
            cache_kwargs = self.CACHE_CONFIGS[resolution]
            stage_base_image = direct_image
            stage_geometry = None
            stage_amplitudes = None
            stage_cgls_image = stage_base_image
            blocks = (
                self.stage_blocks[stage_idx]
                if self.stage_independent_blocks
                else self.shared_blocks
            )
            persistent_updates = (
                self.stage_persistent_updates[stage_idx]
                if self.stage_independent_blocks
                and self.stage_persistent_updates is not None
                else self.persistent_updates
            )
            physics_gated_updates = (
                self.stage_physics_gated_updates[stage_idx]
                if self.stage_independent_blocks
                and self.stage_physics_gated_updates is not None
                else self.physics_gated_updates
            )
            persistent_fusions = (
                self.stage_persistent_fusions[stage_idx]
                if self.stage_independent_blocks
                and self.stage_persistent_fusions is not None
                else self.persistent_fusions
            )
            svct_routers = (
                self.stage_svct_routers[stage_idx]
                if self.stage_independent_blocks
                and self.stage_svct_routers is not None
                else self.svct_routers
            )
            for block_idx, block in enumerate(blocks):
                encoder = self.context_encoders[
                    stage_idx
                ][block_idx]
                persistent_update = (
                    None
                    if (
                        self.propagate_block_feature
                        or self.physics_memory_flow
                        or self.physics_gated_memory
                    )
                    else persistent_updates[block_idx]
                )
                persistent_fusion = (
                    persistent_fusions[block_idx]
                    if self.learned_persistent_fusion
                    else None
                )
                if checkpoint_cells:
                    direct_image, context, persistent = (
                        self._forward_persistent_cell_checkpointed(
                            context=context,
                            direct_image=direct_image,
                            persistent=persistent,
                            encoder=encoder,
                            persistent_update=persistent_update,
                            persistent_fusion=persistent_fusion,
                            block=block,
                            stage_code=stage_code,
                            sparse_sino=sparse_sino,
                            radon=radon,
                            decoder=decoder,
                            cache_kwargs=cache_kwargs,
                            sigma_logit_bias=(
                                self.block_sigma_logit_biases[block_idx]
                            ),
                            gaussian_order=(
                                self.gaussian_stage_orders[stage_idx]
                            ),
                        )
                    )
                else:
                    context_feature = encoder(context, view_condition)
                    memory_diagnostics = None
                    if self.physics_memory_flow:
                        (
                            persistent,
                            persistent_read,
                            memory_diagnostics,
                        ) = self.physics_memory(
                            persistent,
                            context_feature,
                            context,
                            block_idx,
                        )
                    elif self.physics_gated_memory:
                        physics_feature = self.physics_residual_encoder(
                            context,
                            context_feature.shape[-2:],
                        )
                        (
                            persistent,
                            memory_diagnostics,
                        ) = physics_gated_updates[block_idx](
                            persistent,
                            context_feature,
                            physics_feature,
                            stage_idx,
                        )
                        persistent_read = persistent
                    elif not self.propagate_block_feature:
                        persistent = persistent_update(
                            persistent, context_feature
                        )
                        persistent_read = persistent
                    else:
                        persistent_read = persistent
                    if self.learned_persistent_fusion:
                        fused_feature = persistent_fusion(
                            torch.cat(
                                [context_feature, persistent_read],
                                dim=1,
                            )
                        )
                    else:
                        fused_feature = (
                            context_feature + persistent_read
                        ) * (2.0**-0.5)
                    routing = None
                    routing_diagnostics = None
                    if self.svct_factorized_moe:
                        (
                            routing,
                            routing_diagnostics,
                        ) = svct_routers[block_idx](
                            stage_code,
                            view_counts,
                            context,
                            fused_feature,
                        )
                    is_final_cell = (
                        stage_idx == len(self.stage_resolutions) - 1
                        and block_idx == len(blocks) - 1
                    )
                    block_kwargs = {
                        "run_cgls": (
                            self.run_final_block_cgls
                            or not is_final_cell
                        )
                    }
                    need_stage_state = (
                        self.stage_joint_schedule
                        or self.use_coefficient_memory
                    )
                    if self.stage_joint_schedule:
                        block_kwargs.update({
                            "cgls_base_image": stage_cgls_image,
                            "stage_base_image": stage_base_image,
                            "prior_geometry": stage_geometry,
                            "prior_amplitudes": stage_amplitudes,
                            "own_iterations": self.stage_own_iterations,
                            "joint_refine_iterations": (
                                self.stage_joint_refine_iterations
                            ),
                            "return_stage_state": True,
                        })
                    elif self.use_coefficient_memory:
                        block_kwargs["return_stage_state"] = True
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
                        sigma_logit_bias=(
                            self.block_sigma_logit_biases[block_idx]
                        ),
                        return_aux=return_aux,
                        return_debug=return_debug,
                        return_network_feature=(
                            self.propagate_block_feature
                        ),
                        gaussian_order=(
                            self.gaussian_stage_orders[stage_idx]
                        ),
                        **block_kwargs,
                    )
                    if need_stage_state:
                        if return_aux:
                            if self.propagate_block_feature:
                                (
                                    direct_image,
                                    context,
                                    stage_state,
                                    aux,
                                    persistent,
                                ) = result
                            else:
                                (
                                    direct_image,
                                    context,
                                    stage_state,
                                    aux,
                                ) = result
                        else:
                            if self.propagate_block_feature:
                                (
                                    direct_image,
                                    context,
                                    stage_state,
                                    persistent,
                                ) = result
                            else:
                                (
                                    direct_image,
                                    context,
                                    stage_state,
                                ) = result
                        if self.stage_joint_schedule:
                            stage_geometry = stage_state["geometry"]
                            stage_amplitudes = stage_state["amplitudes"]
                            stage_cgls_image = stage_state["image"]
                            stage_base_image = stage_state[
                                "fixed_base_image"
                            ]
                        coefficient_trajectory = stage_state.get(
                            "coefficient_trajectory"
                        )
                        if (
                            self.use_coefficient_memory
                            and coefficient_trajectory is not None
                        ):
                            # The solver trajectory updates persistent memory
                            # after this cell.  The next cell keeps the existing
                            # image-context encoder/injection order unchanged.
                            persistent = self.krylov_coefficient_gru(
                                coefficient_trajectory,
                                persistent_read,
                                stage_idx,
                            )
                    elif return_aux:
                        if self.propagate_block_feature:
                            (
                                direct_image,
                                context,
                                aux,
                                persistent,
                            ) = result
                        else:
                            direct_image, context, aux = result
                    else:
                        if self.propagate_block_feature:
                            (
                                direct_image,
                                context,
                                persistent,
                            ) = result
                        else:
                            direct_image, context = result
                    if return_aux:
                        if routing_diagnostics is not None:
                            aux.update(
                                {
                                    key: value.detach()
                                    for key, value
                                    in routing_diagnostics.items()
                                }
                            )
                        if return_debug:
                            aux["persistent_feature"] = (
                                persistent.detach()
                            )
                            if memory_diagnostics is not None:
                                aux.update(
                                    {
                                        key: value.detach()
                                        for key, value
                                        in memory_diagnostics.items()
                                    }
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
            return image_outputs, aux_outputs
        return image_outputs

    def forward(
        self,
        sparse_sino,
        RADON,
        return_aux=False,
        return_debug=False,
        view_counts=None,
    ):
        if self.use_persistent_feature:
            return self._forward_with_persistent(
                sparse_sino,
                RADON,
                return_aux=return_aux,
                return_debug=return_debug,
                view_counts=view_counts,
            )
        if (
            not self.gradient_checkpointing
            or not self.training
            or return_aux
            or return_debug
        ):
            return super().forward(
                sparse_sino,
                RADON,
                return_aux=return_aux,
                return_debug=return_debug,
            )

        radon = RADON[self.render_size]
        batch = sparse_sino.shape[0]
        fbp_image = torch.nan_to_num(
            recon(sparse_sino, radon=radon),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if self.zero_initial_image:
            direct_image = torch.zeros_like(fbp_image)
        else:
            direct_image = F.interpolate(
                F.adaptive_avg_pool2d(fbp_image, (16, 16)),
                size=(self.render_size, self.render_size),
                mode="bilinear",
                align_corners=False,
            )
        context = self._initial_context(
            direct_image, sparse_sino, radon
        )
        image_outputs = []
        for stage_idx, resolution in enumerate(
            self.stage_resolutions
        ):
            stage_ids = torch.full(
                (batch,),
                stage_idx,
                device=sparse_sino.device,
                dtype=torch.long,
            )
            stage_code = self.stage_embedding(stage_ids)
            decoder = self.image_decoders[str(resolution)]
            cache_kwargs = self.CACHE_CONFIGS[resolution]
            blocks = (
                self.stage_blocks[stage_idx]
                if self.stage_independent_blocks
                else self.shared_blocks
            )
            for block_idx, block in enumerate(blocks):
                encoder = self.context_encoders[
                    stage_idx
                ][block_idx]
                direct_image, context = (
                    self._forward_cell_checkpointed(
                        context=context,
                        direct_image=direct_image,
                        encoder=encoder,
                        block=block,
                        stage_code=stage_code,
                        sparse_sino=sparse_sino,
                        radon=radon,
                        decoder=decoder,
                        cache_kwargs=cache_kwargs,
                        gaussian_order=(
                            self.gaussian_stage_orders[stage_idx]
                        ),
                    )
                )
                image_outputs.append(direct_image)
        return image_outputs


GCT = GCTCGLSTrajectoryContextUnrolled
