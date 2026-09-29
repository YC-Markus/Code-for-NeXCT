"""Image encoders, Gaussian block initialization and acquisition-conditioned routing."""

import torch
import torch.nn as nn
from herkry.core.gaussian import (
    AdaptiveDynamicGaussianRenderer,
    CondConv2d,
    ConditionalNAFBlock,
    FeatureGaussianDecoder,
    ImageFeatureEncoder,
)
from herkry.core.utils import recon


class SVCTFactorizedRouter(nn.Module):
    """One block-level expert decision from acquisition and physics state."""

    def __init__(self, stage_embed_dim, channels, num_experts, hidden=32, view_anchor_logits=False):
        super().__init__()
        self.num_experts = int(num_experts)
        self.use_view_anchor_logits = bool(view_anchor_logits)
        self.stage_path = nn.Linear(stage_embed_dim, self.num_experts, bias=False)
        self.acquisition_path = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, self.num_experts)
        )
        self.trajectory_path = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, self.num_experts)
        )
        self.content_path = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, self.num_experts)
        )
        self.view_anchor_logits = (
            nn.Parameter(torch.zeros(3, self.num_experts)) if self.use_view_anchor_logits else None
        )
        nn.init.zeros_(self.stage_path.weight)
        for path in (self.acquisition_path, self.trajectory_path, self.content_path):
            nn.init.zeros_(path[-1].weight)
            nn.init.zeros_(path[-1].bias)
        if self.use_view_anchor_logits:
            self.acquisition_path.requires_grad_(False)

    @staticmethod
    def trajectory_vector(raw_context):
        if raw_context.shape[1] < 7:
            raise ValueError("SVCT routing requires direct plus five trajectory residuals")
        residual_maps = torch.cat([raw_context[:, 1:2], raw_context[:, -5:]], dim=1)
        energies = residual_maps.abs().mean(dim=(2, 3))
        relative_drop = (energies[:, :1] - energies[:, -1:]) / energies[:, :1].clamp_min(1e-06)
        return torch.cat([torch.log1p(10.0 * energies), relative_drop], dim=1)

    @staticmethod
    def acquisition_vector(view_counts, reference):
        views = torch.as_tensor(
            view_counts, device=reference.device, dtype=reference.dtype
        ).reshape(-1)
        if views.numel() == 1 and reference.shape[0] > 1:
            views = views.expand(reference.shape[0])
        if views.numel() != reference.shape[0]:
            raise ValueError("view_counts must be scalar or contain one value per sample")
        view_fraction = (views / 64.0).clamp(0.0, 1.0)
        sparse_level = (torch.log2(64.0 / views.clamp_min(1.0)) / 2.0).clamp(0.0, 1.0)
        return torch.stack([view_fraction, sparse_level], dim=1)

    def interpolate_view_anchor_logits(self, view_counts, reference):
        if self.view_anchor_logits is None:
            raise RuntimeError("view-anchor logits are not enabled")
        views = torch.as_tensor(
            view_counts, device=reference.device, dtype=reference.dtype
        ).reshape(-1)
        if views.numel() == 1 and reference.shape[0] > 1:
            views = views.expand(reference.shape[0])
        if views.numel() != reference.shape[0]:
            raise ValueError("view_counts must be scalar or contain one value per sample")
        log_views = torch.log2(views.clamp(16.0, 64.0))
        lower = self.view_anchor_logits[0] + (log_views - 4.0)[:, None] * (
            self.view_anchor_logits[1] - self.view_anchor_logits[0]
        )
        upper = self.view_anchor_logits[1] + (log_views - 5.0)[:, None] * (
            self.view_anchor_logits[2] - self.view_anchor_logits[1]
        )
        return torch.where((log_views <= 5.0)[:, None], lower, upper)

    def forward(self, stage_code, view_counts, raw_context, content_feature):
        acquisition = self.acquisition_vector(view_counts, stage_code)
        trajectory = self.trajectory_vector(raw_context).to(stage_code)
        content = content_feature.mean(dim=(2, 3))
        acquisition_logits = (
            self.interpolate_view_anchor_logits(view_counts, stage_code)
            if self.use_view_anchor_logits
            else self.acquisition_path(acquisition)
        )
        logits = (
            self.stage_path(stage_code)
            + acquisition_logits
            + self.trajectory_path(trajectory)
            + self.content_path(content)
        )
        routing = torch.softmax(logits, dim=-1)
        diagnostics = {
            "moe_routing": routing,
            "moe_acquisition": acquisition,
            "moe_acquisition_logits": acquisition_logits,
            "moe_trajectory": trajectory,
        }
        return (routing, diagnostics)


class TrajectoryContextGaussianBlock(nn.Module):

    def __init__(
        self,
        channels,
        token_channels,
        render_size,
        stage_embed_dim,
        cond_dim,
        num_experts,
        cgls_iterations,
        context_output_channels=None,
        trajectory_sample_stride=1,
        image_head_channels=0,
        direct_amplitude_channels=0,
        external_routing=False,
        solver_only_scalar=False,
        reuse_solver_channel_in_nonlinear=False,
        parallel_cgls_base=False,
        projection_safe_initialization=False,
        zero_cgls_initial_amplitudes=False,
        cgls_geometry_gradient=False,
        cgls_delta_context="none",
        higher_order_mode_channels=0,
        higher_order_cgls=False,
        mixed_order_head=False,
        fused_derivative_renderer=False,
        independent_basis_head=False,
        basis_mixer_channels=4,
        hermite_brane=False,
        brane_max_degree=0,
        brane_shared_shape=False,
        brane_kernel_cgls=False,
        hermite_shell_context=False,
        hermite_shell_nonlinear=False,
    ):
        super().__init__()
        self.render_size = int(render_size)
        self.token_channels = int(token_channels)
        self.direct_amplitude_channels = int(direct_amplitude_channels)
        self.solver_only_scalar = bool(solver_only_scalar)
        self.reuse_solver_channel_in_nonlinear = bool(reuse_solver_channel_in_nonlinear)
        self.parallel_cgls_base = bool(parallel_cgls_base)
        self.projection_safe_initialization = bool(projection_safe_initialization)
        self.zero_cgls_initial_amplitudes = bool(zero_cgls_initial_amplitudes)
        self.cgls_geometry_gradient = bool(cgls_geometry_gradient)
        self.cgls_delta_context = str(cgls_delta_context)
        self.higher_order_mode_channels = int(higher_order_mode_channels)
        self.higher_order_cgls = bool(higher_order_cgls)
        self.mixed_order_head = bool(mixed_order_head)
        self.fused_derivative_renderer = bool(fused_derivative_renderer)
        self.independent_basis_head = bool(independent_basis_head)
        self.basis_mixer_channels = int(basis_mixer_channels)
        self.hermite_brane = bool(hermite_brane)
        self.brane_max_degree = int(brane_max_degree)
        self.shell_feature_max_degree = self.brane_max_degree
        self.brane_shared_shape = bool(brane_shared_shape)
        self.brane_kernel_cgls = bool(brane_kernel_cgls)
        self.hermite_shell_context = bool(hermite_shell_context)
        self.hermite_shell_nonlinear = bool(hermite_shell_nonlinear)
        if self.brane_kernel_cgls and (not self.brane_shared_shape):
            raise ValueError("Brane kernel CGLS requires a shared internal shape")
        if self.brane_kernel_cgls and self.higher_order_cgls:
            raise ValueError("Kernel CGLS and free higher-order CGLS are mutually exclusive")
        if self.hermite_brane and (not 1 <= self.brane_max_degree <= 3):
            raise ValueError("Hermite Brane max degree must be 1, 2, or 3")
        if self.hermite_brane and self.higher_order_mode_channels > 0:
            raise ValueError("Hermite Brane and legacy derivative modes are mutually exclusive")
        if self.independent_basis_head and self.basis_mixer_channels < 1:
            raise ValueError("Independent basis head needs at least one mixer channel")
        if self.cgls_delta_context not in ("none", "branch_disagreement", "solver_update"):
            raise ValueError("Unknown CGLS delta context mode")
        self.cgls_iterations = int(cgls_iterations)
        self.external_routing = bool(external_routing)
        self.trajectory_sample_stride = int(trajectory_sample_stride)
        if (
            self.trajectory_sample_stride < 1
            or self.cgls_iterations % self.trajectory_sample_stride != 0
        ):
            raise ValueError("CGLS iterations must be divisible by trajectory stride")
        self.context_step_indices = tuple(
            range(
                self.trajectory_sample_stride - 1,
                self.cgls_iterations,
                self.trajectory_sample_stride,
            )
        )
        self.raw_context_channels = 2 + 2 * len(self.context_step_indices)
        if self.cgls_delta_context != "none":
            self.raw_context_channels += len(self.context_step_indices)
        self.base_raw_context_channels = self.raw_context_channels
        self.shell_context_channels = (
            5 * self.shell_feature_max_degree if self.hermite_shell_context else 0
        )
        self.raw_context_channels += self.shell_context_channels
        self.context_output_channels = int(
            self.raw_context_channels
            if context_output_channels is None
            else context_output_channels
        )
        self.condition = (
            None
            if self.external_routing
            else nn.Sequential(
                nn.Linear(stage_embed_dim + channels, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
            )
        )
        self.context_fuse = CondConv2d(
            channels,
            channels,
            3,
            padding=1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=not self.external_routing,
        )
        self.process_feature = ConditionalNAFBlock(
            channels=channels,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=not self.external_routing,
        )
        self.to_raw_spatial_hidden = CondConv2d(
            channels,
            channels,
            3,
            padding=1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=not self.external_routing,
        )
        self.to_raw_spatial_out = CondConv2d(
            channels,
            5,
            3,
            padding=1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=not self.external_routing,
        )
        self.to_raw_feature_hidden = None
        feature_output_channels = self.token_channels
        if self.direct_amplitude_channels > 0:
            if self.direct_amplitude_channels < 2:
                raise ValueError(
                    "direct amplitude mode needs nonlinear channels plus one scalar CGLS channel"
                )
            feature_output_channels = self.direct_amplitude_channels
        else:
            self.to_raw_feature_hidden = CondConv2d(
                channels,
                channels,
                3,
                padding=1,
                cond_dim=cond_dim,
                num_experts=num_experts,
                use_internal_router=not self.external_routing,
            )
        self.to_raw_feature_out = CondConv2d(
            channels,
            feature_output_channels,
            3,
            padding=1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=not self.external_routing,
        )
        self.to_raw_spatial_out.zero_init()
        self.to_raw_feature_out.zero_init()
        self.to_raw_higher_order_out = None
        if self.hermite_brane:
            brane_feature_channels = self.direct_amplitude_channels - 1
            if brane_feature_channels < 1:
                raise ValueError("Hermite Brane requires direct nonlinear feature channels")
            self.brane_feature_channels = brane_feature_channels
            self.brane_high_mode_count = (self.brane_max_degree + 1) ** 2 - 1
            brane_vectors = (
                1
                if self.brane_shared_shape
                else brane_feature_channels + int(self.higher_order_cgls)
            )
            rng_state = torch.get_rng_state()
            self.to_raw_higher_order_out = CondConv2d(
                channels,
                brane_vectors * self.brane_high_mode_count,
                3,
                padding=1,
                cond_dim=cond_dim,
                num_experts=num_experts,
                use_internal_router=not self.external_routing,
            )
            self.to_raw_higher_order_out.zero_init()
            torch.set_rng_state(rng_state)
        elif self.higher_order_mode_channels > 0:
            rng_state = torch.get_rng_state()
            self.to_raw_higher_order_out = CondConv2d(
                channels,
                5 * self.higher_order_mode_channels,
                3,
                padding=1,
                cond_dim=cond_dim,
                num_experts=num_experts,
                use_internal_router=not self.external_routing,
            )
            self.to_raw_higher_order_out.zero_init()
            torch.set_rng_state(rng_state)
        self.linear_feature_to_scalar = None
        if self.direct_amplitude_channels == 0:
            self.linear_feature_to_scalar = nn.Conv2d(token_channels, 1, kernel_size=1, bias=False)
            nn.init.constant_(self.linear_feature_to_scalar.weight, 1.0 / max(1, token_channels))
        self.image_head_channels = (
            self.direct_amplitude_channels - 1
            if self.direct_amplitude_channels > 0
            else int(image_head_channels)
        )
        if self.reuse_solver_channel_in_nonlinear and self.direct_amplitude_channels <= 0:
            raise ValueError("Reusing the solver channel requires direct amplitudes")
        self.nonlinear_head_channels = self.image_head_channels + int(
            self.reuse_solver_channel_in_nonlinear
        )
        self.image_feature_mixer = None
        self.image_residual_head = None
        if self.image_head_channels > 0 and self.direct_amplitude_channels == 0:
            rng_state = torch.get_rng_state()
            self.image_feature_mixer = nn.Conv2d(
                token_channels, self.image_head_channels, kernel_size=1, bias=False
            )
            with torch.no_grad():
                self.image_feature_mixer.weight.zero_()
                for channel in range(min(token_channels, self.image_head_channels)):
                    self.image_feature_mixer.weight[channel, channel, 0, 0] = 1.0
            hidden = max(16, 2 * self.nonlinear_head_channels)
            self.image_residual_head = nn.Sequential(
                nn.Conv2d(self.nonlinear_head_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.image_residual_head[-1].weight)
            nn.init.zeros_(self.image_residual_head[-1].bias)
            torch.set_rng_state(rng_state)
        elif self.image_head_channels > 0:
            rng_state = torch.get_rng_state()
            hidden = max(16, 2 * self.nonlinear_head_channels)
            self.image_residual_head = nn.Sequential(
                nn.Conv2d(self.nonlinear_head_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.image_residual_head[-1].weight)
            nn.init.zeros_(self.image_residual_head[-1].bias)
            torch.set_rng_state(rng_state)
        if self.hermite_shell_nonlinear:
            if not self.hermite_brane or self.image_residual_head is None:
                raise ValueError(
                    "Hermite shell nonlinear features require the Brane nonlinear image branch"
                )
            original = self.image_residual_head[0]
            expanded = nn.Conv2d(
                self.nonlinear_head_channels * (1 + self.shell_feature_max_degree),
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=original.bias is not None,
            )
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, : original.in_channels].copy_(original.weight)
                if original.bias is not None:
                    expanded.bias.copy_(original.bias)
            self.image_residual_head[0] = expanded
        self.basis_response_mixer = None
        if self.higher_order_mode_channels > 0 and self.image_residual_head is not None:
            original = self.image_residual_head[0]
            rng_state = torch.get_rng_state()
            if self.independent_basis_head:
                self.basis_response_mixer = nn.Conv2d(
                    5, self.basis_mixer_channels, kernel_size=1, bias=False
                )
                with torch.no_grad():
                    self.basis_response_mixer.weight.zero_()
                    seeds = (
                        (0.5, 0.5, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                        (0.2, 0.2, 0.2, 0.2, 0.2),
                        (0.0, 0.0, 0.5, 0.0, -0.5),
                    )
                    for output_index in range(self.basis_mixer_channels):
                        seed = seeds[output_index % len(seeds)]
                        self.basis_response_mixer.weight[output_index, :, 0, 0] = (
                            original.weight.new_tensor(seed)
                        )
                extra_channels = 5 + 1 + self.basis_mixer_channels
            else:
                extra_channels = (
                    3 * self.higher_order_mode_channels
                    if self.mixed_order_head
                    else self.higher_order_mode_channels
                )
            expanded = nn.Conv2d(
                original.in_channels + extra_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=original.bias is not None,
            )
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, : original.in_channels].copy_(original.weight)
                if original.bias is not None:
                    expanded.bias.copy_(original.bias)
            self.image_residual_head[0] = expanded
            torch.set_rng_state(rng_state)
        self.trajectory_adapter = None
        if self.context_output_channels != self.raw_context_channels:
            if self.context_output_channels < 4:
                raise ValueError("Compressed trajectory context needs at least 4 channels")
            rng_state = torch.get_rng_state()
            self.trajectory_adapter = nn.Conv2d(
                self.raw_context_channels, self.context_output_channels, kernel_size=1, bias=False
            )
            self._initialize_trajectory_adapter()
            if self.cgls_delta_context != "none":
                torch.set_rng_state(rng_state)

    def _initialize_trajectory_adapter(self):
        """Initialize the 1x1 adapter as uniform temporal bin averaging."""
        if self.cgls_delta_context != "none":
            with torch.no_grad():
                self.trajectory_adapter.weight.zero_()
                copied = min(self.context_output_channels, 2 + 2 * len(self.context_step_indices))
                indices = torch.arange(copied)
                self.trajectory_adapter.weight[indices, indices, 0, 0] = 1.0
            return
        output_steps = (self.context_output_channels - 2) // 2
        sampled_steps = len(self.context_step_indices)
        if sampled_steps % output_steps != 0:
            raise ValueError("Sampled CGLS steps must be divisible by context slots")
        steps_per_slot = sampled_steps // output_steps
        weight = self.trajectory_adapter.weight
        with torch.no_grad():
            weight.zero_()
            weight[0, 0, 0, 0] = 1.0
            weight[1, 1, 0, 0] = 1.0
            for slot in range(output_steps):
                begin = slot * steps_per_slot
                end = begin + steps_per_slot
                scale = 1.0 / steps_per_slot
                weight[2 + slot, 2 + begin : 2 + end, 0, 0] = scale
                residual_input_begin = 2 + sampled_steps + begin
                residual_input_end = 2 + sampled_steps + end
                weight[2 + output_steps + slot, residual_input_begin:residual_input_end, 0, 0] = (
                    scale
                )

    def adapt_trajectory_context(self, raw_context):
        if self.trajectory_adapter is None:
            return raw_context
        return self.trajectory_adapter(raw_context)


class GCTCGLSTrajectoryContext(nn.Module):
    CACHE_CONFIGS = {
        32: {"tile_size": 32, "max_gauss_chunk": 8},
        64: {"tile_size": 32, "max_gauss_chunk": 8},
        128: {"tile_size": 16, "max_gauss_chunk": 16},
        256: {"tile_size": 16, "max_gauss_chunk": 32},
    }

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
        image_head_channels=0,
        direct_amplitude_channels=0,
        max_offset_cells=0.5,
        anchor_offset_cells=0.5,
        stage_sigma_fractions=None,
        cgls_delta_context="none",
    ):
        super().__init__()
        self.render_size = int(render_size)
        self.stage_resolutions = tuple(stage_resolutions)
        self.channels = int(channels)
        self.blocks_per_stage = int(blocks_per_stage)
        self.token_channels = int(token_channels)
        self.cgls_iterations = int(cgls_iterations)
        self.trajectory_sample_stride = int(trajectory_sample_stride)
        self.image_head_channels = int(image_head_channels)
        self.direct_amplitude_channels = int(direct_amplitude_channels)
        self.max_offset_cells = float(max_offset_cells)
        self.anchor_offset_cells = float(anchor_offset_cells)
        self.stage_sigma_fractions = (
            None
            if stage_sigma_fractions is None
            else tuple((float(value) for value in stage_sigma_fractions))
        )
        self.cgls_delta_context = str(cgls_delta_context)
        if self.cgls_delta_context not in ("none", "branch_disagreement", "solver_update"):
            raise ValueError("Unknown CGLS delta context mode")
        if self.stage_sigma_fractions is not None and len(self.stage_sigma_fractions) != len(
            self.stage_resolutions
        ):
            raise ValueError("stage sigma fractions must match stage resolutions")
        if self.direct_amplitude_channels > 0:
            self.image_head_channels = self.direct_amplitude_channels - 1
        self.gaussian_feature_channels = (
            self.direct_amplitude_channels
            if self.direct_amplitude_channels > 0
            else self.token_channels
        )
        if (
            self.trajectory_sample_stride < 1
            or self.cgls_iterations % self.trajectory_sample_stride != 0
        ):
            raise ValueError("CGLS iterations must be divisible by trajectory stride")
        self.context_steps = tuple(
            range(
                self.trajectory_sample_stride,
                self.cgls_iterations + 1,
                self.trajectory_sample_stride,
            )
        )
        self.raw_context_channels = 2 + 2 * len(self.context_steps)
        if self.cgls_delta_context != "none":
            self.raw_context_channels += len(self.context_steps)
        self.context_channels = int(
            self.raw_context_channels if context_channels is None else context_channels
        )
        if self.cgls_delta_context == "none" and (
            self.context_channels < 4 or (self.context_channels - 2) % 2 != 0
        ):
            raise ValueError("Network context channels must be 2 + 2*k")
        self.context_history_slots = (
            len(self.context_steps)
            if self.cgls_delta_context != "none"
            else (self.context_channels - 2) // 2
        )
        if (
            self.cgls_delta_context == "none"
            and len(self.context_steps) % self.context_history_slots != 0
        ):
            raise ValueError("Sampled steps must be divisible by context history slots")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.output_resolutions = tuple(
            (
                resolution
                for resolution in self.stage_resolutions
                for _ in range(self.blocks_per_stage)
            )
        )
        self.renderer = AdaptiveDynamicGaussianRenderer(
            H=render_size, W=render_size, train_tile_size=4, train_chunk=32, min_sigma_px=0.1
        )
        self.stage_embedding = nn.Embedding(len(self.stage_resolutions), stage_embed_dim)
        self.context_encoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ImageFeatureEncoder(
                            in_channels=self.context_channels,
                            out_channels=self.channels,
                            resolution=resolution,
                            render_size=render_size,
                            use_naf=False,
                        )
                        for _ in range(self.blocks_per_stage)
                    ]
                )
                for resolution in self.stage_resolutions
            ]
        )
        self.shared_blocks = nn.ModuleList(
            [
                TrajectoryContextGaussianBlock(
                    channels=self.channels,
                    token_channels=self.token_channels,
                    render_size=self.render_size,
                    stage_embed_dim=stage_embed_dim,
                    cond_dim=cond_dim,
                    num_experts=num_experts,
                    cgls_iterations=self.cgls_iterations,
                    context_output_channels=self.context_channels,
                    trajectory_sample_stride=self.trajectory_sample_stride,
                    image_head_channels=self.image_head_channels,
                    direct_amplitude_channels=self.direct_amplitude_channels,
                    cgls_delta_context=self.cgls_delta_context,
                )
                for _ in range(self.blocks_per_stage)
            ]
        )
        self.image_decoders = nn.ModuleDict(
            {
                str(resolution): FeatureGaussianDecoder(
                    resolution=resolution,
                    render_size=render_size,
                    token_channels=self.gaussian_feature_channels,
                    residual_scale=(
                        1.0 / self.direct_amplitude_channels
                        if self.direct_amplitude_channels > 0
                        else 1.0
                    ),
                    max_offset_cells=self.max_offset_cells,
                    anchor_offset_cells=self.anchor_offset_cells,
                    sigma_reference_cell_fraction=(
                        None
                        if self.stage_sigma_fractions is None
                        else self.stage_sigma_fractions[stage_idx]
                    ),
                    feature_activation="identity" if self.direct_amplitude_channels > 0 else "tanh",
                )
                for (stage_idx, resolution) in enumerate(self.stage_resolutions)
            }
        )

    def _initial_context(self, initial_image, sparse_sino, radon):
        initial_sino = radon.forward(initial_image.contiguous())
        residual_fbp = recon((sparse_sino - initial_sino).contiguous(), radon=radon)
        context_parts = [
            initial_image,
            residual_fbp,
            initial_image.repeat(1, self.context_history_slots, 1, 1),
            residual_fbp.repeat(1, self.context_history_slots, 1, 1),
        ]
        if self.cgls_delta_context != "none":
            context_parts.append(
                torch.zeros_like(initial_image).repeat(1, len(self.context_steps), 1, 1)
            )
        return torch.cat(context_parts, dim=1)


GCT = GCTCGLSTrajectoryContext
