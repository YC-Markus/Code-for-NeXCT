"""Krylov coefficient memory and role-shared SolverDNA convolutions."""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from herkry.core.gaussian import CondConv2d


class CoefficientTrajectoryEncoder(nn.Module):
    """Encode coefficients and selected-step increments without normalization."""

    def __init__(self, mode_count, width=16):
        super().__init__()
        self.mode_count = int(mode_count)
        self.net = nn.Sequential(
            nn.Conv2d(2 * self.mode_count, width, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=1),
            nn.GELU(),
        )

    def forward(self, coefficients, increments):
        return self.net(torch.cat([coefficients, increments], dim=1))


class KrylovCoefficientGRUCell(nn.Module):
    """A spatially pointwise ConvGRU with a 48-channel persistent state."""

    def __init__(self, input_channels=16, hidden_channels=48):
        super().__init__()
        fused_channels = int(input_channels) + int(hidden_channels)
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(
            fused_channels,
            2 * self.hidden_channels,
            1,
            groups=1,
        )
        self.candidate = nn.Conv2d(
            fused_channels,
            self.hidden_channels,
            1,
            groups=1,
        )
        # Start close to the baseline persistent path.  The reset gate remains
        # neutral while the update gate initially writes almost nothing.
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        nn.init.constant_(
            self.gates.bias[: self.hidden_channels], -6.0
        )
        nn.init.zeros_(self.candidate.bias)

    def forward(self, encoded_coefficients, hidden):
        fused = torch.cat([encoded_coefficients, hidden], dim=1)
        update, reset = torch.sigmoid(self.gates(fused)).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(
                torch.cat(
                    [encoded_coefficients, reset * hidden], dim=1
                )
            )
        )
        return hidden + update * (candidate - hidden)


class KrylovCoefficientGRU(nn.Module):
    """Read five selected CGLS coefficient states into persistent memory."""

    def __init__(self, stage_orders, hidden_channels=48, width=16):
        super().__init__()
        self.stage_orders = tuple(int(order) for order in stage_orders)
        self.encoders = nn.ModuleList(
            [
                CoefficientTrajectoryEncoder(
                    (order + 1) ** 2, width=width
                )
                for order in self.stage_orders
            ]
        )
        self.cell = KrylovCoefficientGRUCell(
            input_channels=width,
            hidden_channels=hidden_channels,
        )

    def forward(self, selected_amplitudes, hidden, stage_idx):
        """
        Args:
            selected_amplitudes: [B, 6, R*R, M], steps 0/2/4/6/8/10.
            hidden: [B, 48, R, R] persistent feature after image injection.
        """
        if selected_amplitudes.ndim != 4:
            raise ValueError(
                "coefficient trajectory must have shape [B,T,N,M]"
            )
        if selected_amplitudes.shape[1] != 6:
            raise ValueError(
                "coefficient GRU requires steps 0/2/4/6/8/10"
            )
        batch, _, gaussian_count, mode_count = selected_amplitudes.shape
        height, width = hidden.shape[-2:]
        if gaussian_count != height * width:
            raise ValueError(
                "Gaussian lattice does not match persistent resolution"
            )
        expected_modes = (self.stage_orders[int(stage_idx)] + 1) ** 2
        if mode_count != expected_modes:
            raise ValueError(
                f"stage {stage_idx} expects {expected_modes} modes, "
                f"got {mode_count}"
            )
        encoder = self.encoders[int(stage_idx)]
        previous = selected_amplitudes[:, 0]
        for current in selected_amplitudes[:, 1:].unbind(dim=1):
            increment = current - previous
            coefficient_map = current.transpose(1, 2).reshape(
                batch, mode_count, height, width
            )
            increment_map = increment.transpose(1, 2).reshape(
                batch, mode_count, height, width
            )
            hidden = self.cell(
                encoder(coefficient_map, increment_map), hidden
            )
            previous = current
        return hidden


class KrylovDeltaEncoder(nn.Module):
    """Stage-specific solution/dynamics encoding without normalization."""

    def __init__(self, mode_count, width=16):
        super().__init__()
        self.mode_count = int(mode_count)
        self.solution = nn.Sequential(
            nn.Conv2d(self.mode_count, width, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=1),
            nn.GELU(),
        )
        self.dynamics = nn.Sequential(
            nn.Conv2d(2 * self.mode_count, width, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=1),
            nn.GELU(),
        )
        self.convergence = nn.Conv2d(
            2 * self.mode_count, width, 1, groups=1
        )

    def forward(self, coefficients, increment, curvature):
        solution = self.solution(coefficients)
        dynamics = self.dynamics(
            torch.cat([increment, curvature], dim=1)
        )
        interaction = solution + dynamics + solution * torch.sigmoid(
            dynamics
        )
        convergence = torch.sigmoid(
            self.convergence(
                torch.cat(
                    [
                        torch.log1p(increment.abs()),
                        torch.log1p(curvature.abs()),
                    ],
                    dim=1,
                )
            )
        )
        return interaction, convergence


class KrylovDeltaGRUCell(nn.Module):
    """A convergence-aware compact GRU shared by all reconstruction stages."""

    def __init__(self, width=16):
        super().__init__()
        self.width = int(width)
        self.reset_gate = nn.Conv2d(2 * self.width, self.width, 1)
        self.write_gate = nn.Conv2d(3 * self.width, self.width, 1)
        self.candidate = nn.Conv2d(2 * self.width, self.width, 1)
        nn.init.zeros_(self.reset_gate.bias)
        nn.init.constant_(self.write_gate.bias, -3.0)
        nn.init.zeros_(self.candidate.bias)

    def forward(self, trajectory, convergence, state):
        reset = torch.sigmoid(
            self.reset_gate(torch.cat([state, trajectory], dim=1))
        )
        write = torch.sigmoid(
            self.write_gate(
                torch.cat([state, trajectory, convergence], dim=1)
            )
        )
        candidate = torch.tanh(
            self.candidate(
                torch.cat([trajectory, reset * state], dim=1)
            )
        )
        return state + write * (candidate - state)


class KrylovDeltaGRU(nn.Module):
    """Compress a variable-length Krylov coefficient trajectory into memory."""

    def __init__(self, stage_orders, persistent_channels=48, width=16):
        super().__init__()
        self.stage_orders = tuple(int(order) for order in stage_orders)
        self.persistent_channels = int(persistent_channels)
        self.width = int(width)
        self.encoders = nn.ModuleList(
            [
                KrylovDeltaEncoder((order + 1) ** 2, width=self.width)
                for order in self.stage_orders
            ]
        )
        self.state_down = nn.Conv2d(
            self.persistent_channels, self.width, 1, groups=1
        )
        self.cell = KrylovDeltaGRUCell(width=self.width)
        self.state_up = nn.Conv2d(
            self.width, self.persistent_channels, 1, groups=1
        )
        nn.init.normal_(self.state_up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.state_up.bias)

    @staticmethod
    def _to_map(values, batch, modes, height, width):
        return values.transpose(1, 2).reshape(
            batch, modes, height, width
        )

    def forward(self, selected_amplitudes, hidden, stage_idx):
        if selected_amplitudes.ndim != 4:
            raise ValueError(
                "KDelta trajectory must have shape [B,T,N,M]"
            )
        if selected_amplitudes.shape[1] < 2:
            raise ValueError(
                "KDelta GRU requires an initial state and at least one update"
            )
        batch, _, gaussian_count, mode_count = selected_amplitudes.shape
        height, width = hidden.shape[-2:]
        if gaussian_count != height * width:
            raise ValueError(
                "Gaussian lattice does not match persistent resolution"
            )
        expected_modes = (self.stage_orders[int(stage_idx)] + 1) ** 2
        if mode_count != expected_modes:
            raise ValueError(
                f"stage {stage_idx} expects {expected_modes} modes, "
                f"got {mode_count}"
            )
        encoder = self.encoders[int(stage_idx)]
        initial_state = self.state_down(hidden)
        state = initial_state
        previous = selected_amplitudes[:, 0]
        previous_increment = torch.zeros_like(previous)
        for current in selected_amplitudes[:, 1:].unbind(dim=1):
            increment = current - previous
            curvature = increment - previous_increment
            coefficient_map = self._to_map(
                current, batch, mode_count, height, width
            )
            increment_map = self._to_map(
                increment, batch, mode_count, height, width
            )
            curvature_map = self._to_map(
                curvature, batch, mode_count, height, width
            )
            trajectory, convergence = encoder(
                coefficient_map, increment_map, curvature_map
            )
            state = self.cell(trajectory, convergence, state)
            previous = current
            previous_increment = increment
        return hidden + self.state_up(state - initial_state)


class SolverDNAFactorizedRouter(nn.Module):
    """One compact stage/view/content code, computed once per block."""

    def __init__(
        self,
        stage_embed_dim,
        channels,
        code_dim=16,
        hidden=32,
        block_idx=0,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.stage_path = nn.Linear(
            stage_embed_dim, self.code_dim, bias=False
        )
        self.acquisition_path = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.code_dim),
        )
        self.content_path = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.code_dim),
        )
        self.block_code = nn.Parameter(torch.zeros(self.code_dim))
        with torch.no_grad():
            self.block_code[int(block_idx) % self.code_dim] = 0.01
        nn.init.zeros_(self.stage_path.weight)
        nn.init.zeros_(self.acquisition_path[-1].weight)
        nn.init.zeros_(self.acquisition_path[-1].bias)
        nn.init.zeros_(self.content_path[-1].weight)
        nn.init.zeros_(self.content_path[-1].bias)

    @staticmethod
    def acquisition_vector(view_counts, reference):
        views = torch.as_tensor(
            view_counts, device=reference.device, dtype=reference.dtype
        ).reshape(-1)
        if views.numel() == 1 and reference.shape[0] > 1:
            views = views.expand(reference.shape[0])
        if views.numel() != reference.shape[0]:
            raise ValueError(
                "view_counts must be scalar or contain one value per sample"
            )
        fraction = (views / 64.0).clamp_min(1.0 / 64.0)
        return torch.stack([fraction, torch.log(fraction)], dim=1)

    def forward(
        self,
        stage_code,
        view_counts,
        raw_context,
        content_feature,
    ):
        del raw_context
        acquisition = self.acquisition_vector(view_counts, stage_code)
        content = content_feature.mean(dim=(2, 3))
        code = torch.tanh(
            self.stage_path(stage_code)
            + self.acquisition_path(acquisition)
            + self.content_path(content)
            + self.block_code[None]
        )
        diagnostics = {
            "solverdna_code": code,
            "solverdna_acquisition": acquisition,
        }
        return code, diagnostics


class SolverDNAConv2d(nn.Module):
    """Role-shared parent kernel with foldable child and low-rank adapters."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        router_dim=16,
        rank=16,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kh), int(kw))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        self.router_dim = int(router_dim)
        flat_input = (
            self.in_channels // self.groups
        ) * self.kernel_size[0] * self.kernel_size[1]
        effective_rank = max(
            1, min(int(rank), self.out_channels, flat_input)
        )
        self.parent_weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels // self.groups,
                *self.kernel_size,
            )
        )
        self.parent_bias = (
            nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        )
        self.spatial_scale = nn.Parameter(
            torch.zeros(1, 1, *self.kernel_size)
        )
        self.filter_scale = nn.Parameter(
            torch.zeros(self.out_channels, 1, 1, 1)
        )
        self.bias_delta = (
            nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        )
        self.low_rank_out = nn.Parameter(
            torch.zeros(self.out_channels, effective_rank)
        )
        self.low_rank_in = nn.Parameter(
            torch.empty(effective_rank, flat_input)
        )
        nn.init.kaiming_uniform_(self.parent_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.low_rank_in, a=math.sqrt(5))
        self.gate_projection = nn.Linear(
            self.router_dim, self.in_channels
        )
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)
        self._cached_weight = None
        self._cached_bias = None

    def train(self, mode=True):
        self._cached_weight = None
        self._cached_bias = None
        return super().train(mode)

    def _effective_parameters(self):
        weight = self.parent_weight
        weight = weight * (
            1.0 + 0.1 * torch.tanh(self.spatial_scale)
        )
        weight = weight * (
            1.0 + 0.1 * torch.tanh(self.filter_scale)
        )
        correction = torch.matmul(
            self.low_rank_out, self.low_rank_in
        ).reshape_as(weight)
        weight = weight + correction
        bias = None
        if self.parent_bias is not None:
            bias = self.parent_bias + self.bias_delta
        return weight, bias

    def _inference_parameters(self):
        if self._cached_weight is None:
            weight, bias = self._effective_parameters()
            self._cached_weight = weight.detach()
            self._cached_bias = None if bias is None else bias.detach()
        return self._cached_weight, self._cached_bias

    def forward(self, x, cond=None, routing=None):
        del cond
        if routing is None:
            raise ValueError("SolverDNAConv2d requires a block routing code")
        if routing.shape != (x.shape[0], self.router_dim):
            raise ValueError(
                "SolverDNA routing must have shape "
                f"({x.shape[0]}, {self.router_dim})"
            )
        gate = 1.0 + 0.1 * torch.tanh(
            self.gate_projection(routing)
        )
        x = x * gate[:, :, None, None].to(dtype=x.dtype)
        if not self.training and not torch.is_grad_enabled():
            weight, bias = self._inference_parameters()
        else:
            weight, bias = self._effective_parameters()
        return F.conv2d(
            x,
            weight.to(dtype=x.dtype),
            None if bias is None else bias.to(dtype=x.dtype),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def _expanded_group_weight(weight, old_groups, new_groups):
    if old_groups == new_groups:
        return weight
    if old_groups != 4 or new_groups != 2:
        raise ValueError(
            f"unsupported group conversion {old_groups} -> {new_groups}"
        )
    out_channels, old_input, kh, kw = weight.shape
    old_output = out_channels // old_groups
    new_input = old_input * (old_groups // new_groups)
    expanded = weight.new_zeros(out_channels, new_input, kh, kw)
    for old_group in range(old_groups):
        output_begin = old_group * old_output
        output_end = output_begin + old_output
        input_offset = (old_group % (old_groups // new_groups)) * old_input
        expanded[
            output_begin:output_end,
            input_offset : input_offset + old_input,
        ] = weight[output_begin:output_end]
    return expanded


def _role_key(path, stage_idx):
    relative = re.sub(r"^stage_blocks\.\d+\.\d+\.", "", path)
    output_head = relative in {
        "to_raw_spatial_out",
        "to_raw_feature_out",
        "to_raw_higher_order_out",
    }
    sharing_scope = (
        f"stage{stage_idx}"
        if output_head
        else ("coarse" if stage_idx < 2 else "fine")
    )
    return relative, sharing_scope


def replace_condconv_with_solverdna(
    model,
    rank=16,
    router_dim=16,
    maximum_groups=2,
):
    """Replace every stage-block CondConv and share parent parameters by role."""
    if model.stage_blocks is None:
        raise ValueError("SolverDNA requires stage-independent blocks")
    entries = []
    for stage_idx, stage in enumerate(model.stage_blocks):
        for block_idx, block in enumerate(stage):
            stack = [(f"stage_blocks.{stage_idx}.{block_idx}", block)]
            while stack:
                prefix, module = stack.pop()
                for child_name, child in list(module.named_children()):
                    path = f"{prefix}.{child_name}"
                    if isinstance(child, CondConv2d):
                        entries.append(
                            (stage_idx, path, module, child_name, child)
                        )
                    else:
                        stack.append((path, child))

    grouped = {}
    replacements = []
    for stage_idx, path, parent_module, child_name, old in entries:
        groups = min(old.groups, int(maximum_groups))
        replacement = SolverDNAConv2d(
            old.in_channels,
            old.out_channels,
            old.kernel_size,
            router_dim=router_dim,
            rank=rank,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=groups,
            bias=old.expert_biases is not None,
        )
        mean_weight = torch.stack(
            [item.detach() for item in old.expert_weights], dim=0
        ).mean(dim=0)
        mean_weight = _expanded_group_weight(
            mean_weight, old.groups, groups
        )
        mean_bias = None
        if old.expert_biases is not None:
            mean_bias = torch.stack(
                [item.detach() for item in old.expert_biases], dim=0
            ).mean(dim=0)
        key = (*_role_key(path, stage_idx), tuple(mean_weight.shape))
        grouped.setdefault(key, []).append(
            (replacement, mean_weight, mean_bias)
        )
        replacements.append((parent_module, child_name, replacement))

    for values in grouped.values():
        shared_weight = nn.Parameter(
            torch.stack([item[1] for item in values], dim=0).mean(dim=0)
        )
        shared_bias = None
        if values[0][2] is not None:
            shared_bias = nn.Parameter(
                torch.stack([item[2] for item in values], dim=0).mean(dim=0)
            )
        for module, _, _ in values:
            module.parent_weight = shared_weight
            if shared_bias is not None:
                module.parent_bias = shared_bias

    for parent_module, child_name, replacement in replacements:
        setattr(parent_module, child_name, replacement)
    return len(replacements), len(grouped)


class KernelSubspaceCondConv2d(nn.Module):
    """DyConv whose experts inhabit a shared stage-local kernel subspace."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_experts=6,
        num_bases=4,
        residual_rank=4,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kh), int(kw))
        self.num_experts = int(num_experts)
        self.num_bases = int(num_bases)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        flat_input = (
            self.in_channels // self.groups
        ) * self.kernel_size[0] * self.kernel_size[1]
        self.residual_rank = max(
            1,
            min(
                int(residual_rank), self.out_channels, flat_input
            ),
        )
        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        self.basis_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(weight_shape)) for _ in range(self.num_bases)]
        )
        for weight in self.basis_weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        coefficients = torch.randn(self.num_experts, self.num_bases)
        coefficients = F.normalize(coefficients, p=2, dim=1)
        self.expert_coefficients = nn.Parameter(coefficients)
        self.residual_out = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(self.out_channels, self.residual_rank)
                )
                for _ in range(self.num_experts)
            ]
        )
        self.residual_in = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.residual_rank, flat_input)
                )
                for _ in range(self.num_experts)
            ]
        )
        for item in self.residual_in:
            nn.init.kaiming_uniform_(item, a=math.sqrt(5))
        self.expert_bias = (
            nn.Parameter(torch.zeros(self.num_experts, self.out_channels))
            if bias
            else None
        )

    def forward(self, x, cond=None, routing=None):
        del cond
        batch, _, height, width = x.shape
        if routing is None:
            raise ValueError(
                "KernelSubspaceCondConv2d requires external routing"
            )
        if routing.shape != (batch, self.num_experts):
            raise ValueError(
                "routing must have shape "
                f"({batch}, {self.num_experts}), got {tuple(routing.shape)}"
            )
        routing = routing.to(device=x.device, dtype=x.dtype)
        basis = torch.stack(list(self.basis_weights), dim=0).to(
            dtype=x.dtype
        )
        sample_coefficients = torch.matmul(
            routing, self.expert_coefficients.to(dtype=x.dtype)
        )
        weight = torch.einsum(
            "bq,qocij->bocij", sample_coefficients, basis
        )
        expert_residuals = torch.stack(
            [
                torch.matmul(out, inp)
                for out, inp in zip(
                    self.residual_out, self.residual_in
                )
            ],
            dim=0,
        ).to(dtype=x.dtype)
        residual = torch.einsum(
            "be,eof->bof", routing, expert_residuals
        ).reshape_as(weight)
        weight = (weight + residual).reshape(
            batch * self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        bias = None
        if self.expert_bias is not None:
            bias = torch.matmul(
                routing, self.expert_bias.to(dtype=x.dtype)
            ).reshape(-1)
        grouped_input = x.contiguous().reshape(
            1, batch * self.in_channels, height, width
        )
        output = F.conv2d(
            grouped_input,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch * self.groups,
        )
        return output.reshape(
            batch,
            self.out_channels,
            output.shape[-2],
            output.shape[-1],
        )


def _stage_local_role(path, stage_idx):
    relative = re.sub(r"^stage_blocks\.\d+\.\d+\.", "", path)
    return int(stage_idx), relative


def replace_condconv_with_kernel_subspace(
    model,
    num_bases=4,
    residual_rank=4,
):
    """Share a learned kernel basis across the three blocks of each stage."""
    if model.stage_blocks is None:
        raise ValueError(
            "Dynamic kernel subspace requires stage-independent blocks"
        )
    excluded_heads = {
        "to_raw_spatial_out",
        "to_raw_feature_out",
        "to_raw_higher_order_out",
    }
    entries = []
    for stage_idx, stage in enumerate(model.stage_blocks):
        for block_idx, block in enumerate(stage):
            stack = [(f"stage_blocks.{stage_idx}.{block_idx}", block)]
            while stack:
                prefix, module = stack.pop()
                for child_name, child in list(module.named_children()):
                    path = f"{prefix}.{child_name}"
                    relative = re.sub(
                        r"^stage_blocks\.\d+\.\d+\.", "", path
                    )
                    if isinstance(child, CondConv2d):
                        if relative not in excluded_heads:
                            entries.append(
                                (
                                    stage_idx,
                                    path,
                                    module,
                                    child_name,
                                    child,
                                )
                            )
                    else:
                        stack.append((path, child))

    grouped = {}
    replacements = []
    for stage_idx, path, parent, child_name, old in entries:
        if old.router is not None:
            raise ValueError(
                "Dynamic kernel subspace expects the external SVCT router"
            )
        replacement_groups = min(old.groups, 2)
        replacement = KernelSubspaceCondConv2d(
            old.in_channels,
            old.out_channels,
            old.kernel_size,
            num_experts=old.num_experts,
            num_bases=num_bases,
            residual_rank=residual_rank,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=replacement_groups,
            bias=old.expert_biases is not None,
        )
        key = (
            *_stage_local_role(path, stage_idx),
            old.in_channels,
            old.out_channels,
            tuple(old.kernel_size),
            replacement_groups,
        )
        grouped.setdefault(key, []).append(replacement)
        replacements.append((parent, child_name, replacement))

    for modules in grouped.values():
        shared_basis = [
            nn.Parameter(item.detach().clone())
            for item in modules[0].basis_weights
        ]
        for module in modules:
            for index, parameter in enumerate(shared_basis):
                module.basis_weights[index] = parameter

    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)
    return len(replacements), len(grouped)


def prune_inactive_hermite_heads(model, cond_dim=96, num_experts=6):
    """Remove stage-inactive high-mode outputs without changing active modes."""
    if model.stage_blocks is None or not model.hermite_brane:
        return 0
    removed = 0
    for stage_idx, (stage, order) in enumerate(
        zip(model.stage_blocks, model.gaussian_stage_orders)
    ):
        active_high_modes = (int(order) + 1) ** 2 - 1
        for block in stage:
            old = block.to_raw_higher_order_out
            if old is None:
                continue
            old_parameters = sum(p.numel() for p in old.parameters())
            if active_high_modes == 0:
                block.to_raw_higher_order_out = None
                block.brane_high_mode_count = 0
                block.brane_max_degree = 0
                removed += old_parameters
                continue
            brane_vectors = (
                1
                if block.brane_shared_shape
                else block.brane_feature_channels
                + int(block.higher_order_cgls)
            )
            output_channels = brane_vectors * active_high_modes
            if output_channels == old.out_channels:
                block.brane_high_mode_count = active_high_modes
                block.brane_max_degree = int(order)
                continue
            compact = CondConv2d(
                old.in_channels,
                output_channels,
                old.kernel_size,
                cond_dim=cond_dim,
                num_experts=num_experts,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=old.expert_biases is not None,
                use_internal_router=old.router is not None,
            )
            compact.zero_init()
            block.to_raw_higher_order_out = compact
            block.brane_high_mode_count = active_high_modes
            block.brane_max_degree = int(order)
            removed += old_parameters - sum(
                p.numel() for p in compact.parameters()
            )
    return removed
