"""Krylov-trajectory GRU and stage-specific HG heads."""

import torch
import torch.nn as nn
from herkry.core.gaussian import CondConv2d


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
        self.convergence = nn.Conv2d(2 * self.mode_count, width, 1, groups=1)

    def forward(self, coefficients, increment, curvature):
        solution = self.solution(coefficients)
        dynamics = self.dynamics(torch.cat([increment, curvature], dim=1))
        interaction = solution + dynamics + solution * torch.sigmoid(dynamics)
        convergence = torch.sigmoid(
            self.convergence(
                torch.cat([torch.log1p(increment.abs()), torch.log1p(curvature.abs())], dim=1)
            )
        )
        return (interaction, convergence)


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
        reset = torch.sigmoid(self.reset_gate(torch.cat([state, trajectory], dim=1)))
        write = torch.sigmoid(self.write_gate(torch.cat([state, trajectory, convergence], dim=1)))
        candidate = torch.tanh(self.candidate(torch.cat([trajectory, reset * state], dim=1)))
        return state + write * (candidate - state)


class KrylovDeltaGRU(nn.Module):
    """Compress a variable-length Krylov coefficient trajectory into memory."""

    def __init__(self, stage_orders, persistent_channels=48, width=16):
        super().__init__()
        self.stage_orders = tuple((int(order) for order in stage_orders))
        self.persistent_channels = int(persistent_channels)
        self.width = int(width)
        self.encoders = nn.ModuleList(
            [KrylovDeltaEncoder((order + 1) ** 2, width=self.width) for order in self.stage_orders]
        )
        self.state_down = nn.Conv2d(self.persistent_channels, self.width, 1, groups=1)
        self.cell = KrylovDeltaGRUCell(width=self.width)
        self.state_up = nn.Conv2d(self.width, self.persistent_channels, 1, groups=1)
        nn.init.normal_(self.state_up.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.state_up.bias)

    @staticmethod
    def _to_map(values, batch, modes, height, width):
        return values.transpose(1, 2).reshape(batch, modes, height, width)

    def forward(self, selected_amplitudes, hidden, stage_idx):
        if selected_amplitudes.ndim != 4:
            raise ValueError("KDelta trajectory must have shape [B,T,N,M]")
        if selected_amplitudes.shape[1] < 2:
            raise ValueError("KDelta GRU requires an initial state and at least one update")
        (batch, _, gaussian_count, mode_count) = selected_amplitudes.shape
        (height, width) = hidden.shape[-2:]
        if gaussian_count != height * width:
            raise ValueError("Gaussian lattice does not match persistent resolution")
        expected_modes = (self.stage_orders[int(stage_idx)] + 1) ** 2
        if mode_count != expected_modes:
            raise ValueError(f"stage {stage_idx} expects {expected_modes} modes, got {mode_count}")
        encoder = self.encoders[int(stage_idx)]
        initial_state = self.state_down(hidden)
        state = initial_state
        previous = selected_amplitudes[:, 0]
        previous_increment = torch.zeros_like(previous)
        for current in selected_amplitudes[:, 1:].unbind(dim=1):
            increment = current - previous
            curvature = increment - previous_increment
            coefficient_map = self._to_map(current, batch, mode_count, height, width)
            increment_map = self._to_map(increment, batch, mode_count, height, width)
            curvature_map = self._to_map(curvature, batch, mode_count, height, width)
            (trajectory, convergence) = encoder(coefficient_map, increment_map, curvature_map)
            state = self.cell(trajectory, convergence, state)
            previous = current
            previous_increment = increment
        return hidden + self.state_up(state - initial_state)


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
            old_parameters = sum((p.numel() for p in old.parameters()))
            if active_high_modes == 0:
                block.to_raw_higher_order_out = None
                block.brane_high_mode_count = 0
                block.brane_max_degree = 0
                removed += old_parameters
                continue
            brane_vectors = (
                1
                if block.brane_shared_shape
                else block.brane_feature_channels + int(block.higher_order_cgls)
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
            removed += old_parameters - sum((p.numel() for p in compact.parameters()))
    return removed
