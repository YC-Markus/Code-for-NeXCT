import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from herkry.core.backbone import LayerNorm2d, NAFBlock
from herkry.core.render_dynamic_adaptive import AdaptiveDynamicGaussianRenderer
from herkry.core.utils import recon


class ImageFeatureEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, render_size, use_naf=True):
        super().__init__()
        layers = []
        current_channels = in_channels
        current_size = render_size

        while current_size > resolution:
            layers += [
                nn.Conv2d(current_channels, out_channels, 3, stride=2, padding=1),
                nn.GELU(),
            ]
            current_channels = out_channels
            current_size //= 2

        if not layers:
            layers = [
                nn.Conv2d(current_channels, out_channels, 3, padding=1),
                nn.GELU(),
            ]

        if use_naf:
            layers += [NAFBlock(out_channels)]
        self.net = nn.Sequential(*layers)
        self.view_films = None

    def enable_view_conditioning(self, condition_dim=2):
        """Add zero-initialized FiLM after each convolution."""
        if self.view_films is not None:
            return
        films = []
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                film = nn.Linear(condition_dim, 2 * layer.out_channels)
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
                films.append(film)
        self.view_films = nn.ModuleList(films)

    def forward(self, x, view_condition=None):
        if self.view_films is None:
            return self.net(x)
        if view_condition is None:
            raise ValueError(
                "view_condition is required for a view-conditioned encoder"
            )
        film_idx = 0
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                gamma, beta = self.view_films[film_idx](
                    view_condition.to(dtype=x.dtype)
                ).chunk(2, dim=1)
                x = x * (1.0 + gamma[:, :, None, None])
                x = x + beta[:, :, None, None]
                film_idx += 1
        return x


class FeatureUpsampleAdapter(nn.Module):
    def __init__(self, channels, out_resolution):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(size=(out_resolution, out_resolution), mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            NAFBlock(channels),
        )

    def forward(self, x):
        return self.net(x)


class CondConv2d(nn.Module):
    """
    Per-sample conditional convolution with expert kernels.

    The routing network mixes expert weights into one effective kernel per sample,
    so the convolutional FLOPs stay close to a single Conv2d while parameters grow
    with the number of experts.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        cond_dim,
        num_experts=6,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        expert_noise_scale=0.02,
        use_internal_router=True,
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        self.num_experts = int(num_experts)
        self.expert_noise_scale = float(expert_noise_scale)
        if self.in_channels % self.groups != 0:
            raise ValueError(f"in_channels ({self.in_channels}) must be divisible by groups ({self.groups})")
        if self.out_channels % self.groups != 0:
            raise ValueError(f"out_channels ({self.out_channels}) must be divisible by groups ({self.groups})")

        self.expert_weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.out_channels, self.in_channels // self.groups, kh, kw))
                for _ in range(self.num_experts)
            ]
        )
        self.expert_biases = (
            nn.ParameterList(
                [nn.Parameter(torch.empty(self.out_channels)) for _ in range(self.num_experts)]
            )
            if bias
            else None
        )
        self.use_internal_router = bool(use_internal_router)
        if self.use_internal_router:
            hidden = max(32, cond_dim // 2)
            self.router = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.num_experts),
            )
        else:
            self.router = None
        self.reset_parameters()

    def reset_parameters(self):
        base_weight = torch.empty_like(self.expert_weights[0])
        nn.init.kaiming_uniform_(base_weight, a=math.sqrt(5))
        if self.num_experts > 1 and self.expert_noise_scale > 0.0:
            noise = torch.empty(
                self.num_experts,
                self.out_channels,
                self.in_channels // self.groups,
                *self.kernel_size,
                device=base_weight.device,
                dtype=base_weight.dtype,
            )
            for expert_id in range(self.num_experts):
                nn.init.kaiming_uniform_(noise[expert_id], a=math.sqrt(5))
            noise = noise - noise.mean(dim=0, keepdim=True)
            for expert_id, weight in enumerate(self.expert_weights):
                weight.data.copy_(base_weight + self.expert_noise_scale * noise[expert_id])
        else:
            for weight in self.expert_weights:
                weight.data.copy_(base_weight)

        if self.expert_biases is not None:
            fan_in = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in)
            base_bias = torch.empty_like(self.expert_biases[0])
            nn.init.uniform_(base_bias, -bound, bound)
            if self.num_experts > 1 and self.expert_noise_scale > 0.0:
                noise = torch.empty(
                    self.num_experts,
                    self.out_channels,
                    device=base_bias.device,
                    dtype=base_bias.dtype,
                )
                nn.init.uniform_(noise, -bound, bound)
                noise = noise - noise.mean(dim=0, keepdim=True)
                for expert_id, bias in enumerate(self.expert_biases):
                    bias.data.copy_(base_bias + self.expert_noise_scale * noise[expert_id])
            else:
                for bias in self.expert_biases:
                    bias.data.copy_(base_bias)
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

    def zero_init(self):
        if self.num_experts > 1 and self.expert_noise_scale > 0.0:
            noise = torch.empty(
                self.num_experts,
                self.out_channels,
                self.in_channels // self.groups,
                *self.kernel_size,
                device=self.expert_weights[0].device,
                dtype=self.expert_weights[0].dtype,
            )
            for expert_id in range(self.num_experts):
                nn.init.kaiming_uniform_(noise[expert_id], a=math.sqrt(5))
            noise = self.expert_noise_scale * (noise - noise.mean(dim=0, keepdim=True))
            for expert_id, weight in enumerate(self.expert_weights):
                weight.data.copy_(noise[expert_id])
        else:
            for weight in self.expert_weights:
                nn.init.zeros_(weight)
        if self.expert_biases is not None:
            for bias in self.expert_biases:
                nn.init.zeros_(bias)

    def forward(self, x, cond, routing=None):
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"CondConv2d expected {self.in_channels} channels, got {c}")

        if routing is None:
            if self.router is None:
                raise ValueError(
                    "External-routing CondConv2d requires routing weights"
                )
            routing = torch.softmax(self.router(cond), dim=-1)
        if routing.shape != (b, self.num_experts):
            raise ValueError(
                "routing must have shape "
                f"({b}, {self.num_experts}), got {tuple(routing.shape)}"
            )
        routing = routing.to(device=x.device, dtype=x.dtype)
        expert_weight = torch.stack([weight for weight in self.expert_weights], dim=0).to(dtype=x.dtype)
        weight = torch.einsum("be,eocij->bocij", routing, expert_weight)
        weight = weight.reshape(
            b * self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )

        if self.expert_biases is None:
            bias = None
        else:
            expert_bias = torch.stack([bias for bias in self.expert_biases], dim=0).to(dtype=x.dtype)
            bias = torch.einsum("be,eo->bo", routing, expert_bias).reshape(-1)

        x_grouped = x.contiguous().reshape(1, b * self.in_channels, h, w)
        out = F.conv2d(
            x_grouped,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b * self.groups,
        )
        return out.reshape(b, self.out_channels, out.shape[-2], out.shape[-1])


class FeatureGaussianDecoder(nn.Module):
    def __init__(
        self,
        resolution,
        render_size,
        token_channels,
        residual_scale=1.0,
        max_offset_cells=0.5,
        anchor_offset_cells=0.5,
        sigma_reference_cell_fraction=None,
        feature_activation="tanh",
    ):
        super().__init__()
        self.resolution = resolution
        self.token_channels = token_channels
        self.residual_scale = residual_scale

        ys, xs = torch.meshgrid(
            torch.arange(resolution, dtype=torch.float32),
            torch.arange(resolution, dtype=torch.float32),
            indexing="ij",
        )
        anchor_offset_cells = float(anchor_offset_cells)
        if not 0.0 <= anchor_offset_cells <= 1.0:
            raise ValueError("anchor_offset_cells must be in [0, 1]")
        anchors = torch.stack([(xs + anchor_offset_cells) / resolution, (ys + anchor_offset_cells) / resolution], dim=-1)
        self.register_buffer("anchors", anchors.view(1, resolution * resolution, 2), persistent=False)

        spacing_px = render_size / resolution
        self.max_offset_cells = float(max_offset_cells)
        self.max_offset = self.max_offset_cells / resolution
        self.sigma_reference_cell_fraction = (
            None
            if sigma_reference_cell_fraction is None
            else float(sigma_reference_cell_fraction)
        )
        self.feature_activation = str(feature_activation)
        if self.feature_activation not in ("tanh", "identity"):
            raise ValueError(
                "feature_activation must be 'tanh' or 'identity'"
            )
        if self.sigma_reference_cell_fraction is None:
            base_sigma_px = max(0.75, 0.75 * spacing_px)
            self.base_sigma = base_sigma_px / render_size
            self.sigma_parameterization = "legacy_bounded_sigmoid"
        else:
            if self.sigma_reference_cell_fraction <= 0.0:
                raise ValueError(
                    "sigma reference cell fraction must be positive"
                )
            base_sigma_px = (
                self.sigma_reference_cell_fraction * spacing_px
            )
            self.base_sigma = base_sigma_px / render_size
            self.sigma_parameterization = (
                "cell_exp_tanh_half_to_double"
            )
        lattice_sum = 2.0 * math.pi * base_sigma_px * base_sigma_px / (spacing_px * spacing_px)
        self.basis_gain = 1.0 / max(1.0, lattice_sum)

    def decode_feature_map(self, raw_feature):
        """Decode an arbitrary coefficient map with the stage's gain policy."""

        b, channels, h, w = raw_feature.shape
        feature = raw_feature.permute(0, 2, 3, 1).reshape(
            b, h * w, channels
        )
        if self.feature_activation == "tanh":
            feature = torch.tanh(feature)
        return self.residual_scale * feature * self.basis_gain

    def decode_bounded_feature_map(self, raw_feature):
        """Decode signed, bounded Brane mode coefficients.

        Unlike the ordinary residual/solver channel, every non-zero Hermite
        mode is explicitly bounded with tanh as in Resonant Brane Splatting.
        The existing lattice gain keeps its starting scale compatible with
        the zero-order Gaussian path.
        """

        b, channels, h, w = raw_feature.shape
        feature = raw_feature.permute(0, 2, 3, 1).reshape(
            b, h * w, channels
        )
        return (
            self.residual_scale
            * torch.tanh(feature)
            * self.basis_gain
        )

    def forward(self, raw_spatial, raw_feature):
        b, _, h, w = raw_spatial.shape
        raw_spatial = raw_spatial.permute(0, 2, 3, 1).reshape(b, h * w, 5)

        center = self.anchors.to(raw_spatial) + torch.tanh(raw_spatial[..., 0:2]) * self.max_offset
        if self.sigma_reference_cell_fraction is None:
            sigma = self.base_sigma * (
                0.5 + 1.5 * torch.sigmoid(raw_spatial[..., 2:4])
            )
        else:
            sigma = self.base_sigma * torch.exp(
                math.log(2.0) * torch.tanh(raw_spatial[..., 2:4])
            )
        theta = math.pi * torch.tanh(raw_spatial[..., 4:5])
        feature = self.decode_feature_map(raw_feature)

        return torch.cat([center.clamp(0.0, 1.0), sigma, theta, feature], dim=-1)


class ConditionalFeatureToResidualDecoder(nn.Module):
    def __init__(self, token_channels, cond_dim, num_experts):
        super().__init__()
        hidden = max(16, token_channels)
        self.conv1 = CondConv2d(
            token_channels,
            hidden,
            3,
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=1,
        )
        self.conv2 = CondConv2d(
            hidden,
            hidden,
            3,
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=1,
        )
        self.conv3 = CondConv2d(
            hidden,
            1,
            3,
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=1,
        )
        self.conv3.zero_init()

    def forward(self, x, cond):
        x = F.gelu(self.conv1(x, cond))
        x = F.gelu(self.conv2(x, cond))
        return self.conv3(x, cond)


class ConditionalNAFBlock(nn.Module):
    """
    NAFBlock-shaped conditional block.

    It keeps the original norm/residual/FFN skeleton and multi-scale grouped
    spatial branch, replacing each convolution with base-initialized CondConv.
    """

    def __init__(
        self,
        channels,
        cond_dim,
        num_experts,
        DW_Expand=1,
        FFN_Expand=2,
        use_internal_router=True,
    ):
        super().__init__()
        dw_channel = channels * DW_Expand
        ffn_channel = channels * FFN_Expand
        if dw_channel % 4 != 0:
            raise ValueError(f"dw_channel ({dw_channel}) must be divisible by 4 for group=4 convolutions.")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = CondConv2d(
            channels,
            dw_channel,
            1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=use_internal_router,
        )

        self.conv_3x3 = CondConv2d(
            dw_channel,
            dw_channel,
            3,
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=1,
            groups=4,
            use_internal_router=use_internal_router,
        )
        self.conv_1x5 = CondConv2d(
            dw_channel,
            dw_channel,
            (5, 1),
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=(2, 0),
            groups=4,
            use_internal_router=use_internal_router,
        )
        self.conv_5x1 = CondConv2d(
            dw_channel,
            dw_channel,
            (1, 5),
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=(0, 2),
            groups=4,
            use_internal_router=use_internal_router,
        )
        self.conv_1x7 = CondConv2d(
            dw_channel,
            dw_channel,
            (7, 1),
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=(3, 0),
            groups=4,
            use_internal_router=use_internal_router,
        )
        self.conv_7x1 = CondConv2d(
            dw_channel,
            dw_channel,
            (1, 7),
            cond_dim=cond_dim,
            num_experts=num_experts,
            padding=(0, 3),
            groups=4,
            use_internal_router=use_internal_router,
        )
        self.conv_fusion = CondConv2d(
            3 * dw_channel,
            dw_channel,
            1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=use_internal_router,
        )
        self.conv3 = CondConv2d(
            dw_channel,
            channels,
            1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=use_internal_router,
        )

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = CondConv2d(
            channels,
            ffn_channel,
            1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=use_internal_router,
        )
        self.conv5 = CondConv2d(
            ffn_channel,
            channels,
            1,
            cond_dim=cond_dim,
            num_experts=num_experts,
            use_internal_router=use_internal_router,
        )
        self.act = nn.GELU()

    def forward(self, inp, cond, routing=None):
        x = self.norm1(inp)
        x = self.conv1(x, cond, routing=routing)
        x = self.act(x)

        x_3x3 = self.conv_3x3(x, cond, routing=routing)
        x_1x5 = self.conv_1x5(x, cond, routing=routing)
        x_5x5 = self.conv_5x1(x_1x5, cond, routing=routing)
        x_1x7 = self.conv_1x7(x, cond, routing=routing)
        x_7x1 = self.conv_7x1(x_1x7, cond, routing=routing)

        x_all = torch.cat([x_3x3, x_5x5, x_7x1], dim=1)
        x = self.conv_fusion(x_all, cond, routing=routing) * x
        x = self.conv3(x, cond, routing=routing)
        y = inp + x

        x = self.norm2(y)
        x = self.conv4(x, cond, routing=routing)
        x = self.act(x)
        x = self.conv5(x, cond, routing=routing)
        return y + x
