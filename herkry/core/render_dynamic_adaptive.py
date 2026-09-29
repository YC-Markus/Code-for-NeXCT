import math
import torch
import torch.nn as nn
from herkry.core.render_backable import GaussianRenderer


class AdaptiveDynamicGaussianRenderer(nn.Module):
    """Dynamic renderer for unconstrained Gaussian sigma."""

    DEFAULT_TRAIN_CONFIGS = {
        (32, 24): (16, 16),
        (64, 32): (8, 32),
        (128, 24): (4, 32),
        (256, 16): (8, 64),
    }
    DEFAULT_INFERENCE_CONFIGS = {
        (32, 1): (32, 16),
        (64, 1): (16, 4),
        (128, 1): (32, 4),
        (256, 1): (16, 4),
        (32, 24): (16, 16),
        (64, 24): (16, 4),
        (64, 32): (16, 4),
        (128, 24): (8, 8),
        (256, 16): (8, 16),
        (256, 24): (8, 8),
    }

    def __init__(
        self,
        H,
        W,
        train_tile_size=4,
        train_chunk=32,
        min_sigma_px=0.1,
        train_configs=None,
        inference_configs=None,
    ):
        super().__init__()
        self.train_config = (int(train_tile_size), int(train_chunk))
        self.train_configs = dict(self.DEFAULT_TRAIN_CONFIGS)
        if train_configs is not None:
            self.train_configs.update(train_configs)
        self.inference_configs = dict(self.DEFAULT_INFERENCE_CONFIGS)
        if inference_configs is not None:
            self.inference_configs.update(inference_configs)
        self.renderer = GaussianRenderer(
            H=H,
            W=W,
            tile_size=self.train_config[0],
            max_gauss_chunk=self.train_config[1],
            min_sigma_px=min_sigma_px,
        )

    @staticmethod
    def _shape_key(gs_params):
        n = gs_params.shape[1]
        c = gs_params.shape[2] - 5
        return (int(math.isqrt(n)), c)

    def train_forward(self, gs_params, H=None, W=None, **kwargs):
        if not kwargs:
            kwargs = {}
            config = self.train_configs.get(self._shape_key(gs_params))
            if config is not None:
                (kwargs["tile_size"], kwargs["max_gauss_chunk"]) = config
        return self.renderer(gs_params, H=H, W=W, **kwargs)

    def inference_forward(self, gs_params, H=None, W=None, **kwargs):
        if not kwargs:
            kwargs = {}
            config = self.inference_configs.get(self._shape_key(gs_params), self.train_config)
            (kwargs["tile_size"], kwargs["max_gauss_chunk"]) = config
        return self.renderer(gs_params, H=H, W=W, **kwargs)

    def forward(self, gs_params, H=None, W=None, **kwargs):
        if torch.is_grad_enabled() or kwargs:
            return self.train_forward(gs_params, H=H, W=W, **kwargs)
        return self.inference_forward(gs_params, H=H, W=W)
