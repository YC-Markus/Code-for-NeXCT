"""Public construction/loading API. Historical state-dict names are preserved."""

import json
from pathlib import Path
import torch
from .core.model import GCTCGLSTrajectoryContextUnrolled
from .core.utils import fanbeam_gen
from .training import set_stage_cgls_schedule


def read_config(path):
    return json.loads(Path(path).read_text())


def build_model(config, device="cuda"):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("HerKry requires Linux, an NVIDIA GPU, Triton and TorchRadon.")
    model = GCTCGLSTrajectoryContextUnrolled(
        render_size=config["image_size"],
        stage_resolutions=tuple(config["stage_resolutions"]),
        blocks_per_stage=3,
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
        gaussian_stage_orders=(3, 2, 1, 0),
        kdelta_width=16,
    ).to(device)
    model.CACHE_CONFIGS = {int(k): v for (k, v) in config["cache_configs"].items()}
    for blocks in model.stage_blocks:
        for block in blocks:
            block.full_hg_auxiliary_render = config["full_hg_auxiliary_render"]
    set_stage_cgls_schedule(model, config["eval_depths"])
    return model


def load_checkpoint(model, path, *, trusted_legacy=False):
    """Load a state dict or our checkpoint; pickle fallback is explicit opt-in."""
    payload = torch.load(path, map_location="cpu", weights_only=not trusted_legacy)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return payload


def make_projector(config, views):
    geometry = config["geometry"]
    return fanbeam_gen(
        (views, views),
        img_size=config["image_size"],
        bias=0,
        det_count=geometry["det_count"],
        pixel_spacing=geometry["pixel_spacing"],
        det_spacing=geometry["det_spacing"],
    )


def reconstruct(model, sinogram, projector, *, return_aux=False):
    """Input [B,1,V,D]; outputs 12 native-grid visual states (not solver states)."""
    return model(
        sinogram,
        {model.render_size: projector},
        view_counts=sinogram.new_full((len(sinogram),), float(sinogram.shape[-2])),
        return_aux=return_aux,
    )
