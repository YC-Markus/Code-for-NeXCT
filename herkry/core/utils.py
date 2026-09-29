"""Projection helpers retained from the evaluated implementation."""

import copy
import numpy as np
import torch
from torch_radon import RadonFanbeam


def recon(projections, radon):
    sino = projections
    filtered_sinogram = radon.filter_sinogram(sino, filter_name="ramp")
    fbp = radon.backprojection(filtered_sinogram)
    return fbp


def fanbeam_gen(
    sparsity, img_size, bias=0, det_spacing=1.2858, pixel_spacing=1.4285, det_count=None
):
    source_det_distance = 1085.6
    source_distance = 595.0
    if det_count == None:
        det_count = img_size
    (sparse_scan, full_scan) = sparsity
    index = full_scan // sparse_scan
    assert full_scan % sparse_scan == 0
    angles = np.linspace(0, 2 * np.pi, full_scan, endpoint=False)
    seq = np.arange(0, full_scan, 1)
    if bias < 0:
        bias = len(seq) + bias
    seq = np.roll(seq, -bias)
    result_seq = seq[::index]
    angles = angles[result_seq]
    ops_example = RadonFanbeam(
        resolution=img_size,
        angles=angles,
        source_distance=source_distance * pixel_spacing,
        det_distance=(source_det_distance - source_distance) * pixel_spacing,
        det_count=det_count,
        det_spacing=det_spacing * pixel_spacing,
        clip_to_circle=True,
    )
    return ops_example


class ModelEMA:

    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_params, model_params in zip(self.ema_model.parameters(), model.parameters()):
                ema_params.data *= self.decay
                ema_params.data += (1.0 - self.decay) * model_params.data

    def get(self):
        self.ema_model.eval()
        return self.ema_model
