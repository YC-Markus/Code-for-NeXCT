"""Shared FBP-target, multi-view, approximate lower-dose real-data protocol."""
import hashlib
import os
from pathlib import Path
import torch
from .core import utils

FULL_VIEWS = 1152
SPARSE_VIEWS = 72
VIEW_COUNTS = (72, 144, 288)
IMAGE_SIZE = 368
STAGE_RESOLUTIONS = (46, 92, 184, 368)
PIXEL_SPACING = 1.4285714285714286 / 2.0
DET_SPACING = 1.285839319229126 * 2.0
SINO_SCALE = (1.4285714285714286 / (2.0 * 0.0192)) / 1.4
EFFECTIVE_I0 = float(os.environ.get('SVCT_EFFECTIVE_I0', '1000000'))
ELECTRONIC_STD = float(os.environ.get('SVCT_ELECTRONIC_STD', '10'))
DATA_SEED = 20260906
_generators = {}
PROTOCOL = dict(version='real_fbp_ldsvct_v1', full_views=FULL_VIEWS,
                train_views=VIEW_COUNTS, view_sampling='uniform_per_batch',
                dose_sampling='uniform_per_sample_[0.25,0.50]',
                effective_i0=EFFECTIVE_I0, electronic_std=ELECTRONIC_STD,
                noise_model='post_rebin_independent_gaussian_incremental_QE',
                dose_calibrated=False, stored_log_scale_assumption=1.0,
                target='full1152_standard_dose_ramp_FBP_no_clamp_no_LS',
                loss_target='raw_FBP', metrics='center256_clipped01',
                seed=DATA_SEED, validation_shift=64)


def make_radon(view_count):
    return utils.fanbeam_gen((view_count, view_count), img_size=IMAGE_SIZE,
                            bias=0, pixel_spacing=PIXEL_SPACING,
                            det_spacing=DET_SPACING, det_count=IMAGE_SIZE)


@torch.no_grad()
def preprocess_real(batch, device, full_radon, validation=False,
                    view_count=None, dose_fraction=.375, return_metadata=False):
    p = batch['sino'].to(device, non_blocking=True).float().clone()
    if p.shape[1:] != (1, FULL_VIEWS, IMAGE_SIZE) or not torch.isfinite(p).all():
        raise ValueError('Invalid cached sinogram shape/values')
    if EFFECTIVE_I0 <= 0 or ELECTRONIC_STD < 0:
        raise ValueError('Invalid effective noise parameters')
    key = str(device)
    if key not in _generators:
        _generators[key] = torch.Generator(device=device).manual_seed(DATA_SEED)
    g = _generators[key]
    b = p.shape[0]
    if validation:
        p = torch.roll(p, shifts=-64, dims=2)
        dose = p.new_full((b, 1, 1, 1), float(dose_fraction))
        if not 0 < dose_fraction <= 1:
            raise ValueError('Invalid validation dose')
        noise = []
        for item, path in zip(p, batch['path']):
            digest = hashlib.sha256(os.path.basename(path).encode()).digest()
            seed = (int.from_bytes(digest[:8], 'little') + DATA_SEED) % (2**63-1)
            eg = torch.Generator(device=device).manual_seed(seed)
            noise.append(torch.randn(item.shape, device=device, generator=eg))
        z = torch.stack(noise)
    else:
        flip = torch.rand(b, device=device, generator=g) < .5
        p[flip] = torch.flip(p[flip], dims=(2, 3))
        shifts = torch.randint(0, FULL_VIEWS, (b,), device=device, generator=g)
        p = torch.stack([torch.roll(x, -int(s), dims=1) for x, s in zip(p, shifts)])
        dose = .25 + .25 * torch.rand((b, 1, 1, 1), device=device, generator=g)
        z = torch.randn(p.shape, device=device, generator=g)
    target = utils.recon((p * SINO_SCALE).contiguous(), full_radon)
    transmitted = EFFECTIVE_I0 * torch.exp(-p)
    q = 1 / transmitted
    e = (ELECTRONIC_STD / transmitted) ** 2
    variance = (1 / dose - 1) * q + (1 / dose.square() - 1) * e
    noisy = (p + variance.sqrt() * z) * SINO_SCALE
    if not torch.isfinite(noisy).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite noise or FBP target')
    if view_count is not None:
        if view_count not in VIEW_COUNTS and view_count != FULL_VIEWS:
            raise ValueError('Unsupported view count')
        noisy = noisy[:, :, ::FULL_VIEWS // view_count].contiguous()
    if return_metadata:
        return target, noisy, dict(doses=dose.flatten().tolist(),
                                   stored_min=float(p.min()), stored_max=float(p.max()),
                                   max_log_noise_std=float(variance.sqrt().max()))
    return target, noisy
