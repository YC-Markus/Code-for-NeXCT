"""Shared acquisition and validation without implicit test-set access."""
import torch
from .api import reconstruct
from .data import prepare_image
from .real_data import preprocess_real
from .metrics import metric_images


def acquisition(batch, config, device, projectors, views, training=False, dose=.375):
    if config["dataset"] == "ldct":
        return preprocess_real(batch, device, projectors[1152],
                               validation=not training, view_count=views,
                               dose_fraction=dose)
    target = prepare_image(batch, device, config["image_size"], training)
    return target, projectors[views].forward(target.contiguous())


@torch.no_grad()
def validation_psnr(model, loader, config, device, projectors):
    model.eval()
    views = config["primary_val_view"]
    total, count = 0., 0
    for batch in loader:
        target, sino = acquisition(batch, config, device, projectors, views)
        output = reconstruct(model, sino, projectors[views])[-1]
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite validation reconstruction")
        pred, ref = metric_images(output, target, config["metric_crop"])
        values = -10*(pred-ref).square().flatten(1).mean(1).log10()
        total += values.sum().item()
        count += len(values)
    return total/count
