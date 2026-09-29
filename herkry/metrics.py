"""Paper image metrics; full-image AAPM/MSD, center-256 LDCT."""
import torch


def metric_images(prediction, target, crop_size=None):
    if crop_size is not None:
        h, w = prediction.shape[-2:]
        if min(h, w) < crop_size:
            raise ValueError("Metric crop exceeds image size")
        y, x = (h-crop_size)//2, (w-crop_size)//2
        prediction = prediction[..., y:y+crop_size, x:x+crop_size]
        target = target[..., y:y+crop_size, x:x+crop_size]
    return prediction.clamp(0, 1), target.clamp(0, 1)


class ImageMetrics:
    def __init__(self, device, lpips=True):
        from generative.metrics import SSIMMetric
        self.ssim = SSIMMetric(spatial_dims=2, data_range=1., reduction="none")
        self.lpips = None
        if lpips:
            from generative.losses import PerceptualLoss
            self.lpips = PerceptualLoss(spatial_dims=2, network_type="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction, target, crop_size=None):
        prediction, target = metric_images(prediction, target, crop_size)
        mse = (prediction-target).square().flatten(1).mean(1)
        scores = {"psnr": -10*mse.log10(), "ssim": self.ssim(prediction, target).flatten()}
        if self.lpips is not None:
            scores["lpips"] = torch.stack([
                self.lpips(p[None], t[None]).reshape(()) for p, t in zip(prediction, target)
            ])
        return {k: v.cpu().tolist() for k, v in scores.items()}
