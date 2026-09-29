"""Evaluate a frozen HerKry checkpoint with per-slice metrics and stage export."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test", "ood"), default="test")
    p.add_argument("--views", type=int, required=True)
    p.add_argument("--dose", type=float, default=.375)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--export-blocks", action="store_true", help="Save all 12 visual and available pre-CGLS states")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--trusted-legacy-checkpoint", action="store_true", help="Allow pickle only for a checkpoint you trust")
    args = p.parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from herkry.api import build_model, read_config, load_checkpoint, make_projector, reconstruct
    from herkry.data import ManifestDataset
    from herkry.engine import acquisition
    from herkry.metrics import ImageMetrics

    config = read_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = True
    model = build_model(config, args.device).eval()
    payload = load_checkpoint(model, args.checkpoint, trusted_legacy=args.trusted_legacy_checkpoint)
    kind = "sino" if config["dataset"] == "ldct" else "image"
    ds = ManifestDataset(args.manifest, args.split, kind)
    loader = DataLoader(ds, batch_size=args.batch_size or config["batch_size"], shuffle=False)
    projectors = {args.views: make_projector(config, args.views)}
    if kind == "sino":
        projectors[1152] = make_projector(config, 1152)
    radon = projectors[args.views]
    metrics = ImageMetrics(args.device, lpips=not args.no_lpips)
    rows = []
    with torch.no_grad():
        for batch in loader:
            target, sino = acquisition(batch, config, args.device, projectors, args.views, dose=args.dose)
            result = reconstruct(model, sino, radon, return_aux=args.export_blocks)
            images, aux = result if args.export_blocks else (result, None)
            pred = images[-1]
            if not torch.isfinite(pred).all():
                raise FloatingPointError("Nonfinite reconstruction")
            scores = metrics(pred, target, config["metric_crop"])
            # Observed-angle clean/reference sinogram fidelity; never crop before A.
            clean = radon.forward(target.contiguous())
            scores["dc_l1"] = (radon.forward(pred.clamp(0,1).contiguous())-clean).abs().flatten(1).mean(1).cpu().tolist()
            for j, index in enumerate(batch["index"].tolist()):
                rows.append(dict(index=index, patient=batch["patient"][j], **{k:v[j] for k,v in scores.items()}))
                if args.export_blocks:
                    arrays = {f"block_{b+1:02d}_visual": x[j,0].cpu().numpy() for b,x in enumerate(images)}
                    arrays.update({f"block_{b+1:02d}_initial": a["cgls_initial_image_for_loss"][j,0].cpu().numpy()
                                   for b,a in enumerate(aux) if a.get("cgls_initial_image_for_loss") is not None})
                    np.savez_compressed(out/f"sample_{index:06d}.npz", target=target[j,0].cpu().numpy(), **arrays)
            print(f"{len(rows)}/{len(ds)}", flush=True)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in scores}
    report = dict(config=config, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  checkpoint_epoch=payload.get("epoch"), split=args.split, views=args.views, dose=args.dose,
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  aggregation="slice mean", samples=len(rows), metrics=summary, rows=rows)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
