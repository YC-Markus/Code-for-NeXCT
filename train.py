"""Train the released HerKry configuration from scratch; never evaluates test data."""
import argparse
import hashlib
import json
import logging
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from herkry.api import build_model, make_projector, read_config, reconstruct
    from herkry.data import ManifestDataset
    from herkry.core.utils import ModelEMA
    from herkry.engine import acquisition, validation_psnr
    from herkry.training import build_optimizer, image_only_loss, pre_cgls_image_loss, set_stage_cgls_schedule

    config = read_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out/"train.log"), logging.StreamHandler()])
    logger = logging.getLogger("herkry")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = True
    kind = "sino" if config["dataset"] == "ldct" else "image"
    train = ManifestDataset(args.manifest, "train", kind)
    val = ManifestDataset(args.manifest, "val", kind)
    loader_generator = None
    if config["loader_seed"] is not None:
        loader_generator = torch.Generator().manual_seed(config["loader_seed"])
    kwargs = dict(batch_size=config["batch_size"], num_workers=args.workers,
                  pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, generator=loader_generator, **kwargs)
    val_loader = DataLoader(val, shuffle=False, **kwargs)
    model = build_model(config, args.device)
    optimizer = build_optimizer(model, config["adam_lr"], logger, mixer_optimizer="muon")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]*len(train_loader), eta_min=1e-6)
    views = set(config["train_views"]+[config["primary_val_view"]])
    if kind == "sino":
        views.add(1152)
    projectors = {v: make_projector(config, v) for v in views}
    view_rng = torch.Generator().manual_seed(config["seed"]+32064)
    depth_rng = torch.Generator().manual_seed(config["seed"]+2468)
    weights = torch.tensor(config["view_probabilities"], dtype=torch.float64)
    record = dict(config=config, manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()))
    (out/"protocol.json").write_text(json.dumps(record, indent=2))
    ema, best, step = None, -float("inf"), 0
    for epoch in range(1, config["epochs"]+1):
        if epoch == config["ema_start"]:
            ema = ModelEMA(model, config["ema_decay"])
        model.train()
        sums = [0., 0., 0.]
        for batch_number, batch in enumerate(train_loader, 1):
            depths = [int(torch.randint(lo, hi+1, (1,), generator=depth_rng))
                      for lo, hi in config["train_depth_ranges"]]
            set_stage_cgls_schedule(model, depths)
            v = config["train_views"][int(torch.multinomial(weights, 1, generator=view_rng))]
            target, sino = acquisition(batch, config, args.device, projectors, v, training=True)
            optimizer.zero_grad(set_to_none=True)
            images, aux = reconstruct(model, sino, projectors[v], return_aux=True)
            direct = image_only_loss(model, images, target, config["stage_loss_ratio"])
            initial = pre_cgls_image_loss(model, aux, target, config["stage_loss_ratio"],
                                          config["pre_cgls_stage_ratios"])
            loss = direct+initial
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; previous checkpoints are preserved")
            loss.backward()
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            step += 1
            for i, value in enumerate((loss, direct, initial)):
                sums[i] += value.item()
            if batch_number == 1 or batch_number % 50 == 0:
                logger.info("epoch=%d batch=%d/%d loss=%.7g direct=%.7g pre=%.7g views=%d depths=%s",
                            epoch, batch_number, len(train_loader), *(x/batch_number for x in sums), v, depths)
        evaluation_model = ema.get() if ema is not None else model
        set_stage_cgls_schedule(evaluation_model, config["eval_depths"])
        score = validation_psnr(evaluation_model, val_loader, config, args.device, projectors)
        payload = dict(epoch=epoch, global_step=step, config=config, best_val_psnr=max(best,score),
                       model_state_dict=evaluation_model.state_dict(), online_state_dict=model.state_dict(),
                       optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict())
        torch.save(payload, out/f"epoch_{epoch:03d}.pth")
        if score > best:
            best = score
            torch.save(payload, out/"best.pth")
        item = dict(epoch=epoch, loss=sums[0]/len(train_loader), direct=sums[1]/len(train_loader),
                    pre_cgls=sums[2]/len(train_loader), val_psnr=score, best=best)
        with (out/"history.jsonl").open("a") as f:
            f.write(json.dumps(item)+"\n")
        logger.info("%s", item)


if __name__ == "__main__":
    main()
