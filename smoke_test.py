"""Synthetic-phantom CUDA forward/backward smoke; no clinical data needed."""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/aapm.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    import torch
    from herkry.api import read_config, build_model, make_projector, reconstruct
    from herkry.training import image_only_loss, pre_cgls_image_loss

    torch.manual_seed(3407)
    c = read_config(args.config)
    m = build_model(c, args.device)
    m.train(args.backward)
    n = c["image_size"]
    z = torch.linspace(-1,1,n, device=args.device)
    y,x = torch.meshgrid(z,z,indexing="ij")
    gt = (.6*((x/.7)**2+(y/.8)**2 < 1).float()
          +.2*((x-.2)**2+(y+.1)**2 < .03).float())[None,None]
    radon = make_projector(c, c["primary_val_view"])
    with torch.set_grad_enabled(args.backward):
        outputs, aux = reconstruct(m, radon.forward(gt), radon, return_aux=True)
        assert len(outputs) == 12
        assert sum(p.numel() for p in m.parameters()) == 8521068
        assert all(x.shape == gt.shape and torch.isfinite(x).all() for x in outputs)
        loss = image_only_loss(m, outputs, gt, 2)+pre_cgls_image_loss(m, aux, gt, 2, (1,.5,.25,0))
        assert torch.isfinite(loss)
        if args.backward:
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    print(json.dumps(dict(status="passed", blocks=len(outputs), parameters=8521068,
                         loss=float(loss.detach()), backward=args.backward,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20), indent=2))


if __name__ == "__main__":
    main()
