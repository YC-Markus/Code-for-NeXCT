# HerKry

Review code for **HerKry: A Hermite–Gaussian Hierarchy with Recurrent Krylov
Trajectory Modeling for Robust Sparse-View CT Reconstruction**.

Four stages combine Hermite–Gaussian representations, coefficient-space CGLS,
and a Krylov-trajectory GRU. The reconstruction is the final learned visual state.

## Setup

Linux, NVIDIA GPU, Python 3.10+. Tested with PyTorch 2.7.1+cu118 and Triton 3.3.1.
Install a matching CUDA PyTorch build and [TorchRadon](https://github.com/matteo-ronchetti/torch-radon), then:

```bash
pip install -e '.[data]'
```

## Train and evaluate

Prepare patient-disjoint arrays and a JSON manifest as described in [data](docs/DATA.md).
Data and pretrained weights are not included.

```bash
python train.py --config configs/aapm.json --manifest data/manifest.json --output outputs/run
python evaluate.py --config configs/aapm.json --manifest data/manifest.json \
  --checkpoint outputs/run/best.pth --views 32 --output outputs/test
```

Use `configs/msd.json` or `configs/ldct.json` for the other datasets. Evaluation
reports PSNR, SSIM and LPIPS; `--export-blocks` saves intermediate images.
Configs specify the executable settings, including mixed Muon/Adam optimization.

## Main files

- `herkry/core/model.py`: reconstruction chain and HC-CGLS.
- `herkry/core/trajectory.py`: Krylov-trajectory GRU.
- `herkry/core/render_gaussian_hermite_fused.py`: Hermite–Gaussian operators.
- `train.py`, `evaluate.py`: training and evaluation.

Third-party credits and licenses are listed in [acknowledgements](THIRD_PARTY_NOTICES.md).
