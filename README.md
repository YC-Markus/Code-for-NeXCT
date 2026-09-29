# HerKry

Code for **HerKry: A Hermite–Gaussian Hierarchy with Recurrent Krylov Trajectory
Modeling for Robust Sparse-View CT Reconstruction**.

HerKry represents a CT reconstruction with an order–density hierarchy of continuous
Hermite–Gaussian primitives. HC-CGLS refines linear coefficients at the native
measurement geometry, while a Krylov-Trajectory GRU carries solver information
between reconstruction blocks. The final output is the learned **visual state**,
not the final CGLS iterate.

This release contains the reconstruction model, differentiable Triton renderers,
training and evaluation entry points, AAPM/MSD/LDCT configurations, and tests.
It does **not** bundle clinical data, pretrained checkpoints, third-party baseline
implementations, server launchers, or experimental model-selection scripts.

## Installation

Use Linux and an NVIDIA CUDA GPU. The original environment used Python 3.12,
PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, Triton 3.3.1,
MONAI 1.5.1 and MONAI Generative 0.2.3. Training used 48-GB RTX A6000 GPUs;
the default AAPM batch of 8 and LDCT batch of 4 are memory intensive.

1. Install the CUDA build of PyTorch/torchvision appropriate for your system.
2. Build [TorchRadon](https://github.com/matteo-ronchetti/torch-radon) against that
   PyTorch/CUDA environment. It is a separate compiled dependency, not installed
   by this package. Its upstream installation instructions may require adaptation
   to modern CUDA/PyTorch versions; a fresh installation has not been validated
   on every platform.
3. From this repository:

```bash
python -m pip install -e '.[data]'
python -m unittest discover -s tests -v
python smoke_test.py --config configs/aapm.json --backward
```

The smoke test uses a synthetic phantom and requires no medical data. CPU-only
execution of the reconstruction model and Windows/macOS CUDA execution are not
supported. LPIPS evaluation may download its pretrained AlexNet weights on first use.

## Data

Obtain the datasets under their original access terms. Create a JSON manifest
containing **patient-disjoint** `train`, `val`, `test` lists; optional `ood` is
evaluation-only. Paths are relative to the manifest. See [data preparation](docs/DATA.md)
for normalization, shapes, splits and the important distinction between simulated
AAPM/MSD measurements and rebinned LDCT data.

```json
{
  "train": [{"patient": "case_a", "path": "images/case_a.npy", "slice": 0}],
  "val":   [{"patient": "case_b", "path": "images/case_b.npy", "slice": 0}],
  "test":  [{"patient": "case_c", "path": "images/case_c.npy", "slice": 0}]
}
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/aapm.json --manifest data/aapm/manifest.json \
  --output outputs/aapm
```

Use `configs/msd.json` or `configs/ldct.json` with the corresponding manifest.
Training starts from scratch and refuses to overwrite an existing output folder.
The default is 50 epochs. Validation PSNR selects `best.pth`; test data are never
used for training or selection. The compact trainer preserves the model, loss,
depth distribution and optimizer definitions, but is not a bitwise replay of
historical data-loader order or interrupted/resumed runs.

## Evaluation and intermediate states

```bash
python evaluate.py --config configs/aapm.json \
  --manifest data/aapm/manifest.json --checkpoint outputs/aapm/best.pth \
  --split test --views 32 --output outputs/aapm_test32

python evaluate.py --config configs/aapm.json \
  --manifest data/aapm/manifest.json --checkpoint outputs/aapm/best.pth \
  --split val --views 32 --export-blocks --no-lpips \
  --output outputs/aapm_blocks
```

For LDCT, set `--dose 0.25` (or another supported retained-dose fraction). The
evaluation writes per-slice PSNR, SSIM, LPIPS and observed-angle projection L1,
along with checkpoint/manifest hashes. `--export-blocks` also writes all 12 visual
states and available auxiliary pre-CGLS tensors. It can consume substantial disk
space; use a small manifest for qualitative inspection. No post-hoc GT mean
matching or display scaling is applied. Only load trusted historical pickled
checkpoints with `--trusted-legacy-checkpoint`; ordinary state dictionaries use
PyTorch's restricted loader by default.

## Code map

| Paper component | Implementation |
|---|---|
| HG hierarchy and reconstruction chain | `herkry/core/model.py`, `gaussian.py` |
| Differentiable HG rendering and transpose | `herkry/core/render_gaussian_hermite_fused.py` |
| Cached HC-CGLS | `unrolled_cached_cgls` in `herkry/core/model.py` |
| KT-GRU, internally named `KrylovDeltaGRU` | `herkry/core/solverdna.py` |
| Spatial trajectory encoding / persistent fusion | `herkry/core/blocks.py`, `model.py` |
| Direct + pre-CGLS supervision / Muon partition | `herkry/training.py`, `optim.py` |
| Acquisition, noise and metrics | `api.py`, `real_data.py`, `metrics.py` |

Historical internal names and low-level compatibility branches are retained to
keep checkpoint parameter keys stable. The public constructor selects only the
released HerKry configuration. See [method details](docs/METHOD.md),
[reproducibility notes](docs/REPRODUCIBILITY.md) and
[third-party acknowledgements](THIRD_PARTY_NOTICES.md).
Measured checkpoint-compatibility checks are recorded in [validation](validation/README.md).
