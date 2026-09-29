# Third-party acknowledgements

This release imports third-party software rather than vendoring its full code.
Consult each upstream project for its license and compatible installation:

- [TorchRadon](https://github.com/matteo-ronchetti/torch-radon), the differentiable
  fan-beam projection/backprojection dependency (GPL-3.0).
- [PyTorch](https://github.com/pytorch/pytorch),
  [torchvision](https://github.com/pytorch/vision), and
  [Triton](https://github.com/triton-lang/triton).
- [MONAI](https://github.com/Project-MONAI/MONAI) and
  [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels), including
  the SSIM and AlexNet perceptual metric wrappers.
- [Muon](https://github.com/KellerJordan/Muon): the optimizer in `herkry/optim.py`
  batches the reference Newton–Schulz/Muon update for same-shaped matrices.
  The upstream notice is retained in `licenses/Muon.txt`.
- [NAFNet](https://github.com/megvii-research/NAFNet): the backbone uses an adapted
  residual image-restoration block; its retained historical `NAFBlock` name does
  not imply an unmodified upstream NAFNet (this variant includes GELU and
  multi-kernel grouped convolutions). The upstream notices are retained in
  `licenses/NAFNet.txt`.

Dataset access and redistribution conditions are separate from software licenses.
No clinical data or third-party baseline checkpoints are included. Anonymization
must not remove attribution to upstream work. A new blanket license has not been
assigned by this cleanup; retain the destination repository's existing license
and review compatibility before changing redistribution terms.
