# Release checks

Validated on an RTX A6000, PyTorch 2.7.1+cu118 and Triton 3.3.1.

- Seven unit tests cover loss normalization, stage-weight gradients, patient
  split leakage rejection, metric crop/clipping and the three configurations.
- Synthetic phantom forward **and backward** passed at 256² and 368². All 12
  visual states and parameter gradients were finite. Parameter count remains
  8,521,068. Batch-1 peak allocated memory was approximately 4.2 / 8.5 GiB;
  this is not a training-batch memory estimate.
- AAPM e45, MSD e38 and LDCT e50 checkpoints load strictly with no missing or
  unexpected parameter keys. Frozen-source and release inference were compared
  at every block on one batch per dataset (8, 8 and 4 slices respectively).
- A bounded end-to-end CLI test passed: one synthetic training step, validation,
  best-checkpoint save/reload, PSNR/SSIM/LPIPS/DC evaluation, and export of all
  12 visual states plus 11 available auxiliary states. This did not resume or
  modify any research training run.

| Checkpoint | Final PSNR change, release − original | Final image RMSE | Original repeat RMSE |
|---|---:|---:|---:|
| AAPM | +0.000259 dB | 3.14e-5 | 3.12e-5 |
| MSD | +0.000008 dB | 2.71e-5 | 2.70e-5 |
| LDCT | +0.000027 dB | 1.82e-5 | 1.81e-5 |

RMSE is measured in the normalized image representation. Differences are at the
level of the original implementation's GPU reduction nondeterminism; **bitwise
identity is not claimed**. Direct + auxiliary loss values were also compared.
Complete block-level values and repeat controls are in the three JSON reports.
These are regression checks on small batches, not a new full-dataset benchmark
or a guarantee of identical retraining. No source checkpoint was modified.
