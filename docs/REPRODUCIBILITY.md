# Reproducibility and release scope

This is a focused release of the evaluated HerKry architecture. Training and
evaluation entry points have been consolidated; core parameter names, operators,
render-cache presets and model dimensions are preserved for strict loading of
historical state dictionaries. Unused standalone legacy networks, cluster
launchers, local paths and model-search scripts are excluded. Remaining low-level
compatibility branches are internal, not additional claimed paper methods.

## Concrete settings take precedence over shorthand descriptions

The submitted text summarizes several implementation details. To reproduce the
actual checkpoints, use these explicit settings in the released configs:

| Item | Evaluated implementation |
|---|---|
| Optimizer | Muon for matrix/conv parameters + auxiliary Adam; not all-AdamW |
| Initial group learning rates | Muon .002; Adam .0002, betas (.9,.95); weight decay .01 |
| Scheduler | Per-step cosine annealing to 1e-6, 50-epoch intended budget |
| EMA | Starts at epoch 4; decay .9995; evaluation uses EMA thereafter |
| Depths | Random 1–3/3–5/5–7/7–9 in training; fixed 2/4/6/8 at evaluation |
| AAPM/MSD batch | 8 |
| LDCT batch | 4, gradient norm clipped to 10 |
| Augmentation | AAPM/MSD continuous bilinear rotation plus flips; LDCT projection-space augmentation |
| Geometry | Literal native-operator settings in `docs/DATA.md` and JSON configs |
| LDCT noise | Post-rebin incremental Gaussian quantum/electronic approximation, not calibrated raw Poisson sampling |

The source geometry is not replaced by the alternative distances quoted in the
submitted appendix. Likewise, neither optimizer nor noise is changed to match
an abbreviated description: doing so would define a different experiment.

The compact trainer uses explicit manifests and slice-weighted validation PSNR,
does not evaluate test/OOD during training, and does not implement historical
ad-hoc resume branches. It saves online and evaluation model states separately.
It is a reproducible training recipe, not a guarantee of bitwise historical
retraining. GPU atomic reductions, cuDNN autotuning, data ordering and numerical
library versions can affect results. Use original inference batch sizes for
checkpoint comparisons; conditional-convolution routing can make batch context
relevant. Do not compare across different crops or preprocessing conventions.

## Frozen checkpoints behind the reported experiments

Weights are not distributed in this code-only release. Historical checkpoint
provenance is given to prevent selecting on the test set:

- AAPM: D configuration, selected epoch 45 after 50 training epochs.
  SHA-256 `6ca11137f23bbb9f0bd52c38b778ac18a4b1ac32c7174b1b42a2e246cb29fe5e`.
- LDCT: separately trained D configuration; selected epoch 50 by val72 at dose .375.
- MSD: separately trained D configuration; original run's epoch 38 selected by
  val32, rather than the lower-scoring independent continuation branch.

The evaluation CLI records checkpoint SHA-256, checkpoint epoch, manifest hash,
view count, dose, per-slice measurements and the slice-mean aggregate. Checkpoints
from another dataset must not be substituted even if their tensor shapes match.

## Baselines and interpretation

Third-party comparison methods are not repackaged here. Use their original
implementations and access terms; configurations/checkpoint selection must be
reported separately. The historical baseline training budgets were not uniformly
50 epochs: some AAPM assets had longer histories, and MSD LEARN++/ProCT/ReCoDiff/
CvGDiff were stopped after 19/48/38/48 completed epochs. The corresponding
selected epochs were 19/47/37/48. These are not matched-training-budget ablations.
Do not present intentionally early-epoch qualitative baselines as fully trained
best-checkpoint comparisons.

Intermediate exports are raw model states. `initial` denotes the auxiliary tensor
used by that dataset configuration; on AAPM/MSD it is the complete-HG k=0 state.
The LDCT legacy auxiliary path is retained and should not be silently described
as identical to the corrected AAPM/MSD supervision. Block 12 skips the solver.
No GT-dependent mean matching, error rescaling, patient selection or segmentation
postprocessing is part of the model's evaluation.

## Anonymous mirroring

Mirror only this release folder, not the parent research workspace. It contains
no dataset arrays, checkpoints, local experiment manifests, user/host paths or
training logs. Keep third-party credits and licenses intact. Check the anonymous
mirror itself for owner links, commit metadata, badges, release URLs and cached
files before sharing it; cleaning source files alone does not anonymize GitHub
history or the repository owner's identity.

Run `python audit_release.py` before upload. Its pattern scan is a basic safeguard,
not a substitute for reviewing the actual mirror and repository history.
