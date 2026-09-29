# Data preparation and protocols

Clinical data and patient images are intentionally not redistributed. Keep the
original dataset license/access restrictions and preserve patient-level splits.
The loader accepts a manifest of arrays and rejects train/validation/test patient
overlap. It never creates a new random slice-level split.

`prepare_manifest.py` expands already normalized `[Z,H,W]` volumes into one row
per slice. Supply an assignments JSON with `train`/`val`/`test` lists of
`{"patient":"case_a","path":"images/case_a.npy"}` records, with paths relative
to that assignments file, then run:

```bash
python prepare_manifest.py --assignments data/assignments.json --output data/manifest.json
```

This helper does not normalize CT, rebin sinograms, choose patients, or filter
slices. For the AAPM historical subsampling rule, supply selected 2D files or
prepare the selected volume before expansion. Inspect the resulting counts.

## AAPM image-domain benchmark

The experiment used 136/30/30 chest-abdomen subjects (17,294/3,542/4,021 selected
slices) for training/validation/test and a separate 1,136-slice cranial OOD cohort.
Raw DICOM preprocessing uses rescale slope/intercept, masks outside the inscribed
circle to air, clips HU to `[-1000,400]`, then maps to `[0,1]` by
`(HU+1000)/1400`. Save an individual 2D `.npy`, or a `[Z,H,W]` patient volume with
one manifest row per selected axial slice. Do not include channel axes in files.
The loader pools to 256². Keep native normalized slices when reproducing the
original order of augmentation followed by pooling.

The historical selection retained every third slice for chest-case filenames
whose patient identifier began with `C`. Reconstructing exact benchmark numbers
requires the same selected slice list and order; the release does not infer this
rule from new filenames or provide restricted data. Excluded all-black images
must be specified in the evaluation manifest, not removed after observing scores.

Synthetic projections use the actual TorchRadon configuration from the evaluated
code: `source_distance=595*1.4285`,
`det_distance=(1085.6-595)*1.4285`, `det_spacing=1.2858*1.4285`,
256 bins, equiangular full-circle views, `clip_to_circle=True` and ramp FBP.
These are the literal operator inputs in the historical pixel convention; do
not replace them with a differently interpreted mm geometry. Train views are
16/24/32/48/64 with probabilities .30/.25/.20/.15/.10. No synthetic measurement
noise is added for AAPM or MSD training. Training augmentation uses random
horizontal/vertical flips and continuous ±180° bilinear rotation.

## MSD Task06 Lung

Use the 63 labelled cases only; the unlabelled official test cohort is not used.
The original experiment split patients 51/6/6, seed 20260719, giving
14,387/1,642/1,628 slices. CT preprocessing is HU clipping and area pooling to
256². Tumor masks were max-pooled to preserve small lesions; they are **not** used
by reconstruction training. Use the same AAPM acquisition configuration.
Do not mix the original NIfTI physical spacing with the simulated reconstruction
operator's pixel units.

Downstream segmentation is an independent, frozen nnU-Net v2 3D-fullres model
trained only on GT and labels (128³ patches, batch 2, 50 epochs). It was selected
using mean complete-patient validation Dice, not sampled patch pseudo-Dice.
This repository releases the reconstruction component; it does not substitute a
2D segmentation network or bundle the separately trained nnU-Net checkpoint.

## LDCT raw-projection benchmark

Inputs are **already rebinned** standard-dose post-log fan-beam projections,
not DICOM images and not raw photon counts. Each file is float32 `[1152,368]`;
manifest rows use `patient` and `path` without `slice` for separate sinogram files.
The patient split has 8/1/1 cases and 4,185/503/463 slices. Rebinning raw helical
data is not implemented in this release; use the source dataset's appropriate
rebinning procedure before using this adapter.

`herkry/real_data.py` preserves the stored-unit conversion, full-1152-view
ramp-FBP target, training angular shifts/flips, and fixed validation shift of
64 angular indices. The target is the original standard-dose FBP, not a noiseless
CT or a least-squares surrogate. Metrics use the central 256² crop of the 368²
output. The physical projection metric uses the full uncropped image grid.

Train views are 72/144/288 uniformly per batch. Relative dose is uniform in
[.25,.50] per sample; selection uses val72 at dose .375. Test noise is seeded
by the **input filename** with a fixed data seed; preserve filenames when
reproducing these conditions.

The implemented noise is a post-rebin **incremental Gaussian approximation**
to quantum/electronic variance, with effective `I0=1e6`, electronic std `10`:

```text
I = I0 * exp(-p)
var_increment = (1/dose - 1)/I + (1/dose² - 1)*(10/I)²
```

It retains existing acquisition noise. This is not a newly sampled raw
Poisson-count process, not a scanner-calibrated dose model, and does not reproduce
correlations introduced by helical rebinning. Dose fractions are relative to the
stored baseline, not a claim about absolute patient radiation dose.
