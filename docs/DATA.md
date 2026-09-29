# Data

Obtain datasets under their original access terms. Keep patients disjoint across
`train`, `val` and `test`. Paths below are relative to the manifest:

```json
{
  "train": [{"patient": "case_a", "path": "images/a.npy", "slice": 0}],
  "val": [{"patient": "case_b", "path": "images/b.npy", "slice": 0}],
  "test": [{"patient": "case_c", "path": "images/c.npy", "slice": 0}]
}
```

**AAPM / MSD.** Save float32 `[Z,H,W]` CT volumes (or `[H,W]` slices without the
`slice` field). Convert DICOM values to HU where applicable, clip to `[-1000,400]`
and normalize with `(HU+1000)/1400`. AAPM masks outside the inscribed circle to
air; the loader pools to 256². MSD uses pre-pooled 256² volumes. The experiments
used patient splits 136/30/30 for AAPM and 51/6/6 for the 63 labelled MSD cases.
Use the same selected slices when reproducing a benchmark. Training samples
16/24/32/48/64 views without added measurement noise.

**LDCT.** Save already-rebinned, standard-dose post-log sinograms as float32
`[1152,368]` arrays, with one manifest row per file and no `slice` field.
Raw helical rebinning is not included. The original split is 8/1/1 patients.
The adapter builds full-view FBP targets, samples 72/144/288 views and 25–50%
relative dose using an incremental Gaussian variance approximation. Evaluation
uses a central 256² crop of the 368² output. Preserve filenames for fixed noise
seeds; set `--dose` when evaluating (default 0.375).

Native fan-beam geometry is set in the configs and `herkry/core/utils.py`.
The auxiliary pre-CGLS render uses complete HG on AAPM/MSD; the LDCT config keeps
its original solver-initial auxiliary path. Validation PSNR selects the checkpoint.
