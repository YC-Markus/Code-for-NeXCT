# Method and implementation correspondence

HerKry has four stages, each with three independent blocks. AAPM/MSD primitive
grids are 32², 64², 128² and 256², with square tensor-product Hermite degrees
3, 2, 1 and 0 (16, 9, 4 and 1 modes). All stages render onto the same 256²
image grid and use the same observed sinogram; no coarse sinogram is synthesized.
The LDCT adaptation uses 46², 92², 184² and 368² primitive grids and a 368²
rendering grid. The implementation has 8,521,068 parameters.

## One block

The backbone predicts shared geometry and eight coefficient channels: seven
nonlinear feature channels and one scalar physical branch. The neural decoder
and initial linear render update the **previous visual image**. In parallel,
HC-CGLS minimizes the observed sinogram residual with geometry fixed. Geometry
is detached inside the linear solve, not across the entire learned renderer.
The physical branch does not overwrite the visual image. Its ordered coefficient
trajectory and image/residual context condition subsequent blocks.

`KrylovDeltaGRU` is the paper's KT-GRU: it encodes coefficients, first differences
and second differences with a 16-channel recurrent state, conditioned by the
persistent feature state. The persistent grid feature has 48 channels. At stage
transitions the implementation performs the corresponding feature resizing and
projection; it is not a GRU directly running on full-resolution CT images.
Hermite-shell feature/context rendering is part of the evaluated implementation.

The final block (block 12) omits CGLS and trajectory encoding. All 12 blocks
contribute direct visual supervision; only applicable pre-CGLS states contribute
the auxiliary term.

## Loss and iteration schedule

For blocks `b=1..12`, let `s(b)` be their stage. The implemented loss is

```text
[sum_b w[s(b)] L1(visual_b, GT)
 + sum_{b=1..11} w[s(b)] rho[s(b)] L1(initial_b, GT)] / 5.625
```

Here `w=[1/8,1/4,1/2,1]`, `rho=[1,1/2,1/4,0]`, and
`5.625 = 3 * sum(w)`. Disabled auxiliary terms do not change the denominator.
The auxiliary image and the nonlinear direct image are distinct tensors.
AAPM/MSD use the complete-HG auxiliary render; the LDCT compatibility setting
preserves the auxiliary tensor from that historical source rather than silently
changing its training objective.

At training time, one depth per stage is sampled per batch, independently from
the inclusive ranges `[1,3]`, `[3,5]`, `[5,7]`, `[7,9]`. Evaluation uses
`[2,4,6,8]`; each active block in a stage uses that stage's depth. The recurrent
encoder supports the corresponding variable trajectory length. Shorter inference
depths are not a claim of universally improved robustness and are not the default.

## Interpretation boundaries

HC-CGLS is a fixed-geometry coefficient-space least-squares solve. The network
learns how to use its trajectory; it does not learn a causal stopping rule, and
the GRU does not gate or replace the numerical CGLS iterations in this release.
Continuous HG primitives are sampled by a discrete renderer and TorchRadon
projector; the code does not analytically integrate continuous HG functions along
each ray. The native acquisition operator remains unchanged across stages.
