# Mathematical review: current claims and evidence

Further strengthening after checkpoint `6af8d4c`:
[a uniform quadratic-limit theorem](quadratic_limit_review_20261003.md) now
supports the original staged accounting claim on growing central windows.
[Covariant transport operators](connection_holonomy_review_20261003.md) give
a precise noncommutativity interpretation and an exact spherical area control.
The complete Fourier and geometric calculations are written proofs; their
quantitative and algebraic bridges are mechanized. The original estimator's
universal curvature interpretation now has an
[exact spherical counterexample](holonomy_estimator_review_20261003.md):
its area-normalized limit can exceed 1.2 with full-rank distinct charts, or
vanish with identical charts. A separate spherical reconstruction supplies
genuine transport under explicit landmark/model assumptions and quantitative
metric-error bounds. A new
[curved reservoir construction](curved_reservoir_coherence_review_20261003.md)
now derives the applicability bounds for actual staged Markov metrics and
recovers a coherent unit-sphere limit, including shrinking-loop curvature.
It changes microdynamics and supplies cell lenses; it does not validate the
canonical learned sphere ladder.
An [exact prototype-optimization obstruction](prototype_obstruction_review_20261003.md)
also shows that reweighting supported prototypes cannot improve the canonical
finest grid/gasket worst persistence defects. A recovery at stage five must
change those lenses, as the constructive controls do.

The finite metric and closure constructions are valid after repair. There are
now constructive coherent grid and fractal regimes under the original micro
dynamics and staging five. The recursive gasket construction has a compact
metric limit of Hausdorff dimension `log(3)/log(2)`, with a persistent local
Hilbert obstruction. These are new cell-lens constructions; the canonical
learned-lens runs still have large prototype defects. They cannot be cited as
passing the same small-defect criterion.
The curved reservoir family retains stage five and simultaneously makes all
coherence defects vanish, including equal-time routes on every microstate.
Its uniform metric error vanishes faster than the selected loop areas, supplying
the missing quantitative bridge to spherical transport. This is a genuine
construction rather than another conditional limit criterion or finite plot.

The manuscript and historical experiment packs have not been edited. This
ledger gives the mathematical conclusions available for a later revision,
without assigning corrected interpretations to those historical artifacts.
The requested pre-review baseline is commit `4e2e9a0`.

## Claim coverage

| Subject | Strongest justified conclusion | Conditions and limits |
|---|---|---|
| Distance from likelihood | Nonnegative weighted protocol costs give an extended pseudometric; zero-distance separation gives an extended metric preserving distances. | Use `-log(max(p,eta))`, with probability and floor bounds. Thresholding retains absent edges as absent. Connectivity is required for finite distances. The paper's additive `p+eta` rule can give negative cycles. |
| Finite closure | The actual `E=P^tau C U` and `K=U P^tau C` are stochastic. Fiber prototypes give `UC=I`, signed L1 preservation, prototype defect equal to escape, and full microstate closure defect bounded by worst macro escape. | Stochasticity, nonempty fibers and fiber support are explicit. Repetition bounds grow with the finite horizon; there is no uniform arbitrary-horizon stability theorem. |
| Constructive grid coherence | Block lenses on the original open grid have vanishing closure, prototype, normalized distortion and prototype-input route defects in a joint substrate/cell regime, including stage five. | Blocks are supplied interfaces. Distance units and a decreasing floor are declared. The limit readout is an L1 square; coherence does not imply a Euclidean inner product. |
| Constructive fractal coherence | Recursive cell lenses on the original gasket at stage five have the same vanishing-defect properties. Their normalized metrics converge to a compact, self-similar metric space of dimension `log(3)/log(2)`. | Cell genealogy is recognition input. Cell size and macro depth both grow; this is not infinite refinement of a fixed finite substrate. The floor scales below positive edge weights. |
| Fractal non-smoothing | At every point of that gasket limit, balls of radii `2^-j` contain a four-point obstruction ruling out every real Hilbert fit with uniform distance error at most `2^-j/16`. | This is a specified approximation criterion. No curvature tensor, anomalous-diffusion theorem or different tangent-space definition is inferred. |
| Canonical learned lenses | Valid finite metrics and recorded audit defects; the canonical gasket also has an exact local nonembedding witness. The grid/gasket worst persistence defects are exact minima over every supported stochastic prototype choice at their fixed partition and stage. | Worst prototype defects are about 0.783 (grid), 0.985 (sphere), and 0.849 (gasket). Full-ladder small-defect coherence fails. The sphere value is for uniform prototypes only. Finite nonembedding also occurs on the grid, so it alone is not a fractal classifier. |
| Grid/sphere loop residue | Gauge-correct finite O(2) Procrustes loop residue separates the two canonical cases: median sphere/grid ratio about 25.34. Covariant shifts on a common section space commute exactly when their square holonomy is trivial. A supplied unit-sphere connection has an exact area law. | Individual SO(2) matrices commute, while covariant shifts can fail to commute. The estimator is not proved to recover that geometric connection, and no coherent curved learned-lens limit is established. Shrinking-loop curvature uses area normalization. |
| Exact metric estimator obstruction | On exact spherical metrics the original local MDS estimator can have area-normalized holonomy limit 1.203170959... with full-rank distinct neighborhoods; identical neighborhoods give exactly zero. Both patches extend to globally dense sphere samples. | These are written counterexamples with an exact rational leading coefficient. They refute universal curvature consistency, not every possible balanced sampling regime. |
| Spherical transport repair | Under an explicit unit-sphere metric model, three marked orthogonal landmarks recover coordinates and great-circle parallel transport. Uniform metric error epsilon gives a quantified angle error with fixed frame and arc conditioning margins. | Area-normalized consistency requires epsilon/area tending to zero. Recognition residuals do not supply the true-model premise; applicability to actual learned Markov metrics remains unproved. This is a separate estimator, not a new interpretation of historical MDS results. |
| Constructive curved coherence | Explicit spherical reservoir kernels at stage five and nested cube-cell lenses have vanishing closure, persistence, normalized distortion and equal-time route defects, including all microstate inputs. Their actual normalized likelihood path metrics converge to the nondegenerate geodesic unit sphere. | This changes the original kNN dynamics and supplies the lenses. Lazy component one half is retained; probabilities, metric units and decreasing inactive floors are explicit. Automatic discovery, fixed-precision execution of the full limit family and sample efficiency are not claimed. |
| Markov transport applicability | The reservoir family supplies uniform metric error O(r 2^(-r/2)). On loops of scale 2^(-r/8), error divided by area tends to zero. The repaired estimator therefore recovers spherical transport and area-normalized curvature 1 from actual staged Markov metrics. | Three marked orthogonal landmarks and the verified sphere model are explicit recognition content. The universal argument is a written proof with substantive Lean cost/path/transport bridges; the whole family is not fully formalized. Canonical learned metrics are not reinterpreted. |
| Staged quadratic cost | The original lattice walk has a derived uniform local probability estimate, a quadratic negative-log limit with coefficient 2/n and offset log(pi n/2), vanishing axis residual and a Euclidean square-root readout limit. The exact stage-128 certificate remains sharper on its 529 points. | Fixed central windows in diffusion units, growing torus size, and unclipped probabilities are explicit. The Fourier proof is outside Lean; the log/residual bridges are mechanized. This readout remains separate from the macro shortest-path metric. |
| Constraints and controls | Directional gating changes the kernel and metric even with the lens held fixed. L1 axis separability is exactly zero, while its squared-distance Pythagorean residual is nonzero. | Anisotropy can retain a positive quadratic form and weighted Pythagoras. It need not destroy inner-product geometry. Saturated and unsupported diagnostics fail explicitly. |

## Constructive replacements

The [curved construction](curved_reservoir_coherence_review_20261003.md) derives
its macro probability brackets from first-contact gateway flux and exponential
moments, allowing multiple contacts. A per-edge baseline prevents coarsening
errors accumulating along paths. Cube meshes give limit compatibility and a
packing bound for every-microstate route coherence. Derived metric error and
explicit loop scales then justify the repaired spherical transport estimator.
The contact law encodes the supplied spherical substrate; this supports the
existence claim rather than automatic geometry discovery.

The [block-lens proof](block_lens_coherence_review_20261003.md) derives the
original-stage grid bounds directly from the open-grid transition law. Its
stage-five extension uses stationary domination to bound prototype escape by
`5/b`, and bounds all normalized metric pairs by their L1 readout. The slow
logarithmic metric error tends to zero; the finite smallest-block checks are
not presented as tight small-error certificates.

The [recursive-gasket proof](recursive_gasket_coherence_review_20261003.md)
keeps the original lazy walk and stage five. Exact interface flux is
`11905/16384`. With cell volume V, macro escape and full microstate closure
defect are at most `3*(11905/16384)/(V-3)`. Recursive graph distances have an
explicit refinement recurrence, supplying compactness and a compatible limit.
Uniform intrinsic ball growth supplies the dimension argument. A separate
fixed-address witness supplies the limiting Hilbert gap; a shrinking finite
witness is not improperly transferred through a comparable approximation
error.

The [learned-lens certificate](local_nonembedding_review_20261003.md) reconstructs
the ideal staged macro probabilities from the recorded labels, checks exact
multiplicative shortest paths, and encloses logarithms rationally. It excludes
fits of the canonical gasket patch within 9.1% of its radius, in any real
inner-product dimension. The corresponding grid exclusion is 5.6%. These
certificates verify the metric conditional on the recorded partition; they do
not certify an eigensolver or infer an infinite regime from ten finite cases.

## Mechanization and verification boundary

The [quadratic-limit audit](quadratic_limit_review_20261003/evidence.json) and
[connection audit](connection_holonomy_review_20261003/evidence.json) extend
the preceding checkpoint, along with the exact prototype obstruction. The
preceding checkpoint passed **88 tests** with **53** transitive axiom checks.
The preceding [estimator audit](holonomy_estimator_review_20261003/evidence.json)
adds exact refinement obstructions, metric-only spherical reconstruction and
quantitative transport bounds. That checkpoint passed **96 tests** and **56**
transitive axiom checks. Its receipts are in
[holonomy_estimator_review_20261003](holonomy_estimator_review_20261003/python_tests.txt).
The current [curved audit](curved_reservoir_review_20261003/evidence.json)
checks simultaneous gates, actual Markov-to-transport controls and exact staged
polynomial coefficients. **112 tests** pass; the fresh Lean build has **59**
transitive axiom checks using only standard foundations. Current receipts are in
[curved_reservoir_review_20261003](curved_reservoir_review_20261003/python_tests.txt).
The preceding full-suite, Lean and exact recheck receipts are in
[quadratic_limit_review_20261003](quadratic_limit_review_20261003/python_tests.txt).

[The Lean coverage description](../../lean/README.md) identifies the represented
statements and the remaining external applicability arguments. Weighted metric,
finite Markov and quantitative Hilbert bridges are proved in Lean. The entire
recursive graph family, measure construction and Hausdorff-dimension proof are
written proofs outside Lean. Numerical implementations are not formally
verified by the scalar bridges.

The validation receipts at the preceding `6af8d4c` checkpoint are:

- [77 passing Python tests](recursive_gasket_review_20261003/python_tests.txt),
  including false-target controls, fractional-input rejection, direct comparison
  with the original staged kernels, exact-certificate tampering and recursive
  graph-law checks.
- [Fresh Lean build](recursive_gasket_review_20261003/lean_build.txt) and
  [transitive axiom audit](recursive_gasket_review_20261003/lean_axioms.txt), using
  only the standard foundations, with no added axioms or `sorryAx`.
- [Independent certificate recheck](local_nonembedding_review_20261003/verification.txt)
  reconstructing all ten finite witnesses. This is a fresh computation with the
  verifier, not an independent reviewer.
- Source-stamped [gasket audit](recursive_gasket_review_20261003/evidence.json),
  [stage-five grid audit](staged_block_lens_review_20261003/evidence.json) and
  [learned-lens certificates](local_nonembedding_review_20261003/certificate.json).
  Recorded Python source hashes matched the sources at that checkpoint.

The [initial repair checkpoint](mathematical_review_20261003.md) records the
twenty corrected canonical and sweep configurations. The
[closure/quadratic follow-up](closure_and_refinement_review_20261003.md) records
the exact quadratic certificates and broader learned-lens tests. Their older
test counts and receipts describe those checkpoints, not this final state.

The preceding adversarial pass was a self-review. It checked interface stabilization,
graph excursions and distinct ports, route input domains, units and floor
scaling, stationary domination, uniform limit compatibility, ball-measure
bounds, and the limiting witness rather than relying solely on passing finite
tests. No independent mathematical reviewer was used.
The preceding estimator self-review additionally checked spectral derivatives,
overlap conditioning, area units, dense-sample extension and the explicit model
assumptions of the spherical repair, as detailed in its companion note.
The curved-family self-review checks multiple contacts, per-edge path bounds,
all-microstate route domains, recognition margins and shrinking-loop rates.
The [goal audit](curved_goal_completion_audit_20261003.md) records requirement
coverage and the explicit original-claim corrections.

## Reproduction

Install the optional audit dependencies with `pip install -e '.[audit]'`.
The scripts write new results rather than replacing historical packs:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_block_lens_stage_five.py --output results/staged_block_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_recursive_gasket_coherence.py --output results/recursive_gasket_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/certify_local_nonembedding.py --verify docs/notes/local_nonembedding_review_20261003/certificate.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_quadratic_limit.py --output results/quadratic_limit_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_connection_holonomy.py --output results/connection_holonomy_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_prototype_obstruction.py --verify docs/notes/prototype_obstruction_review_20261003/evidence.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_holonomy_estimator.py --output results/holonomy_estimator_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_holonomy_estimator.py --verify docs/notes/holonomy_estimator_review_20261003/evidence.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_curved_reservoir_coherence.py --output results/curved_reservoir_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_curved_reservoir_coherence.py --verify docs/notes/curved_reservoir_review_20261003/evidence.json
.venv/bin/python3 -m pytest -q
cd lean
lake build
lake env lean Audit.lean
```

The later manuscript revision must preserve the distinction between the proved
cell-lens constructions, validated finite diagnostics and false historical
interpretations. A coherent curved limit with justified transport is now
constructed under explicit modified dynamics and supplied lenses. Automatic
lens discovery, the original learned-sphere ladder's coherence, and a Euclidean
macro path-metric limit remain unproved. Those claims were not substituted for
the curved construction or inferred from finite diagnostics. The paper remains
unchanged; the original estimator retains its discrete-connection interpretation.
