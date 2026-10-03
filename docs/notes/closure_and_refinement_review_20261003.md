# Closure proofs and finite refinement follow-up

The finite Markov construction is now mechanized, and the staged quadratic-cost
exhibit has an exact uniform certificate on declared finite windows. Additional
substrate sizes retain a grid/gasket ball-growth difference but do not establish
the paper's full-ladder coherence or non-smoothing interpretations. Those two
interpretations still require an author decision before any material revision.
The paper and historical experiment packs remain unchanged.

This continues [the initial mathematical review](mathematical_review_20261003.md).
Its earlier receipts remain historical; the new receipts are in
[refinement_review_20261003](refinement_review_20261003/evidence.json).

## Mechanized finite closure

[MarkovClosure.lean](../../lean/GeoSBT/MarkovClosure.lean) uses finite real
matrices with explicit nonnegativity and row normalization. Stochasticity of
`E = P^tau C U` and `K = U P^tau C` follows from stochasticity of P and U and
the deterministic coarse-map construction. Fiber support is needed for
`U C = I`, rather than being hidden inside the stochasticity claim. Stochastic
fiber-supported prototypes force all macro fibers to be nonempty.

The module proves that a bound on every extreme-row TV discrepancy is
equivalent to the same bound over the entire probability simplex. It also proves
the unconditional matrix identity

```
(B U)^2 - B U = (B (U B) - B) U,    B = P^tau C.
```

For arbitrary signed vectors and disjoint prototype supports, lifting gives
the exact L1 norm weighted by each prototype's L1 norm. For nonnegative
normalized fiber prototypes, this becomes an L1 isometry. Consequently the
actual prototype defect is `TV(U_x E, U_x) = 1 - K_xx`. Normalization and
support are proved applicability conditions, not extra conclusions.

If every row of `E^2 - E` has TV at most D and E is stochastic, then for every
distribution mu and natural k,

```
TV(mu E^(k+1), mu E) <= k D.
```

The proof uses stochastic powers, matrix associativity, the extreme-row bound,
and the TV triangle inequality. It does not smuggle a bound uniform in k.
The independently true TV upper bound of one can cap this estimate, but does
not make an unspecified "bounded defect" informative.

These declarations establish the mathematical validity of the diagnostics.
They do not establish small values on any ladder, an emergence theorem, a
manifold, or an identification of loop residue with curvature.

## Exact finite quadratic-cost certificate

[certify_quadratic_walk.py](../../scripts/certify_quadratic_walk.py) computes
integer transition counts for the canonical lazy walk: stay weight 4, four
cardinal weights 1, denominator 8. After tau steps its probability at z is
`count_tau(z) / 8^tau`. The recurrence checks total mass exactly. Centered
coordinates have zero mean, variance tau/4 on each axis and zero cross moment.
The comparison coefficient 2/tau is fixed in advance from that covariance;
it is never fitted to the observed costs. The finite certificate checks the
comparison directly and does not rely on a central-limit theorem.

For N = 512 and tau <= 128, N > 2 tau prevents wraparound aliases with other
reachable displacements. For every integer point with
`|x|, |y| <= floor(sqrt(tau))`, the checker verifies

```
|log(count_tau(0) / count_tau(x,y)) - 2 (x^2 + y^2)/tau| <= delta.
```

Every point in each window is present, reachable and has count at most the
origin count, so the centered cost is nonnegative. Verification reconstructs
the recurrence; a free-standing table of plausible probabilities is rejected.

The logarithmic inequalities are certified by exact rational exponential
enclosures. For rational a >= 0, write `t_j = a^j/j!` and
`S_n = sum_{j=0}^n t_j`. If a < n+2, all later term ratios are at most
`r = a/(n+2) < 1`, giving

```
S_n <= exp(a) <= S_n + t_(n+1)/(1-r).
```

Using n = 80, the checker establishes `exp(target-delta) <= count_0/count_z`
by the upper enclosure when target-delta is positive. Otherwise it follows
from `count_0/count_z >= 1`. It establishes
`count_0/count_z <= exp(target+delta)` using the lower enclosure. Monotonicity
of log gives the displayed uniform bound. Floating logarithms only propose
delta; an incorrect proposal fails the rational check.

The verified [certificate](refinement_review_20261003/quadratic_walk_certificate.json)
contains these bounds:

| tau | Window half-width | Points | Uniform centered-cost error delta |
|---:|---:|---:|---:|
| 4 | 2 | 25 | 363/500 |
| 8 | 2 | 25 | 117/1000 |
| 16 | 4 | 81 | 177/1000 |
| 32 | 5 | 121 | 11/200 |
| 64 | 8 | 289 | 11/250 |
| 128 | 11 | 529 | 1/50 |

The errors need not decrease at every stage; the windows change too.
For `L(z) = sqrt((C_tau(z)-C_tau(0))/2)` and
`r(z) = sqrt(x^2+y^2)/sqrt(tau)`, the squared-value discrepancy is at most
delta/2. The new theorem `quadratic_cost_readout_error` in
[Pythagoras.lean](../../lean/GeoSBT/Pythagoras.lean) proves that a nonnegative
quadratic-cost bound implies `|L-r| <= sqrt(delta/2)`. Thus at tau = 128 the
whole declared window has error at most **0.1**.

The finite recurrence and Taylor checker are Python with exact integer/rational
arithmetic; they are not verified by Lean. The conditional square-root bridge
is mechanized. This separation matters: there is no Lean theorem importing
this JSON as a verified premise. Nor is there a theorem that L is a pairwise
metric, that the shortest-path envelope preserves it, or that it converges
to Euclidean geometry. This strengthens the finite exhibit without claiming
those missing bridges.

## Larger substrates and the remaining interpretation

[audit_refinement_families.py](../../scripts/audit_refinement_families.py)
uses the existing spectral partitions, uniform prototypes and metric pipeline.
The grid ladder is always 4, 8, 16, 32, 64, 128 macro states, with six spectral
coordinates, laziness 0.5 and stages 1 and 5. Source hashes accompany the
[family results](refinement_review_20261003/evidence.json). No acceptance
tolerance was inferred from the results.

| Grid side | Finest maximum escape, tau = 1 | Finest maximum escape, tau = 5 |
|---:|---:|---:|
| 25 | 0.402778 | 0.782590 |
| 33 | 0.375000 | 0.782590 |
| 49 | 0.218750 | 0.531853 |

Larger substrate blocks can improve persistence at fixed macro count. This
supports studying a joint size/scale regime but does not rescue the canonical
full ladder. Adjacent fitted distortions divided by the fine diameter range
from about 0.19 to 0.65 across this family; finiteness alone is inadequate.

There is also an elementary obstruction to unrestricted persistence under
refinement. With singleton fibers, C = U = I and tau = 1, the canonical grid
has prototype defect `1-P_zz = 1/2`, independent of substrate size. A claim
of arbitrarily small prototype defects all the way to microstate resolution
cannot hold for this fixed dynamics and staging. Small idempotence alone
does not repair this: the two-state uniform kernel is exactly idempotent but
its singleton prototypes each have defect 1/2.

The gasket/grid size comparison uses tau = 5 and macro count
`min(256, max(16, n//3))`. The ball-growth slopes are:

| Grid states | Grid slope | Gasket states | Gasket slope |
|---:|---:|---:|---:|
| 49 | 1.7264 | 42 | 1.3156 |
| 121 | 1.6133 | 123 | 1.2762 |
| 361 | 1.7221 | 366 | 1.3765 |
| 1089 | 1.7727 | 1095 | 1.3849 |

This extends the finite scaling contrast beyond one substrate size. It does
not recover Hausdorff dimension or prove fractal persistence of the packaged
metric. The macro fraction changes when the count hits 256, so this is not a
single established scaling limit.

Local two-dimensional MDS reconstruction is inconclusive about non-smoothing.
At the largest sizes, the median relative errors with eight neighbors are
0.05438 for the grid and 0.05221 for the gasket; with 24 neighbors they are
0.05294 and 0.07006. Neighborhood radii relative to diameter shrink, but neither
comparison establishes a limiting obstruction. A candidate reconstruction's
error is an **upper** bound on best Euclidean fit error. It cannot be used as
a lower bound proving nonembedding. A genuine non-smoothing result needs a
specified refinement regime and a persistent obstruction to Euclidean local
approximation, not merely failure of one fitting algorithm.

Two concrete author options remain:

1. Keep the validated finite conclusions: constructed extended quotient
   metrics, measured audit values, corrected loop-residue separation,
   fixed-lens deformation, finite grid/gasket scaling contrast, and the new
   exact quadratic window. Treat full-ladder small-defect coherence and
   fractal non-smoothing as research questions in the later paper revision.
2. Pursue stronger conclusions with an explicit mathematical target: declare
   scale and time ranges, meaningful tolerances and a persistence or transport
   criterion, then demonstrate applicability. For fractal non-smoothing,
   specify a joint substrate/lens/neighborhood regime and prove or certify a
   geometric lower bound that persists in it. Current evidence cannot decide
   that target on the author's behalf.

Selecting favorable levels after inspecting the results or merely slowing
the dynamics would not establish the original unrestricted interpretation.
No material downgrade has been applied.

## Verification and implementation review

Factoring the closure defect avoids forming and squaring the dense microstate
closure matrix. The disjoint-support branch uses the exact weighted L1 identity,
including signed and unnormalized prototypes; overlapping supports retain the
full lifted difference. Prototype multiplication is reassociated exactly.
Tests compare these paths to the original dense expression, including inputs
that deliberately violate stochasticity. Floating roundoff remains possible.

The current Python suite passes **50 tests**. New adversarial tests reject
fabricated recurrence counts, incorrect coefficients, changed normalization,
missing domain points, fractional witness counts and false uniform bounds.
They also check exact walk moments and small-stage counts. A separate CLI
verification reconstructs and checks the complete saved certificate.

Fresh `lake build` passes; the expanded [axiom audit](refinement_review_20261003/lean_axioms.txt)
reports only `propext`, `Classical.choice` and `Quot.sound`, with no added axioms
or `sorryAx`. The mathematical statements, applicability conditions, factor
algebra, Taylor-tail argument and distinction between finite evidence and
limits received a self-review. This is not an independent external review.

Reproduce the receipts without writing to paper or historical packs:

```bash
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/certify_quadratic_walk.py --verify docs/notes/refinement_review_20261003/quadratic_walk_certificate.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_refinement_families.py --output results/refinement_review
cd lean
lake build
lake env lean Audit.lean
```

The initial baseline was committed before repairs. The review remains open at
the two material interpretation decisions above; successful tests and Lean
builds do not resolve them.
