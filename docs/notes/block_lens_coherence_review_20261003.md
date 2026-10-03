# A constructive coherence control on the original grid

An explicit nested block lens gives a nontrivial route to small closure and
prototype defects, connected metrics, and vanishing **normalized** distance
distortion as both substrate size and macro resolution grow. The micro kernel
is the existing open-boundary `grid_2d(N, lazy=.5)`, with fixed staging one;
its transition probabilities are not slowed. This strengthens the restoration
options in the [earlier review](closure_and_refinement_review_20261003.md).

The lens is supplied using the grid's integer indices. It is a constructive
control, not the learned spectral lens and not evidence that the historical
canonical runs pass their audits. The conclusion has an L1 geometry, not a
Euclidean inner-product geometry. The paper remains unchanged, and its
non-smoothing/fractal interpretation is still unresolved.

## Construction and exact macro dynamics

Fix integers M >= 2 and b >= 2, set N = Mb, and label each microstate (r,c)
by `(floor(r/b), floor(c/b))`. Each macro prototype is uniform on its b-by-b
fiber. Adjacent lenses have block widths differing by an integer factor and
their quotient maps are literal block containment. C is the deterministic
one-hot coarse matrix and U has mass 1/b^2 at each state of its own fiber.
Hence U C = I with stochastic fiber-supported prototypes.

The substrate remains the original open grid: stay probability 1/2, with
the other half shared uniformly among its valid cardinal neighbors. Interior
nodes have degree four, side nodes degree three, and corners degree two.

For a macro edge crossing in the vertical direction, let h = 1 if its column
is the leftmost or rightmost macro column, and h = 0 otherwise. The analogous
indicator for a horizontal crossing tests its macro row. Exact counting gives

```
K_xy = (U P C)_xy = 1/(8b) + h/(24b^2).
```

There are b crossing-source nodes. When h = 0, each contributes
`(1/b^2)(1/8)`. When h = 1, one of them is a degree-three micro node, whose
contribution is `(1/b^2)(1/6)` instead. The difference is `1/(24b^2)`.
Because b >= 2, crossing sources are never micro corners and the opposite
source has the same degree. The macro matrix is therefore symmetric despite
the original micro matrix's boundary asymmetry. There are no other nonzero
off-diagonal transitions. The diagonal is one minus the outgoing sum.

An interior macro cell has escape 1/(2b). A noncorner boundary cell has escape
`3/(8b) + 1/(12b^2)`; a corner cell has
`1/(4b) + 1/(12b^2)`. For M = 2 all cells are corners. In every case

```
max_x (1-K_xx) <= 1/(2b).
```

The exact formula and its domain restrictions are checked by a separate
rational enumeration of every micro move in
[audit_block_lens_coherence.py](../../scripts/audit_block_lens_coherence.py).
The boundary correction cannot be replaced by a periodic-grid formula.
Tests also compare the sparse micro construction against the repository's
original dense `grid_2d`, including its nonuniform boundary degrees.

## Closure defect and prototype persistence

Put B = P C and E = B U. The existing mechanized fiber-lifting result gives
`s(x) = TV(U_x E, U_x) = 1-K_xx`. Thus every prototype defect is at most
1/(2b).

The new [Lean theorem](../../lean/GeoSBT/MarkovClosure.lean)
`closure_defect_le_macro_escape` proves a general implication: if B and U are
stochastic, U is supported on the deterministic fibers, and every macro
escape `1-(UB)_xx` is at most D, then **every microstate row** of `E^2-E` has
TV at most D. Its proof factors the actual closure, uses the fiber L1
isometry, and applies the extreme-row bound to K versus the identity on input
B_z. Consequently this block control has

```
delta(E) <= 1/(2b),    max_x s(x) <= 1/(2b).
```

This is a bound on the paper's full microstate-supremum idempotence defect,
not a bound on sampled inputs. The general implication is mechanized;
the explicit grid boundary count is a mathematical derivation with exact
finite Python checks. The grid family itself is not fully formalized in Lean.
Repeated closure has the already mechanized finite-horizon loss k/(2b),
without a conclusion uniform in arbitrary repetition length.

## Induced metric and normalized refinement distortion

Choose the declared probability floor `eta_b = 1/(16b)` and edge threshold
zero. The floor lies below every positive off-diagonal macro weight, so it
does not modify any edge cost. This scale choice is essential: a fixed positive
floor would eventually clip this growing family. Zero weights remain absent
edges. Weight averaging preserves K because it is symmetric.

The macro graph is the connected open M-by-M grid. Let c_b = log(8b) and
e_b = log(1+1/(3b)). Every off-diagonal edge cost lies between c_b-e_b and
c_b, both strictly positive. If L(x,y) is macro Manhattan distance, any path
has at least L edges and a monotone path has exactly L. Thus the actual
path metric d_b satisfies

```
(c_b-e_b) L(x,y) <= d_b(x,y) <= c_b L(x,y).
```

In particular, all distances are finite, distinct vertices are separated,
and the metric is nondegenerate. Define its specified physical-unit readout
`rho_b = d_b/(M c_b)`. Uniformly over **every macro pair**,

```
|rho_b(x,y) - L(x,y)/M| <= epsilon_b,
epsilon_b = 2 log(1+1/(3b))/log(8b).
```

The bound uses L <= 2(M-1). It tends to zero as b grows; there is no fitted
coefficient or favorable-pair restriction.

For a fine side M_f = s M_c, block width b_c = s b_f, and the containment map
r, the remainder on each coordinate is between zero and s-1. The reverse
triangle inequality for absolute values gives

```
|L_f(x,y)/M_f - L_c(r(x),r(y))/M_c| <= 2(s-1)/M_f.
```

Adding the two cost errors proves the normalized distortion bound

```
max_{x,y} |rho_f(x,y) - rho_c(r(x),r(y))|
  <= 2(s-1)/M_f + epsilon_bf + epsilon_bc.
```

In raw cost units the corresponding scale factor is
`alpha = M_f log(8b_f)/(M_c log(8b_c))`, and raw distortion is M_f log(8b_f)
times the normalized distortion. **Vanishing normalized distortion does not
imply vanishing or uniformly bounded raw distortion.** Adopting this result
requires explicitly fixing distance units; it cannot silently replace the
historical fitted raw-cost diagnostic.

## Equal-time route comparison on finest prototypes

For a nested fine/middle/coarse triple, compare

```
A = U_f P^2 C_c,
V = U_f P C_m U_m P C_c.
```

Both use two micro evolution steps, and their domain is the finest macro
simplex via U_f. Starting inside a fine block, two moves cannot leave it unless
the starting node lies within two lattice layers of its boundary. The uniform
mass of this strip is at most 8/b_f (for b_f < 4 use the bound one). Therefore
the direct route's escape from its initial coarse label is at most 8/b_f.

On the staged route, the first step changes the middle label only if it changes
the fine label, with probability at most 1/(2b_f). After middle completion,
escape from a middle label is at most 1/(2b_m), and changing the coarse label
requires such an escape. A union bound gives staged coarse escape at most
`1/(2b_f) + 1/(2b_m)`. TV to the initial coarse point mass equals escape;
the TV triangle inequality yields

```
max_x TV(A_x, V_x) <= 8.5/b_f + 0.5/b_m.
```

This intentionally loose bound tends to zero. It is **not** a route bound on
all micro point masses: arbitrarily concentrated boundary inputs do not have
the uniform-prototype boundary-strip estimate. The fixed domain and equal
time budget are part of the conclusion.

## Joint refinement family and finite checks

For each k >= 1 set t = 2^k, N = 4t^2 and choose three block widths
`4t, 2t, t`, with macro sides `t, 2t, 4t`. This is a finite three-level ladder
at each substrate size, not an infinitely refined ladder on one fixed grid.
Both the minimum block width and minimum macro side tend to infinity.
Every level has closure and prototype defects at most 1/(2t); connectivity
holds exactly. Both adjacent normalized distortions are at most
`1/t + epsilon_t + epsilon_(2t)`, and the prototype-input route mismatch is
at most 8.75/t. These bounds all tend to zero.

The limit readout is nontrivial: the normalized macro L1 diameter is
`2(M-1)/M`, tending to two, while cell mesh tends to zero. Small defects have
not been obtained by collapsing all distances or retaining a fixed number
of macro labels. Nevertheless the supplied block structure is substantive
recognition input, rather than something inferred without coordinates.

The [finite receipt](block_lens_review_20261003/evidence.json) audits complete
ladders at N = 16 and N = 64. Their worst measured prototype escapes are 0.25
and 0.125; prototype-input route mismatches are 0.0589193 and 0.03125. Every
finite macro-pair readout and adjacent distortion check satisfies its stated
analytic bound. The same JSON separately labels larger-family bounds as
**analytic formulas, not executions**. For example t = 256 yields a worst
defect bound 1/512 and normalized distortion bound approximately 0.004404,
but no 262144-by-262144 micro grid was allocated or simulated.

## What this restores and what remains open

This construction proves that meaningful small-defect coherence is attainable
under the actual finite closure, at fixed micro dynamics, with a declared
joint scale regime and normalization. It supplies a concrete positive control
for a future revision. It does not verify the learned spectral partitions,
the canonical tau = 5 runs, arbitrary-length repetition, microstate-domain
route commutation, or a Euclidean or fractal limit.

L1 readout is particularly relevant to claim coverage. Four points of an L1
square have side distances one and both diagonal distances two. They cannot
embed isometrically in any real Hilbert space: equality in the triangle
inequality along either two-edge diagonal path would force both intermediate
points to be the same midpoint, despite their mutual distance two. Thus
coherence alone does not imply Euclidean inner-product geometry, and failure
of Euclidean fit alone does not identify a fractal. The staged quadratic-cost
certificate remains a separate construction with a separate readout.

A later paper revision could use this positive control while presenting the
learned-lens results as measured finite audits, or pursue a comparable theorem
for the learned lenses. Either change needs an explicit author choice if it
materially changes the current main interpretation. The fractal non-smoothing
claim still needs a persistent geometric obstruction in a specified regime;
the existing finite fits do not provide it.

The derivation received a separate self-review of boundary degrees, floor
scaling, normalization, route domain, time budget, and limit quantifiers. This
is not independent external review. New tests distinguish the original open
grid from a periodic surrogate and compare the counted kernel to the actual
implementation. Fresh Lean build and axiom receipts accompany this control.
The current suite passes **57 tests**, including a control showing that clipping
edge probabilities with an overly large floor invalidates the stated readout
bound. The expanded axiom audit uses only the standard foundations and no
`sorryAx`.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_block_lens_coherence.py --output results/block_lens_review
.venv/bin/python3 -m pytest -q
cd lean
lake build
lake env lean Audit.lean
```
