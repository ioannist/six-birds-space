# A coherent fractal construction from the original gasket dynamics

The original gasket dynamics admit a constructive family whose closure and
prototype defects vanish, whose normalized metrics are compatible under
refinement, and whose metric limit has non-Euclidean local structure and
Hausdorff dimension `log(3)/log(2)`. This restores the **existence of a coherent
fractal regime**, using an explicit combinatorial cell lens. Staging remains
**five**, as in the original canonical experiment, and micro transition
probabilities remain unchanged.

This is a new positive control. It does not make the canonical learned-lens
prototype defects small: those original finite runs still fail that criterion.
The recursive cover is supplied by the graph construction; no planar
coordinates are used, and no efficient discovery theorem for an arbitrary
unlabeled kernel is claimed. The probability floor and distance units below
are explicit scale choices. The universal graph and limit argument is a
written mathematical proof, supported by exact finite implementation checks.
Its scalar Hilbert-obstruction bridges are mechanized; the entire graph family,
measure construction and Hausdorff argument are not formalized in Lean.

## Finite construction and interface law

Let the micro graph be the repository's level L gasket, with lazy probability
one half. Its noncorner degrees are four and its three exterior corner degrees
are two. It has `(3^(L+1)+3)/2` vertices, by the recurrence `v_(L+1)=3v_L-3`.

Write L = s+m, with s >= 3. The recursive construction covers the graph by
3^m copies of the level s gasket. Each copy has

```
V_s = (3^(s+1)+3)/2
```

vertices, shares at most its three outer corners with other copies, and each
shared vertex belongs to exactly two copies. Assign a shared vertex to the
lexicographically first cell containing it. The resulting disjoint fibers
have sizes n_x in `[V_s-3,V_s]`; each prototype is uniform on its fiber.
Thus prototypes are stochastic, supported on nonempty fibers, and U C = I.

The ownership rule is nested. Grouping three last-generation child cells gives
their parent cover, and the smallest child address has the smallest parent
address. Hence the containment map is precisely `x -> floor(x/3)`, including
shared vertices. [audit_recursive_gasket_coherence.py](../../scripts/audit_recursive_gasket_coherence.py)
constructs these covers using the original graph gluing and verifies equality
with the original substrate, rather than substituting a different graph.

At stage one, each contact between cells has two crossing micro edges. All
their incident micro degrees are four; s >= 1 prevents another outer corner
from being adjacent to the junction. The aggregate transition flow across one
interface is therefore 1/4, before division by the source fiber size.

At stage five the same contact support remains. Distinct outer corners of a
cell are 2^s graph steps apart, at least eight, so a five-step walk cannot cross
two distinct interfaces. Any path contributing to an interface flow lies
within five steps of its shared corner. This neighborhood is independent of
s once s >= 3: it is the union of two stabilized corner neighborhoods, has
degree four throughout every contributing path, and reaches neither another
cell junction nor an exterior degree-two corner. Assigning the shared vertex
to the other side swaps two symmetric copies and leaves aggregate flow
unchanged. Local reversibility also makes the two directed aggregate flows
equal.

The finite integer recurrence on the level-four micro graph with three
level-three cells gives 23810 crossing counts, with denominator 8^5. Thus

```
kappa = 23810/32768 = 11905/16384,
K_xy = (U P^5 C)_xy = kappa/n_x  for adjacent cells x,y.
```

Other off-diagonal entries vanish and the diagonal is one minus outgoing
mass. The recurrence uses integer stay weight four and neighbor weight one
at degree-four nodes, and neighbor weight two at exterior degree-two nodes.
The exact calibration and every audited interface are checked independently
against those micro moves. Stability of the contributing radius-five
neighborhood extends that finite calculation to all s >= 3 and m >= 1.
No fitted effective transition rate is assumed.

## The cell graph and its recursive metric

Define G_0 to be one vertex. G_m consists of three disjoint copies of G_(m-1)
joined by three single edges, one for each pair of copies, at the corresponding
outer corners. This is the actual contact graph of the recursive cells. Its
vertices are cell addresses of length m over `{0,1,2}`, its maximum degree is
three, and its outer corner addresses are the three constant words. Let D_m
be its unit-edge graph metric.
All metric balls in this note are closed balls.

The following facts hold by induction:

1. Every recursive copy is isometrically embedded.
2. The diameter and every outer corner-pair distance are `2^m-1`.
3. Under last-digit deletion r, `|D_m(x,y)-2D_(m-1)(r(x),r(y))| <= 1`.

For the first two facts, suppose a child copy has diameter and corner distance
D = 2^(m-1)-1. A shortest path cannot leave it and return through the same
connecting edge, which would repeat an edge. An excursion returning through
its other connecting edge must pass through the two other children, with
length at least `2D+3`. Replacing it by an internal corner-to-corner path of
length D strictly shortens the path. Thus a child is isometric and a geodesic
visits it at most once. Across two children, the direct bridge route has
length at most `2D+1`. For outer corners it has exactly that length; the route
through the third child costs `3D+2` and cannot improve it. This proves the
diameter formula and, by iteration, isometry of every nested copy.

For the third fact, G_m is also obtained by replacing each vertex of G_(m-1)
by a triangle. Every external edge attaches to its own distinct port. A
coarse geodesic of length q lifts to q bridges and at most q+1 triangle edges,
so the fine distance is at most 2q+1. A fine geodesic cannot leave a triangle
and return: any external excursion is longer than an internal edge. Its
projection therefore has q' >= q bridges, and each intermediate triangle
requires a transition between distinct ports. Its length is at least
`q'+(q'-1) >= 2q-1`. If q = 0, the fine distance is zero or one. This proves
the stated uniform bound without assuming a geometric embedding.

The script verifies the contact graph from exact micro interface counts,
integer BFS distances, corner distances, diameter, distinct ports and
containment. These finite checks support the implementation; the induction
above is the universal argument.

## Closure, metric and route audits

Each cell has at most three neighbors, so its prototype defect is exactly
its escape mass and satisfies

```
s(x) = 1-K_xx <= 3 kappa/(V_s-3).
```

The mechanized theorem `closure_defect_le_macro_escape` applies to
B = P^5 C and the actual U. It gives the same bound on **every microstate row**
of `(BU)^2-BU`; hence `delta <= 3 kappa/(V_s-3)`. This is not merely a bound
on prototype inputs. Stochasticity of P^5, C and U, and fiber support, are
established applicability facts. Finite-horizon repetition incurs the already
proved multiplicative loss in the horizon; there is no arbitrary-horizon
stability claim.

Use symmetric weight averaging. For a contact edge,

```
W_xy = (kappa/2)(1/n_x+1/n_y),
kappa/V_s <= W_xy <= kappa/(V_s-3).
```

Choose `eta_s = kappa/(2V_s)` and threshold zero. The floor lies below all
positive edge weights and does not modify them; zero weights remain absent.
Let `c_s = log(V_s/kappa)` and `e_s = log(V_s/(V_s-3))`. Every edge cost lies
in `[c_s-e_s,c_s]`, with positive lower endpoint. Connectivity of G_m gives
an ordinary finite metric d_(s,m), and

```
(c_s-e_s) D_m <= d_(s,m) <= c_s D_m.
```

Normalize distance by `rho_(s,m) = d_(s,m)/(2^m c_s)`. Since the unit diameter
is less than 2^m, its uniform discrepancy from `D_m/2^m` is at most
`epsilon_s = e_s/c_s`, tending to zero as s grows. Adjacent levels on the
same micro graph have parameters `(s,m)` and `(s+1,m-1)`, giving

```
max |rho_(s,m) - rho_(s+1,m-1) composed with r|
  <= 2^-m + epsilon_s + epsilon_(s+1).
```

This is normalized distortion with declared units. Multiplying back to raw
negative-log cost units can destroy vanishing distortion. A fixed positive
probability floor would also eventually clip the growing family. Neither
issue is hidden in the result.

For a nested fine/middle/coarse triple compare
`A = U_f P^10 C_c` with `V = U_f P^5 C_m U_m P^5 C_c`. Both have equal total
staging ten and their domain is the finest prototype simplex. Here is a
uniform route bound independent of substrate size.

The stationary micro law is proportional to vertex degree. A fine fiber of
size n contains at most one global degree-two corner; all other degrees are
four. Its uniform prototype differs from the stationary conditional law by
TV at most `1/(2n-1)`. If there is no exterior corner, the two laws coincide.
The conditional stationary law is pointwise at most the global stationary
law divided by the fiber mass, and that domination persists under P. There
are at most six directed crossing edges, each contributing half divided by
the fiber's total degree. Thus each step's outgoing event probability is at
most `3/(4n-2)`. A union bound over ten steps, plus the initial-law TV bound
(valid also on the finite path-event space), gives direct escape at most
`32/(4n-2)`.

The staged route changes a middle label only by leaving the initial fine
label, with probability at most its five-step escape bound. After middle
completion, coarse escape is at most the middle prototype escape bound.
TV to the initial coarse point mass equals escape, so the triangle inequality
gives, with V_f = V_s and V_m = V_(s+1),

```
max_x TV(A_x,V_x)
  <= 32/(4(V_f-3)-2) + 3 kappa/(V_f-3) + 3 kappa/(V_m-3).
```

This tends to zero with cell size. It is a prototype-domain route bound, not
a microstate-domain bound. Arbitrarily concentrated boundary inputs are not
covered by the conditional stationary estimate for a uniform prototype.

Take three-level ladders `(s+2,m-2)`, `(s+1,m-1)`, `(s,m)` with s,m tending
to infinity and s,m >= 3. All these closure, persistence, connectivity and
normalized refinement requirements hold together; the route bound also
vanishes. Macro counts grow rather than staying fixed, and normalized
diameters tend to one rather than collapsing.

## The actual metric limit

Use infinite ternary addresses, with prefixes identified with G_m vertices.
Put `delta_m(a,b)=D_m(a|m,b|m)/2^m`. The refinement bound implies
`|delta_(m+1)-delta_m| <= 2^-(m+1)` uniformly. Therefore these readouts have
a uniform limit delta, with

```
|delta-delta_m| <= 2^-m.
```

Each delta_m is a pseudometric on the address space, so symmetry, positivity,
zero diagonal and the triangle inequality pass to the limit. Quotienting
zero-distance addresses gives an ordinary metric space X, of diameter one:
the three constant-address corners have distances tending to one. The
finite-prefix functions are continuous on the ternary product space, and
their uniform limit is continuous. Compactness can also be seen directly:
choose a subsequence with each successively longer prefix fixed, then use
the uniform tail bound to obtain convergence in delta. Thus X is compact.

Every recursive copy is isometric in G_m, so prefixing any fixed word p of
length j gives exactly

```
delta(pa,pb) = 2^-j delta(a,b).
```

This descends to an injective scaled copy of X in X. The 3^j prefix copies
cover X and have diameter 2^-j. No compactness, compatibility or quotient
identification is inferred from bounded costs alone.

For any sequence s_m >= 3 tending to infinity, the actual Markov metrics on
the corresponding cell addresses converge uniformly to this same delta:

```
|rho_(s_m,m)-delta| <= epsilon_(s_m)+2^-m -> 0.
```

The substrate levels are s_m+m. This is a specified joint substrate/lens
regime, not infinite refinement of one fixed finite substrate.

## Non-integer dimension from intrinsic ball growth

Let `d_f = log(3)/log(2)`. For G_m and 1 <= R < 2^m choose k so
`2^(k-1) <= R < 2^k`. The recursive G_(k-1) copy containing a vertex x
has diameter `2^(k-1)-1`, so its 3^(k-1) vertices lie in B_x(R).

Recursive-copy convexity rules out a geodesic leaving and revisiting a copy.
For positive-depth copies, internal corner degree two and global degree at
most three place bridges at distinct outer corners. A geodesic using q bridges
between copies has q-1 intermediate copies, each requiring a corner-to-corner
path of length at least `2^(k-1)-1`.
For depth zero the internal cost is zero and the same counting bound applies.
Thus the total length is at least `q 2^(k-1)-(2^(k-1)-1)`.
For R < 2^k, q is at most two. The quotient
contact graph has maximum degree three, so a radius-two neighborhood meets
at most `1+3+6=10` copies. Consequently

```
3^(k-1) <= |B_x(R)| <= 10*3^(k-1),
(1/3) R^d_f <= |B_x(R)| <= 10 R^d_f.
```

The bounds are uniform over vertices and unsaturated scales. They follow
from the intrinsic graph metric, not a drawn planar gasket.

Push the uniform ternary product probability through the zero-distance
quotient, giving a probability mu on X. At finite prefixes every vertex has
mass 3^-m. Apply the ball bounds to delta_m and use the uniform error 2^-m:
balls of radii r-2^-m and r+2^-m sandwich the limiting ball. Sending m to
infinity gives, for 0 < r < 1,

```
(1/3) r^d_f <= mu(B_x(r)) <= 10 r^d_f.
```

This also supplies a direct Hausdorff-dimension argument. For q > d_f the
3^m prefix copies have total q-power diameter
`3^m 2^(-mq) -> 0`, giving the upper bound. For any sufficiently fine cover,
a set of diameter ell is contained in a ball of radius ell, so its outer
mu mass is at most `10 ell^d_f`. Countable subadditivity and total mass one
give `sum ell^d_f >= 1/10`. Thus the d_f-dimensional Hausdorff content is
positive and the lower dimension bound follows. Hence

```
dim_H X = log(3)/log(2), which lies strictly between one and two.
```

This measure is compatible with the original uniform micro readout: fiber
sizes lie in `[V_s-3,V_s]`, so the packaged mass density relative to uniform
macro mass lies between `(V_s-3)/V_s` and `V_s/(V_s-3)`. It tends uniformly
to one as s grows. No correspondence between the historical fitted entropy
slopes and this exact dimension is assumed.

## Persistent local Hilbert obstruction

A fixed six-cycle in G_2 is isometric in every larger G_m. Four of its
vertices, ordered `[1,7,2,5]`, have diagonals three and other distances
`1,2,2,1`. The cost bracket therefore gives diagonal lower bounds
`3(c_s-e_s)` and side upper bounds `c_s,2c_s,2c_s,c_s`.

For V_s >= 42 and 0 < kappa <= 1,
`e_s <= 3/(V_s-3) <= 1/13` and `c_s >= 2`, hence `e_s <= c_s/26`.
The log estimate uses `log(1+t) <= t`; `exp(2)<9<42` gives the cost lower
bound. `gasket_staged_log_bracket` mechanizes these estimates. The mechanized
`gasket_quadrilateral_gap` shows a strict Hilbert gap at additive error
`c_s/4`: after division by c_s^2 its minimum gap at e_s/c_s = 1/26 is
`855/1352 > 0`. Thus the finite corner patch cannot have a Hilbert fit within
one twelfth of its certified radius bound 3c_s, even while its normalized
radius is at most `3/2^m` and closure defects vanish.

**That finite witness alone does not prove the limiting statement**: the
global approximation error 2^-m is comparable to the witness radius. To
avoid that invalid bridge, construct four fixed limiting points by their
addresses:

```
A = 0 111...       B = 20 111...
C = 01 222...      D = 1 222...
```

For prefix length m >= 3, write a = 2^(m-2). Their exact unit distances are

```
D_m(A,B)=D_m(C,D)=3a-1,
D_m(A,C)=a-1,
D_m(A,D)=D_m(B,C)=2a,
D_m(B,D)=a+1.
```

These follow from the isometric-copy corner distances and the two possible
routes through first-level contacts. For example, A to B goes from one outer
corner of copy 0 to its other corner (2a-1), across the bridge (one), and
between corners inside copy 20 (a-1); the alternative through copy 1 costs
3a+1. C to D similarly uses the other two-child route. A to C stays inside
copy 01 (a-1); B to D uses one internal child route and two bridges (a+1).
A to D costs a bridge plus a first-level corner route (2a); C to B uses two
child corner routes of length a-1 and two bridges (2a). These minima account
for all possible first-level routes, since a geodesic visits no child twice.
The script checks these formulas on prefixes m = 3 through 6.

Divide by 2^m and pass to the actual limit. The diagonal distances are 3/4
and the other four are `1/4,1/2,1/2,1/4`. At fitting error 1/16, the
quadrilateral gap is exactly 15/128, so no Hilbert fit within that error exists.
`gasket_limit_quadrilateral_gap` mechanizes this strict scalar comparison;
the graph distances and passage to this limit are the written proof above.
Prefixing a word of length j scales this obstruction by 2^-j. For **every**
point x in X, choose any address representing it; its length-j prefix copy
contains x and lies in B_x(2^-j). That copy contains the scaled four-point
obstruction. Consequently, for every point, arbitrarily small balls admit no
Hilbert approximation with uniform pair-distance error at most one sixteenth
of the ball radius. This is the stated non-smoothing sense; an exact curvature
tensor or an unrelated tangent-space formalism is not asserted.

## Finite receipts, trust boundary and remaining paper corrections

The [finite audit](recursive_gasket_review_20261003/evidence.json) checks three
complete nested ladders, with staging five and the original micro kernel:

| Micro level | Micro states | Finest macro states | Worst finest prototype defect | Prototype-input route mismatch |
|---:|---:|---:|---:|---:|
| 6 | 1095 | 27 | 0.0544968 | 0.00600428 |
| 7 | 3282 | 27 | 0.0180155 | 0.00199619 |
| 9 | 29526 | 81 | 0.00598866 | 0.000664797 |

Micro/interface counts, covers and unit distances use exact integers. Finite
metric and TV audits use floating arithmetic and check analytic bounds with
the declared tolerance; strict nonembedding gaps use rational log enclosures.
Tests compare stage-five macro probabilities directly with the original
public gasket implementation, check port and nesting laws, and verify uniform
finite ball-growth bounds. Source hashes, build, axiom and test receipts are
saved alongside the audit. The written universal proof received a separate
self-review of graph excursions, interface stabilization, floor scaling,
stationary domination, limit compatibility, measure bounds and the limiting
witness construction. No independent reviewer was used.

This supplies a true coherent fractal regime without changing the substrate
or staging. The lens strategy, normalization and floor rule are explicit new
choices. The canonical learned gasket still has worst prototype defect about
0.849; its old coherence interpretation cannot be justified by these new
cells. Likewise the existing dimension-proxy values must not be relabeled as
the exact Hausdorff dimension proved for this different regime. The later
paper revision can preserve the central existence claim by distinguishing
this construction from those original exploratory learned-lens runs.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_recursive_gasket_coherence.py --output results/recursive_gasket_review
.venv/bin/python3 -m pytest -q
cd lean
lake build
lake env lean Audit.lean
```
