# Exact metric holonomy obstructions and spherical transport repair

The existing local planar MDS and overlap Procrustes estimator is a connection
diagnostic, but it is not a universally consistent curvature estimator. Two
written counterexamples establish a structural failure even on exact spherical
geodesic distances. One gives zero holonomy; a second, with distinct charts and
uniformly nondegenerate overlaps, has a curvature-normalized limit strictly
larger than 1.2. Refinement and gauge correction do not remove these failures.

A separate metric reconstruction under an explicit unit-sphere model recovers
great-circle parallel transport. Its quantitative error bound identifies the
additional applicability obligation: uniform metric error must be smaller
than the shrinking loop area. This obligation has not been established for the
canonical learned Markov metrics. No paper or historical run pack was edited.

## The actual estimator and its flat theorem

The implementation uses classical MDS of each center and its metric nearest
neighbors, truncating to two positive Gram eigenvalues. It centers overlap
coordinates and takes the full orthogonal polar factor of their cross covariance.
Rank-deficient overlaps and cross covariances are excluded. The triangle readout
is the principal angle magnitude of `R01 R12 R20`, excluding reversing loops.

For an exact planar Euclidean metric, every full-rank local MDS chart is
`(q - chart_mean) Q_i`, with `Q_i` orthogonal. After overlap centering, the cross
covariance is `Q_i^T G_overlap Q_j`, where `G_overlap` is positive definite.
Its unique polar factor is `Q_i^T Q_j`. Their triangle product is exactly I.
This proves flatness for arbitrary neighborhoods satisfying the overlap rank
condition, including reflected gauges and different chart centroids.

The same argument supplies the first curved obstruction. If all three charts
contain the same point set, their distance submatrices differ only by a row
permutation. They therefore represent the same truncated Gram matrix, regardless
of whether that matrix came from a plane or a sphere. If its retained rank is
two and the overlap cross covariance has full rank, their transports are again
pure gauge. Their holonomy is exactly zero. A positive spherical area is lost.
This applies to `k = number_of_patch_points - 1` and to any neighborhood expansion
that makes the three patches identical. It is an obstruction to this universal
interpretation, not a claim that all choices of neighborhoods give zero.

## A distinct chart obstruction with exact coefficient

Let `q_i` be the following fixed integer points in a tangent plane:

```text
(0,0), (2,0), (0,2), (1,-4), (5,1), (-3,-3),
(6,-4), (4,5), (2,-3), (1,3), (-4,0), (1,1).
```

Put them on the unit sphere by the exponential map at the north pole:

```text
p_h(q) = (h sinc(h |q|) q_x, h sinc(h |q|) q_y, cos(h |q|)).
```

Here `sinc(t) = sin(t)/t`, continuously extended at zero. Use the production
estimator with `k=8`, no neighborhood expansion, and triangle centers 0,1,2.
The exact certificate records all three distinct neighborhoods, strict squared
distance gaps at their kNN boundaries, and positive covariance determinants
for every chart and every pairwise overlap. Those strict gaps imply the same
neighborhood membership for all sufficiently small positive h. Ties *within*
the selected neighborhood can reorder rows without changing its geometry.

The following calculation derives the limiting answer rather than fitting it.
For any two fixed plane points a,b, expansion of their spherical dot product and
the analytic squared-angle function at dot product 1 gives

```text
d_h(a,b)^2 = h^2 |a-b|^2 - h^4 det(a,b)^2 / 3 + O(h^6).
```

All remainders are uniform on this finite cloud. For each chart, let X be its
centered plane coordinates, G=`X^T X`, J its centering matrix, and

```text
W_ab = -det(a,b)^2/3,      E = -J W J/2,      P = X G^-1 X^T.
```

The centered spherical Gram matrix divided by `h^2` is
`X X^T + h^2 E + O(h^4)`. Its top-two spectral part has derivative

```text
F = P E + E P - P E P.
```

To justify this step, the two positive eigenvalues of `X X^T` are separated
from the remaining zero eigenvalues. The collective spectral projection varies
smoothly across this gap. In the range/kernel block decomposition, its
first-order retained matrix has blocks `E_PP`, `E_PQ`, `E_QP`, and zero in the
kernel/kernel block: these are exactly F. No gap between the two retained
eigenvalues is required. A smooth coordinate factor exists up to an orthogonal
gauge; one possible derivative is

```text
Y = E X G^-1 - X G^-1 (X^T E X) G^-1 / 2.
```

Indeed `X Y^T + Y X^T = F`, an identity checked with rational arithmetic in the
certificate. Thus we may compute the gauge-invariant loop using coordinates
`h(X + h^2 Y + O(h^4))`, even if a numerical eigensolver changes gauges.

For an edge i,j, overlap centering gives a common zeroth-order matrix A and
chart derivatives Yi,Yj. Its cross covariance divided by `h^2` is

```text
A^T A + h^2 M1 + O(h^4),     M1 = A^T Yj + Yi^T A.
```

The polar factor is `I + h^2 omega J2 + O(h^4)`, where
`J2 = [[0,-1],[1,0]]`. Differentiating the requirement that the rotated cross
covariance be symmetric gives

```text
omega = (M1_21 - M1_12) / trace(A^T A).
```

The three exact rational edge coefficients sum to

```text
-7884106465972959320549378723 / 3276386621095812942397680000.
```

The spherical triangle area is `2 h^2 + O(h^4)`. For an independent area
justification, its gnomonic image has vertices `(0,0)`, `(tan(2h),0)`, and
`(0,tan(2h))`. The spherical area density is `(1+x^2+y^2)^(-3/2)`, so its area is
between `L^2/[2(1+L^2)^(3/2)]` and `L^2/2`, with `L=tan(2h)`.
Both bounds divided by `2h^2` tend to one. Consequently the estimator's
principal angle divided by the actual area tends to the exact number

```text
7884106465972959320549378723 / 6552773242191625884795360000
= 1.2031709590085036...
> 6/5.
```

All overlaps remain full rank, with singular values divided by `h^2` bounded
away from zero. The loop is proper for small h by continuity from identity.
Its nonzero coefficient also makes the absolute principal angle expansion
valid. Neither a rank failure nor a reversing chart gauge causes this bias.

These examples can be included in globally densifying sphere samples. Keep the
12-point patch, remove background samples within spherical distance `20h` of
the north pole, and place an h-net on the remaining sphere. All patch points
lie within `8h`; for each of the three centers, every background point is
farther than every patch point for small h. The kNN charts are unchanged, while
the full sphere covering radius is O(h). Thus the failure is not avoided merely
by requiring the total sample to become dense.

## Quantitative stability of the original alignment

The estimator does have a quantitative connection-stability bound. If M,N are
nonsingular square cross covariances with polar factors U,V, then

```text
||U-V||_F <= 2 ||M-N||_F / (sigma_min(M)+sigma_min(N)).
```

Here is a direct proof. Write `M=U S`, `N=V T`, with S,T positive definite.
Polar optimality gives
`tr(U^T M)-tr(V^T M) >= sigma_min(M) ||U-V||_F^2/2`:
the symmetric part of `I-V^T U` is positive semidefinite, and its trace is
`||U-V||_F^2/2`. Add the corresponding inequality for N and apply the
Frobenius Cauchy-Schwarz inequality to `tr((U-V)^T(M-N))`.
Cancel the norm when nonzero; the zero case is immediate. Reflections are
allowed. The full-rank requirement is substantive.

Overlap coordinate errors eA,eB yield

```text
||M-M0||_F <= ||A0||_F eB + eA ||B0||_F + eA eB.
```

Products of three orthogonal transports have operator error at most the sum
of the three edge errors, by telescoping. This controls the estimated
connection relative to a specified reference connection; it does not identify
that reference with Levi-Civita transport. In shrinking charts, coordinate
curvature corrections are O(h^3), so these bounds generally allow an O(h^2)
edge error. That is the order of curvature times area. The exact counterexample
shows that treating this bound as a vanishing *normalized* error would be wrong.

## Metric reconstruction under an explicit spherical model

The new control uses a unit-sphere metric and three marked orthogonal landmarks
`a1,a2,a3` as recognition input. It sets

```text
r_i = (cos d(i,a1), cos d(i,a2), cos d(i,a3)),    p_i = r_i/|r_i|.
```

For exact spherical distances this recovers ambient coordinates in the
landmark basis, without reading the generating coordinates. The implementation
checks landmark orthogonality, radial residual and *all-pairs* reconstructed
metric residual, and rejects a candidate exceeding the stated tolerance.
A finite residual is not itself a proof that an unknown metric has this model.

For unit vectors p,q away from antipodal pairs, write K for the cross-product
matrix of `p cross q`. The minimal ambient rotation is

```text
R(p,q) = I + K + K^2/(1+p dot q).
```

It transports p to q and acts as parallel transport on their tangent spaces
along the shorter great-circle arc. To see this, use the arc's constant normal
n and its unit tangent t: R preserves n and sends t at the source to t at the
target. Along the arc, the ambient derivative of t is normal to the sphere,
while n is constant. Both therefore have zero covariant derivative.

With a fixed unit reference e, a tangent frame is formed by normalizing
`e-(e dot p)p` and then taking `p cross first_vector`. Row-coordinate transport
is `F_p^T R(p,q)^T F_q`. Independent O(2) frame changes conjugate every loop
at its base, so its angle magnitude and route-disagreement norm are invariant.
The reference patch must be nonsingular; antipodal arcs are excluded.

The loop angle equals enclosed area on a small geodesic triangle. This can be
derived directly, without treating the numerical agreement as a proof. For a
smooth oriented frame `e1,e2` with normal n, let `a=e2 dot de1`,
`b=n dot de1`, and `c=n dot de2`. Orthogonality gives
`de1=a e2+b n`, `de2=-a e1+c n`, and `dn=-b e1-c e2`.
Hence `da=c wedge b`, the negative of the oriented sphere area form
`b wedge c`. Parallel row coordinates rotate by the integral of a. Stokes'
theorem gives angle `-area` around the triangle. Below area pi the principal
angle magnitude is its area. This is a written differential-geometric proof,
not a Lean formalization.

## An error bound with explicit applicability conditions

Suppose all measured distances differ from those of the true unit sphere by
at most epsilon, including distances to the true orthogonal landmarks, and
`epsilon <= 1/(2 sqrt(3))`. Cosine is 1-Lipschitz, so the raw landmark coordinate
error is at most `sqrt(3) epsilon`. Normalization increases this bound by at most
a factor two. Thus every recovered point has error

```text
E = 2 sqrt(3) epsilon.
```

Require the true and reconstructed reference projections to have length at
least kappa>0, and both versions of every arc to satisfy `1+p dot q >= gamma>0`.
Projection perturbation is at most 2E. The first frame vector error is at most
`4E/kappa`, and the second at most `E+4E/kappa`. Their frame operator error is
therefore bounded by `E(1+8/kappa)`.

The cross-product matrix error is at most 2E; the squared matrix error is at
most 4E; and the arc denominator error is at most 2E. Substitution into the
rotation formula gives ambient operator error at most

```text
E (2 + 4/gamma + 2/gamma^2).
```

Consequently, each row-coordinate transport has operator error at most `E C`,
where

```text
C = 2 + 16/kappa + 2 + 4/gamma + 2/gamma^2.
```

The triangle operator error is at most `3 E C`. For proper 2D rotations,
circular angle distance is at most pi/2 times their operator distance, because
the latter is `2 sin(angle_distance/2)`. Principal angle *magnitudes* are
1-Lipschitz for circular angle distance. Thus

```text
|estimated_angle - true_area| <= (3 pi/2) E C.
```

With fixed positive margins, curvature consistency follows if
`epsilon / area -> 0`. Raw metric convergence alone does not imply this.
For loops of diameter h and area comparable to `h^2`, a sufficient condition
is `epsilon=o(h^2)`. The finite noisy control supplies epsilon=`h^4` from a
known metric perturbation; its applicability is declared, not inferred from a
good residual. The actual Markov-metric bound remains to be constructed.

## Verification and adversarial self review

[The evidence pack](holonomy_estimator_review_20261003/evidence.json) contains
the exact rational coefficient and ranks, source hashes, five refinement
controls, gauge and reciprocity checks, analytic area intervals and numerical
area quadrature at two resolutions. The original normalized answers approach
1.203170959; the metric-reconstructed spherical answers approach 1.
The finite answers and quadrature use floating arithmetic. The asymptotic
counterexample and quantitative repair bounds are written proofs; their finite
rational coefficient is recomputed exactly by the verifier.

Lean proves pure-gauge triangular cancellation, a normed-ring error bound for
the actual product of three approximate contraction transports, and division
of angle error by positive area. It does not formalize the MDS spectral
derivative, sphere recognition, Stokes calculation, or their numerical code.

The adversarial pass was a self-review. It checked the chart/overlap covariance
distinction, row transport signs, smooth retained spectral subspace versus
possibly discontinuous eigenvector gauges, strict kNN membership boundaries,
absolute-angle expansion, and dense-sample extension. It also checked that
the repair uses supplied sphere recognition rather than silently imputing it
to a Markov metric. The global coherent curved Markov construction and its
error rate at shrinking-loop scale remain open. Canonical learned-lens
persistence failures and their certificates remain intact.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_holonomy_estimator.py --output results/holonomy_estimator_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_holonomy_estimator.py --verify docs/notes/holonomy_estimator_review_20261003/evidence.json
.venv/bin/python3 -m pytest -q
cd lean
lake build
lake env lean Audit.lean
```
