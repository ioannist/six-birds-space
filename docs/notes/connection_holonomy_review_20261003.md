# Holonomy and noncommutation of covariant transport operators

Loop holonomy can be identified exactly with noncommutation of covariant
shifts on a common space of frame fields. This remains true when the individual
connection matrices lie in the abelian group SO(2). The construction repairs
the paper's operator interpretation by specifying the operators and their
domain. A separate geometric control on the unit sphere has exact holonomy
equal to spherical area, providing a genuine curved example and its flat
counterpart.

These are mathematical bridges and controlled connections. The canonical
grid/sphere Procrustes loop diagnostic still has its validated finite
separation, but no claim is made that its MDS charts converge to this spherical
connection or that its learned lenses pass the coherence gates. The distinction
is essential to retaining the valid protocol interpretation without identifying
every measured residue with geometric curvature.

## Two routes and common operators

Let X be a set with commuting translations u and v. At each x let A_x and B_x
be orthogonal transports from the u-neighbor and v-neighbor fibers back to x.
All fibers have explicit two-dimensional frames. A section s assigns a vector
to each framed fiber. Define operators on the **same section space**:

```
(S_u s)(x) = A_x s(u x),
(S_v s)(x) = B_x s(v x).
```

These are linear operators, including the shift in base point as well as the
fiber transport. The two coefficients for routes from uv x back to x are

```
L_x = A_x B_(u x),
M_x = B_x A_(v x),
H_x = L_x M_x^(-1).
```

Commutation of the translations makes the endpoint fibers identical. Direct
composition gives

```
([S_u,S_v]s)(x) = (L_x-M_x)s(uv x)
               = (H_x-I)M_x s(uv x).                (1)
```

Thus these two operators commute on every section if and only if H_x=I at
every x. The reverse implication uses a section with any desired vector at
uv x; no unrelated source or target fibers are equated. Orthogonality makes
M_x invertible, so a nonzero coefficient acts nontrivially on some vector.

For a finite translation grid with counting measure and the standard section
L2 norm, uv is a permutation. The norm of the commutator is therefore exactly

```
||[S_u,S_v]|| = max_x ||H_x-I||_operator.             (2)
```

Indeed the commutator is a pointwise block multiplication after a permutation.
The upper bound follows by summing squared block bounds; the lower bound is
attained by a section supported at a block maximizing its singular value.
Right multiplication by the orthogonal M_x does not change those values.
If H_x is a rotation with principal angle magnitude theta_x in [0,pi],

```
||H_x-I||_operator = 2 sin(theta_x/2),
||H_x-I||_Frobenius = 2 sqrt(2) sin(theta_x/2).
```

This follows by multiplying `(R-I)^T(R-I)=2I-R-R^T`
and using `R+R^T=2cos(theta)I`. Reflected loop transports retain (1) and (2),
but have no rotation angle; they must not be assigned one.

Although two SO(2) matrices commute when multiplied at one common fiber,
that fact does not force `A_x B_(u x)=B_x A_(v x)`. The coefficients are
evaluated at different points. Conversely, constant SO(2) coefficients give
commuting shifts and trivial square holonomy, even when each edge angle is
nonzero. Patchwise framing does not itself force curvature.

[Connection.lean](../../lean/GeoSBT/Connection.lean) mechanizes the operator
equivalence for group-valued frame fields with the faithful regular action:
`S_u(s)(x)=a(x)s(u x)`. It applies to arbitrary groups, including abelian ones,
with commuting translations as an explicit premise. The vector-section
linear formula and norm equality above are written proofs, not additional
Lean declarations.

For the original triangular loops, invertible transports also give the exact
route criterion `a b c^(-1)=1` if and only if `a b=c`. This criterion is
mechanized as `triangleHolonomy_eq_one_iff`. It describes agreement of two
routes with the same endpoints. It is distinct from commuting a pair of
SO(2) matrices and requires no square-grid reinterpretation of the triangles.

The same criterion gives a quantitative interpretation of the actual
repository triangle scores. In its row-coordinate convention, put
`H=R_xy R_yz R_zx` and `D=R_xy R_yz-R_zx^T`. Then
`D=(H-I)R_zx^T`, so `||D||_operator=2sin(theta/2)` on every proper loop.
Full-rank orthogonal Procrustes alignment is reciprocal, since reversing its
cross covariance transposes the unique polar factor. Therefore `R_zx^T` is
the direct x-to-z transport. The recorded nonzero angles genuinely measure
disagreement of these two routes; identifying that disagreement with the
Levi-Civita connection still requires the separate estimator bridge below.

At fixed charts, the reciprocal overlap transports form a discrete connection.
A nonidentity triangle holonomy is an obstruction to gauging that connection
to identity on every edge: if every transformed edge were identity, the
transformed loop would be identity, whereas conjugation cannot turn a
nonidentity loop into identity. Thus the measured scores detect non-flatness
of the estimated connection in this precise sense. Changing the chart family
or its truncation is more than a frame gauge change. Connection curvature
defined by these loop obstructions does not automatically equal intrinsic
curvature of the induced metric or the Levi-Civita curvature of a manifold.

## Frame changes

Change each frame independently by Q_x in O(2), including reflections. Then

```
A'_x = Q_x A_x Q_(u x)^(-1),
B'_x = Q_x B_x Q_(v x)^(-1),
s'(x) = Q_x s(x).
```

The shifted operators intertwine the frame transformation T_Q:
`S'_u T_Q=T_Q S_u` and `S'_v T_Q=T_Q S_v`. Consequently

```
H'_x = Q_x H_x Q_x^(-1),
[S'_u,S'_v] = T_Q [S_u,S_v] T_Q^(-1).
```

T_Q is orthogonal on the finite section space, so the operator norm is
invariant. Absolute rotation angles are invariant too, although a reflected
base frame reverses their signs. The intertwining and loop conjugation laws
are mechanized for the abstract group action, without assuming a common
global orientation. This matches the reason that local MDS frames require
full O(2) alignment rather than forcing an SO(2) alignment on each overlap.

## A genuine spherical connection

Use latitude phi and longitude lambda on a nonsingular unit-sphere patch:

```
r(phi,lambda) = (cos(phi)cos(lambda), cos(phi)sin(lambda), sin(phi)),
e_phi = (-sin(phi)cos(lambda), -sin(phi)sin(lambda), cos(phi)),
e_lambda = (-sin(lambda), cos(lambda), 0).
```

These are an orthonormal tangent frame. The intrinsic metric is
`dphi^2+cos(phi)^2 dlambda^2` and the area element is
`cos(phi) dphi dlambda` for |phi|<pi/2.

For a tangent field, define covariant differentiation by tangentially
projecting its ambient derivative. Product differentiation proves metric
compatibility, and projecting the ambient difference of derivatives proves
torsion freeness. This is the Levi-Civita connection of the displayed metric.
Direct differentiation of the frame gives

```
nabla_(partial_phi) e_phi = nabla_(partial_phi) e_lambda = 0,
nabla_(partial_lambda) e_phi = -sin(phi) e_lambda,
nabla_(partial_lambda) e_lambda = sin(phi) e_phi.
```

For a parallel vector `a e_phi+b e_lambda` along a latitude line,
`a'=-sin(phi)b`, `b'=sin(phi)a`, where prime is longitude differentiation.
Its component transport from lambda to lambda+w is therefore the rotation
`R(sin(phi)w)`. Along a meridian, components are unchanged.

Consider the rectangle with latitudes phi_0<phi_1 and longitude width w>0.
Use the inverse transports in the convention of (1): meridional A=I and
longitudinal B at latitude phi is `R(-sin(phi)w)`. Its square holonomy is

```
H = R(-w[sin(phi_1)-sin(phi_0)]),
area = integral_rectangle cos(phi) dphi dlambda
     = w[sin(phi_1)-sin(phi_0)].                     (3)
```

Thus its signed angle is minus the oriented area, modulo 2pi. When area<pi,
the principal angle magnitude is **exactly area**, and the common-space
operator norm is `2sin(area/2)`. No fitted chart, empirical curvature proxy
or imported probability kernel is needed for this geometric statement.

For an exact calibration choose `sin(phi_0)=1/4`, `sin(phi_1)=1/2`, and
w=1/4. Then area and principal angle magnitude are **1/16**, and the
commutator norm is **2sin(1/32)>0**. The strict sign follows from 0<1/32<pi;
floating matrices merely check the implementation of this established law.
The same transport construction on a flat Euclidean frame has A=B=I and
zero holonomy. Arbitrary independent frame gauges preserve both conclusions.

The rectangular loops above use meridians and latitude arcs, not geodesic
triangles. Their area law is derived directly from the connection and does
not assume a spherical-excess formula for a different loop type.

## Refinement scaling and the remaining estimator gap

For rectangles centered at a fixed latitude phi with meridional width h and
longitude width h,

```
area = 2h cos(phi) sin(h/2),
theta/area = 1,
||[S_u,S_v]||/[h^2 cos(phi)] -> 1.
```

The last limit uses sin(t)/t->1 twice. It recovers the unit sphere's curvature
magnitude at these shrinking loops. Raw angle and raw commutator norm both
tend to zero; persistence of curvature requires the declared area scaling,
or loops with fixed geometric area. A claim that a nonzero raw angle must
remain bounded away from zero on every shrinking curved loop would be false.

[audit_connection_holonomy.py](../../scripts/audit_connection_holonomy.py)
constructs the two operators on the same finite vector-section space for
four-node rectangles. It checks (1), (2), the sphere area law, independent
O(2) gauges, a constant flat connection and shrinking-loop scaling. The
geometric connection is supplied by the controlled sphere metric; it is not
learned from the canonical Markov kernel. The finite receipts are explicitly
floating checks of the written law. Tests exercise commuting edge rotations
with noncommuting shifts, pure-gauge flat fields and invalid orientation-angle
domains.

This supplies a legitimate mathematical interpretation of curvature as
noncommutation of transport protocols on a common domain. It preserves the
scope of the original finite loop diagnostic while strengthening its algebraic
and geometric foundations. It does not show that canonical MDS/Procrustes
transports approximate Levi-Civita transport: that would require a quantitative
chart and alignment consistency theorem for their actual induced distances.
The learned-lens coherence defects also remain a separate obstruction.

The argument received an adversarial self-review of translation endpoints,
faithfulness of the action, gauge domains, operator versus matrix commutation,
spherical connection signs, principal-angle branches and shrinking-area units.
It is not an independent mathematical review or a full Lean formalization of
spherical differential geometry.
