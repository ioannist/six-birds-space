# A coherent curved Markov construction with derived transport

An explicit sequence of finite Markov systems has nested lenses satisfying
closure, prototype persistence, connectivity, normalized metric coherence,
and equal-time route coherence together. Their actual negative-log shortest-path
metrics converge to the unit sphere. The repaired metric-based transport
estimator recovers spherical parallel transport, including curvature on
shrinking loops, with a derived error rate.

This construction uses new reservoir dynamics and supplied dyadic cell lenses.
It retains staging five and a lazy component of one half. It does not validate
the original sphere kNN kernel, learned partitions, or historical MDS curvature
interpretation. Spherical geometry enters the contact law, as substrate geometry
enters the original synthetic controls. Neither automatic discovery nor a
sample-efficient trajectory experiment is claimed. The result supports
existence of a coherent curved regime; the original estimator correction remains.

The universal result is a written proof. Finite audits exercise the actual
staged microstate kernel. New Lean bridges cover probability costs and their
actual protocol infimum. These evidence types are distinct.

## The finite Markov system

Choose m distinct unit-sphere points `q_i`, an integer V>=4, and beta>0.
Each point has a cycle of V microstates `(i,j)`, with gateway `(i,0)`.
Inside every reservoir the symmetric base kernel B has holding probability
5/8 and probability 3/16 to each cycle neighbor. For i!=k, set

```text
Q_ik = 2^(-ceil(beta d_S(q_i,q_k))) / (8m).
```

Here `d_S` is unit-sphere geodesic distance. Replace each gateway diagonal
by `5/8-sum_k Q_ik`; other entries retain their base values. With
`A=Q-diag(Q 1)` and gateway projection G, the actual microstate kernel is

```text
P = I_m tensor B + A tensor G.
```

Every row sums to one. Contacts are symmetric with row sum at most 1/8, so
the gateway diagonal is at least 1/2. Thus `P=(I+W)/2` for stochastic W.
The actual holding probability need not equal 1/2; this differs explicitly
from walks having exactly that diagonal. Cycle moves keep a fixed positive
probability as V grows. Positive contacts make P connected, with uniform
stationary measure. There are mV microstates.

The finest lens labels whole reservoirs. Coarser lenses group reservoirs;
every prototype is uniform on its fiber. With deterministic coarse matrix C
and uniform lift U, the actual closure and macro kernel are `E=P^5 C U`
and `K=U P^5 C`.

## Persistence and closure from gateway density

A prototype on a union of g reservoirs has density `1/(gV)` at every state
in its fiber. Kill the walk on leaving that fiber. The killed matrix is
symmetric and substochastic, so its column sums are at most one. Its surviving
density therefore stays at most `1/(gV)` at every state and time. There are g
gateways, each with outgoing contact rate at most 1/8. First exit at any step
has probability at most `1/(8V)`. Thus for every stage tau,

```text
prototype escape <= tau/(8V).
```

At stage five, fiber support makes prototype TV defect exactly escape. The
existing finite Markov bridge bounds every microstate row of `E^2-E` by the
worst macro escape. Both worst prototype defect and full microstate closure
defect are consequently at most `5/(8V)`, at all levels simultaneously.
This is not arbitrary-horizon stability; the bound grows with tau.

## Deriving the macro likelihood geometry

Write `t=beta log(2)`. Weighted contact rows obey

```text
sum_k Q_ik exp(t d_S(q_i,q_k)) <= 1/8.
```

Including transitions within the same reservoir, full weighted row sums are
at most 9/8. By the triangle inequality, the weighted displacement moment
after l additional steps is at most `(9/8)^l` times its starting value.

Start from a uniform single-reservoir prototype and stop at the first contact
to another reservoir. Pre-contact gateway density is at most 1/V, by the
killed-matrix argument. At each possible first-contact step u, weighted flux
is at most `1/(8V)`. Summing and allowing any number of subsequent contacts gives

```text
weighted probability of contacted paths by time tau <= C_tau/V,
C_tau = (tau/8)(9/8)^(tau-1),    C_5 = 32805/32768.
```

Suppose coarser representatives `r_x` are within rho of every finest point
in their fibers. Average the estimate over the g source reservoirs. A path
ending in a different coarse fiber must have contacted another reservoir;
its displacement is at least `d_S(r_x,r_y)-2rho`. For x!=y,

```text
K_xy <= (C_5/V) exp(-t [d_S(r_x,r_y)-2rho]).
```

For a lower bound, start at one chosen gateway in the source fiber, contact
one chosen destination gateway at step one, and hold there four times.
Initial mass is `1/(g_x V)`. The contact is at least
`exp(-t d_S(q_i,q_k))/(16m)` and the holds contribute at least 1/16. Therefore

```text
K_xy >= exp(-t [d_S(r_x,r_y)+2rho]) / (256mVg_x).
```

Equal-size fibers make K symmetric. In general, averaging K with its transpose
preserves the brackets with `g_max` in the lower bound. The cube family below
has equal-size fibers at every level.

Use threshold zero and a probability floor below all positive macro transitions.
If `log(V/C_5)>=2t rho`, actual off-diagonal negative-log costs w satisfy

```text
t d_S(r_x,r_y) <= w_xy
                <= t d_S(r_x,r_y)+2t rho+log(256mVg_max).
```

Diagonal cost is zero, representing the empty protocol, and satisfies the same
brackets. Every finite path costs at least `t d_S` between its endpoints,
by the sphere triangle inequality. The direct edge bounds the infimum above.
Thus the implemented shortest-path metric D satisfies, for every pair,

```text
0 <= D_xy/t-d_S(r_x,r_y) <= epsilon,
epsilon = 2rho+log(256mVg_max)/t.
```

This is an actual path-metric estimate, not a direct-cost fit. The baseline
inequality absorbs coarsening displacement per edge; otherwise an error could
accumulate along a long path and invalidate the lower bound.

A lawful floor is

```text
eta = 2^(-ceil(beta pi)) / (256mVg_max).
```

Every off-diagonal K has a direct-contact contribution at least twice this
number, using actual finest distances at most pi. The floor is positive and
inactive. A fixed positive floor is not valid in the limit.

## Explicit nested spherical meshes

On each of six cube faces, subdivide `[-1,1]^2` into `2^ell` cells per coordinate.
Radially normalize face cell centers to the sphere. Replace the representative
of the cell immediately above and right of `(0,0)` by that face's axis point;
at level zero it is already the axis. Replacements stay inside closed cells.
All points are distinct; all six axes are present. Mark the positive three as
orthogonal landmarks. Parent indices use dyadic division in each coordinate,
so every cell has four children.

Face vector lengths are at least one, so normalization increases Euclidean
displacement by at most two. Spherical distance is at most pi/2 times chord
distance. A cell point is within face-coordinate distance
`2 sqrt(2) 2^-ell` of its representative, including the marked corner choice.
Covering radius and descendant-to-representative displacement are bounded by

```text
rho_ell = 2 pi sqrt(2) 2^-ell.
```

The bound can always be capped at pi. For each integer r>=5, take finest points
at level r+2 and lenses at levels r,r+1,r+2, with

```text
m_r = 6 * 4^(r+2),      V_r = 2^r,
beta_r = 2^(r/2),       t_r = log(2) 2^(r/2),
g_max = 16,             tau = 5.
```

The baseline is positive for every r>=5. Its displacement term decreases with r
while log-volume increases. At r=5, `2t_r rho_r=pi log(2)<pi`, whereas
`log V_r=5 log(2)>=10/3` and `log C_5<=37/32768`. The elementary bounds
`pi<22/7` and `log(2)>=2/3` give a strict positive margin. The log bound follows
by integrating the nonnegative derivative of `log x-2(x-1)/(x+1)` for x>=1.

At all three levels the common uniform error is

```text
epsilon_r = 4 pi sqrt(2) 2^-r
            +(16+log_2(6)+3r) 2^(-r/2) -> 0.
```

Closure and prototype defects are at most `5/(8*2^r)`. Macro metrics are finite
and connected. Axis antipodes and the lower metric bound give diameter at least
pi; the upper bound gives diameter at most `pi+epsilon_r`. Geometry does not
collapse.

Adjacent lens projections move both endpoints by at most rho_r. Their uniform
normalized distortion is at most `2epsilon_r+2rho_r`, tending to zero.
Consecutive systems r and r+1 have the same vanishing bound with their respective
errors and the parent cell map, using their own declared t units. This is a
joint substrate/lens limit, not infinite refinement of one finite micrograph.

The representative images are rho_r-nets. A correspondence matching every
sphere point to a nearby representative has distortion at most
`epsilon_r+2rho_r`. Thus the normalized finite spaces converge to the compact
geodesic unit sphere, with explicit limit compatibility and nondegeneracy.

## Equal-time routes including every microstate

On finest prototypes compare the actual maps

```text
direct = U_f P^10 C_c,
via    = U_f P^5 C_m U_m P^5 C_c.
```

Each is within TV `10/(8V)` of the initial coarse point mass. The direct route
uses ten-step prototype escape; the second uses two five-step bounds, with the
second starting at a middle prototype. Their discrepancy is at most `5/(2V)`,
uniformly over the finest simplex. Both routes use total microscopic time ten.

All-microstate inputs also have a vanishing bound. Let n=`2^(r+2)` be the finest
face side count. The count N(s) in any spherical ball of radius s satisfies

```text
N(s)/m <= 100(s+1/n)^2.
```

On each occupied face choose one of its points in the ball. Other such points
are within distance 2s. Their dominant coordinates have magnitude at least
`1/sqrt(3)`; inverse face-coordinate ratios vary by less than five times chord
distance, hence less than 10s. Grid spacing 2/n gives at most `10ns+2` values
in each coordinate interval. Add one for the marked replacement, sum over six
faces, and divide by `6n^2`. The bound follows from
`100s^2+40s/n+5/n^2<=100(s+1/n)^2`.

The exponential layer-cake identity then bounds every gateway contact row rate:

```text
a <= min(1/8, (25/2)[2/t^2+2/(nt)+1/n^2]) =: a_r.
```

Integrate `t exp(-ts) N(s)/m` from zero to infinity and multiply by 1/8.
Including the starting point only enlarges the bound. In this family
`a_r=O(2^-r)` uniformly. Starting at any microstate, a contact at each step
has probability at most a_r. Thus `P^10 C_c` is within `10a_r` of the initial
coarse point mass. For `P^5 C_m U_m P^5 C_c`, the first segment contributes
at most `5a_r`, and the second prototype segment at most `5/(8V)`.
Uniform all-microstate route discrepancy is therefore at most

```text
15a_r+5/(8V_r) -> 0.
```

The packing estimate is the additional argument needed for this input domain;
the prototype bound has not been silently promoted to all microstates.

## Transport derived from the actual Markov metric

Apply the landmark estimator in the [estimator review](holonomy_estimator_review_20261003.md)
to D/t. Clip values above pi to pi for reconstruction only. True distances lie
in `[0,pi]`, so clipping cannot increase uniform error epsilon_r. The path
metric used for coherence remains intact.

The marked positive axes are genuinely orthogonal and persist under parent
maps. The derived metric estimate, including landmark distances, supplies the
estimator applicability premise. Recovered unit vectors have error at most
`E_r=2 sqrt(3) epsilon_r` for large r. Full recognition residual is at most
`(1+2pi sqrt(3)) epsilon_r`: each point's angular error is at most pi/2 times
chord error, and the metric comparison adds epsilon_r. Radial and landmark
orthogonality residuals satisfy the same tolerance.

Set `h_r=2^(-r/8)`. On the positive-z finest face select the north landmark
and representatives with gnomonic coordinates `(u_r,s_r)` and `(s_r,u_r)`,
where `s_r=1/n` and u_r is the grid coordinate closest to h_r. Then
`|u_r-h_r|<=s_r`; eventually `s_r<=h_r/4` and `h_r<=1/4`. Their planar triangle
area is at least `h_r^2/4`. Spherical area density is at least one half on the
shrinking patch, giving true spherical triangle area at least `h_r^2/8`.

Frames formed by projecting the positive-x axis are uniformly nonsingular;
shorter arcs stay uniformly away from antipodal pairs. Since E_r tends to zero,
eventually both true and recovered geometry satisfy kappa=0.8 and gamma=1.8.
Specifically, `|x|<=5/16` and pairwise dot products are at least
`1/(1+13/128)>0.9`. Once `E_r<=1/40`, recovered `|x|<=27/80` and pairwise
dot products exceed 0.85. Their frame projection norms exceed 0.8 and arc
denominators exceed 1.8. These conditions follow from the constructed rates.
The earlier quantitative transport theorem gives

```text
|estimated_angle-true_area| <= 3 pi sqrt(3) C epsilon_r,
C = 2+16/0.8+2+4/1.8+2/1.8^2.
```

Dividing by area gives error at most `24 pi sqrt(3) C epsilon_r/h_r^2`.
It vanishes, because

```text
epsilon_r/h_r^2 = 4 pi sqrt(2) 2^(-3r/4)
                  +(16+log_2(6)+3r) 2^(-r/4) -> 0.
```

This proves shrinking-loop curvature readout 1 for the actual Markov-induced
metrics under the repaired estimator. The limiting connection has loop angle
equal to area by the written frame/Stokes derivation. Triangle two-route
disagreement has norm `2 sin(angle/2)` and the same area-normalized limit.
This is connection route dependence, not noncommutation of constant SO(2)
matrices.

Sphere recognition, marked landmarks and distance units are explicit here.
No unknown-manifold recognition theorem is implied. The original MDS estimator
retains its proved counterexamples and its finite discrete-connection meaning.

## Exact staged calculation and finite checks

Expand `(I tensor B+A tensor G)^tau` into words. Outer B factors disappear
against the uniform initial vector and constant terminal vector. For k contacts,
internal gaps l contribute `(B^l)_00`; their two outer gap lengths have
`tau-k-sum(l)+1` choices. Hence

```text
K_tau = I+(1/V) sum_{k=1}^tau c_k A^k.
```

For every V>=4, the exact stage-five coefficients are

```text
c_1=5, c_2=7345/1024, c_3=109/16, c_4=31/8, c_5=1.
```

The internal return moments are `1,5/8,59/128,385/1024`; a length-four cycle
cannot change a three-step return. The general word formula explicitly accounts
for short-cycle returns at stage ten. Tests compare both stages with the public
`macro_kernel` on dense microstate matrices at several volumes. Nested-lens
audits additionally compare against sparse microstate staging and check every
macro pair and every microstate row.

[The evidence pack](curved_reservoir_review_20261003/evidence.json) records
all simultaneous gates, three actual Markov-metric-to-transport controls,
and evaluations of universal bounds. Formula evaluations do not execute the
exponentially large universal systems. Finite transport controls at beta
64,128,256 give normalized angles about 1.077,1.083,1.033; that sequence alone
does not prove consistency. The derived bounds and parameter hierarchy do.

Double precision cannot represent contact/floor scales indefinitely. The
mathematical family uses explicit positive real probabilities; the finite
audit rejects weights outside its representable range. No uniform fixed-precision
or sample-efficient implementation is claimed.

## Mechanization and adversarial self review

`CoherentMetric.lean` proves exponential probability brackets pass to actual
floored likelihood costs with an inactive floor. It proves reference-distance
domination for every finite protocol and a bracket on their actual infimum.
Existing Markov and transport theorems cover closure from escape and loop-product
error. Gateway moments, cube geometry, asymptotic parameters and differential
geometry are written proofs outside Lean, not a fully formalized family.

The adversarial pass is a self-review. It checked multiple contacts, killed
column sums, per-edge coarsening errors, diagonal conventions, the decreasing
inactive floor, marked cells, equal-size fibers, and changing substrates.
It checked the packing argument for all-microstate routes, the error/area
hierarchy, recognition residuals, tangent/arc margins, and the distinction
between supplied couplings and recovered transport from the actual staged
metric. False controls remove geometric contacts or impose a fixed floor.
No independent reviewer was used.

Canonical learned-lens failures and exact nonembedding/prototype certificates
remain valid and unchanged. Recovery with original kNN dynamics and learned
lenses is not claimed. The construction changes microdynamics and observation
family, retaining likelihood cost, shortest-path readout, staging five and all
coherence gates. These changes suffice; necessity or minimality is not asserted.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_curved_reservoir_coherence.py --output results/curved_reservoir_review
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_curved_reservoir_coherence.py --verify docs/notes/curved_reservoir_review_20261003/evidence.json
.venv/bin/python3 -m pytest -q
cd lean
lake build
lake env lean Audit.lean
```
