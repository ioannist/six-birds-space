# Uniform quadratic accounting limit for the original lazy walk

The paper's exact isotropic walk has a uniform quadratic negative-log cost
limit on every fixed central window in diffusion units. This strengthens the
finite certificates to a theorem about growing stages and growing windows.
The argument derives a local probability estimate from the walk's Fourier
formula, controls relative error before taking logarithms, and supplies the
nonnegative centered cost needed for the square-root readout. A matched
correlated-step control has a nonvanishing coordinate-axis residual, with exact
finite certificates confirming failure at the original staging values.

The micro dynamics for the positive result are unchanged: stay probability
one half and each cardinal move probability one eighth. The torus must grow
with stage to avoid aliasing. This is a written Fourier proof, with the
logarithmic and residual bridges mechanized in Lean. It does not establish a
Euclidean macro shortest-path metric or repair the canonical learned-lens
coherence defects.

## Statement of the isotropic theorem

Let p_n(z) be the n-step probability of the above walk on the integer lattice,
and let |z| denote Euclidean displacement. Define

```
g_n(z) = 2/(pi n) exp(-2|z|^2/n),
a = 11/96,
B_n = [7/(768 pi a^3)] n/(n-1)^3
      + [pi/(2n)] exp(-n/(2 pi^2)) + [2/(pi n)] exp(-n/8).
```

For every integer n >= 2 and **every** z in the lattice,

```
|p_n(z)-g_n(z)| <= B_n.                         (1)
```

For any fixed R >= 0 put

```
E_R(n) = exp(2R^2) [ (4032/1331) n^2/(n-1)^3
                    + (pi^2/4) exp(-n/(2pi^2)) + exp(-n/8) ].
```

This equals `B_n / [2/(pi n) exp(-2R^2)]` and tends to zero at rate O_R(1/n).
Whenever E_R(n) < 1, every |z| <= R sqrt(n) has positive probability. Its
centered cost `F_n(z)=log(p_n(0)/p_n(z))` is nonnegative and obeys

```
|F_n(z)-2|z|^2/n| <= d_R(n),
d_R(n) = 2 E_R(n)/(1-E_R(n)) -> 0.              (2)
```

Thus the axis residual tends uniformly to zero on the same window:

```
|F_n(x,y)-F_n(x,0)-F_n(0,y)| <= 3 d_R(n).       (3)
```

The scale factor `2/n` is derived from the actual transition law, rather than
fitted. These are uniform statements on growing central domains, not RMS
statements or a claim about every reachable far-tail displacement.

## Fourier probability estimate

The characteristic function of one step is real and nonnegative:

```
phi(t) = 1/2 + (cos(t1)+cos(t2))/4,
0 <= phi <= 1,  t in [-pi,pi]^2.
```

This follows by multiplying the finite one-step probabilities by their
characters. Independence gives phi^n. Expanding the finite polynomial and
integrating each integer character gives the exact inversion formula

```
p_n(z) = (1/(2pi)^2) integral_[−pi,pi]^2 phi(t)^n exp(−i t.z) dt.
```

Write r=|t|, `q=1-phi` and `h=r^2/8`. Elementary sine and Taylor bounds give

```
r^2/(2pi^2) <= q <= h,
0 <= h-q <= r^4/96,
q >= (11/96)r^2  when r <= 1.                  (4)
```

For completeness, `1-cos u=2 sin^2(u/2)` and concavity of sine on
[0,pi/2] give `1-cos u >= 2u^2/pi^2` for |u| <= pi. Integrating the second
derivative of cosine, and then its fourth derivative bound, gives
`u^2/2-u^4/24 <= 1-cos u <= u^2/2`. Summing the two coordinates proves
(4), using `t1^4+t2^4 <= r^4`.

For 0 <= q <= 1,

```
0 <= exp(-q)-(1-q) <= q^2/2.
```

The lower bound follows from the tangent line of the exponential. For the
upper bound use its integral remainder with second derivative at most one
on the nonpositive half-line. Telescoping the difference of n-th powers gives

```
0 <= exp(-nq)-(1-q)^n <= (nq^2/2) exp(-(n-1)q).
```

Also `0 <= exp(-nq)-exp(-nh) <= n(h-q)exp(-nq)` by integrating the
derivative over [q,h]. On the unit disk these inequalities and (4) imply

```
|phi(t)^n-exp(-nh)| <= (7n/384) r^4 exp(-(n-1)a r^2).
```

Integrating over the disk and enlarging to the whole plane yields its error
contribution

```
[1/(4pi^2)] (7n/384) integral_R^2 r^4 exp(-(n-1)a r^2) dt
  = [7/(768pi a^3)] n/(n-1)^3.
```

The radial integral is `2pi/alpha^3` for `alpha=(n-1)a`, by substituting
u=r^2 and integrating u^2 exp(-alpha u).

Outside the unit disk, the original Fourier integrand is bounded by
`exp(-nr^2/(2pi^2))`. Enlarging its domain to the complement of the disk
gives `pi/(2n) exp(-n/(2pi^2))`. The Gaussian Fourier integrand on the
whole complement gives `2/(pi n) exp(-n/8)`. These are the two remaining
terms in B_n.

The inverse Fourier transform of `exp(-n|t|^2/8)` on the full plane is g_n.
One may derive the one-dimensional transform directly: its derivative in z,
followed by integration by parts in t, gives `I'(z)=-(4z/n)I(z)`;
the zero value is the Gaussian integral `sqrt(8pi/n)`. Solving this scalar
equation and multiplying the two coordinates gives the displayed g_n.
The Gaussian integral itself follows by squaring it, using polar coordinates,
and integrating the radial exponential. All exchanges here are justified by
integrable Gaussian bounds, and the original inversion involves a finite
polynomial. No weak-convergence-to-point-probability inference is used.

Splitting the two inverse transforms over the unit disk and its complements
and applying the triangle inequality proves (1). In particular, the estimate
does not omit a second Fourier maximum or assume positive probabilities where
the walk has no support.

## Logarithmic return and Euclidean readout

In the central window, `g_n(z) >= [2/(pi n)]exp(-2R^2)`. Dividing (1) by
this explicit lower bound gives relative error at most E_R(n), including at
the origin. If `|u-1| <= e < 1`, then u >= 1-e > 0 and

```
|log u| <= e/(1-e).
```

This follows either by integrating 1/u or from `log u <= u-1` and the same
inequality for 1/u. Applying it separately to p_n(z)/g_n(z) and p_n(0)/g_n(0)
gives (2). [QuadraticLimit.lean](../../lean/GeoSBT/QuadraticLimit.lean) proves
this exact logarithmic bridge with all positivity and relative-error premises
explicit. It also proves the three-error residual bound in (3).

The uncentered cost has the corresponding unfitted law:

```
|-log p_n(z) - [log(pi n/2)+2|z|^2/n]| <= E_R(n)/(1-E_R(n)).
```

This follows from the same relative bound using just one logarithm;
`log_cost_error` mechanizes that step. It supplies both the predicted
quadratic coefficient and the predicted offset in the paper's cost formula.

The nonnegative cost premise is derived independently. Taking the real part
of the exact inversion formula gives

```
p_n(0)-p_n(z) = (1/(4pi^2)) integral phi(t)^n [1-cos(t.z)] dt >= 0.
```

Here phi^n >= 0 is essential. Thus, wherever p_n(z)>0, F_n(z)>=0.
The existing mechanized square-root bridge now applies with a derived,
uniform premise:

```
|sqrt(F_n(z)/2)-|z|/sqrt(n)| <= sqrt(d_R(n)/2) -> 0.   (5)
```

This gives an actual Euclidean limit of the displacement readout. On any
bounded continuum set K, choose scaled lattice representatives
`z_n(u)=(floor(sqrt(n)u1),floor(sqrt(n)u2))`. For all u,v in K, one fixed R
bounds their normalized displacement for all stages. Coordinate rounding
changes that displacement by at most sqrt(2/n). Consequently

```
sup_(u,v in K) |sqrt(F_n(z_n(u)-z_n(v))/2)-|u-v||
  <= sqrt(d_R(n)/2)+sqrt(2/n) -> 0.
```

The finite readout need not satisfy the triangle inequality exactly. Its
uniform limit does. This does not identify F_n itself, a squared-displacement
account, with a metric; nor does it identify the separately optimized raw
negative-log macro path metric with this square-root readout.

For the torus, choose integer N_n > 2n and eventually n >= R^2. Any walk
displacement has each coordinate in [−n,n], so two supported displacements
cannot be congruent modulo N_n. The central lattice probabilities therefore
equal the torus probabilities exactly. This includes the finite original
N=512 stages through 128, but an asymptotic statement must grow N_n. At fixed
N the long-time law is uniform instead. A fixed positive probability floor
would also eventually clip central probabilities of order 1/n: the theorem
uses the true positive probabilities, or a floor below them, rather than
silently taking a limit through fixed smoothing.
The theorem's window also does not use a fixed positive fitting threshold,
which would eventually discard probabilities of order 1/n.

## A matched control with correlated steps

Retain lazy probability one half, give each cardinal move probability 1/16,
and add (1,1) and (−1,−1), each with probability 1/8. This changes the actual
step covariance and keeps finite-range, symmetric, aperiodic dynamics. Its
covariance and limiting accounting form are

```
Sigma = [[3/8,1/4],[1/4,3/8]],
Q_n(x,y) = (12/5)(x^2+y^2)/n - (16/5)xy/n.
```

The covariance is computed directly from the one-step probabilities.
Its determinant is 5/64 and its inverse is
`[[24/5,-16/5],[-16/5,24/5]]`. Thus Q_n is exactly
`z^T Sigma^(-1) z/(2n)`, a positive quadratic form.

The same Fourier argument supplies a uniform local limit, rather than
assuming that a covariance computation establishes it. For this walk,

```
phi_c = 1/2 + [cos(t1)+cos(t2)]/8 + cos(t1+t2)/4,
0 <= phi_c <= 1,
q_c >= A r^2,                    A=1/(4pi^2),
h_c = (1/2)t^T Sigma t,
r^2/16 <= h_c <= 5r^2/16,
0 <= h_c-q_c <= 3r^4/64.
```

The cardinal terms give the global lower bound on q_c. The Taylor bound on
the three cosine terms gives the remainder: `t1^4+t2^4<=r^4` and
`(t1+t2)^4<=4r^4`. The power-telescoping estimate then bounds the Fourier
difference by

```
(49n/512) r^4 exp(-(n-1)A r^2).
```

Integrating this on the square and enlarging to the whole plane, then bounding
the Gaussian tail outside the square by the complement of the radius-pi disk,
gives the uniform absolute estimate

```
|p_n^c(z)-g_n^c(z)|
 <= [49/(1024pi A^3)] n/(n-1)^3 + [4/(pi n)] exp(-pi^2 n/16),
g_n^c(z) = [4/(pi sqrt(5)n)] exp(-Q_n(z)).
```

To obtain the Gaussian transform, diagonalize the positive covariance matrix
by an orthogonal change of variables and apply the preceding one-dimensional
calculation. In every fixed central window |z|<=R sqrt(n), Q_n(z)<=4R^2,
so dividing the estimate by `4/(pi sqrt(5)n) exp(-4R^2)` gives relative error
O_R(1/n). The logarithmic bridge proves uniform convergence of the centered
control cost to Q_n. The same nonnegative-characteristic-function argument
also proves its centered cost is nonnegative.

At `x=y=k_n=floor(sqrt(n))`, the control's axis residual therefore converges
to **−16/5**, since `k_n^2/n -> 1`. This is a real dynamical failure of the
paper's coordinate-axis test. Its nonzero residual is caused by a cross term,
not by unsupported far-tail points or smoothing noise. It does **not** show
that all anisotropy destroys Pythagoras: the control still has a weighted
inner-product geometry, whose orthogonal directions differ from these axes.
Anisotropic walks with diagonal covariance can retain axis separability.

## Evidence and claim coverage

[audit_quadratic_limit.py](../../scripts/audit_quadratic_limit.py) checks the
uniform probability estimate against nonnegative convolution at stages
16 through 256, labels these comparisons as floating computations, and checks
the torus implementation against the same lattice walk. The analytic formulas
at larger stages are explicitly recorded as bounds, not executions. The exact
finite certificates remain sharper than this conservative universal estimate
at the original stage 128.

For the control, the script reconstructs integer counts with denominator 16
per step, verifies total mass and obtains rational logarithm enclosures from
the exact ratios. The four saved stages certify an axis residual strictly
below −1. The verifier rejects changed steps, stages, covariance, counts or
intervals and independently reruns the recurrence. This is exact arithmetic
in Python, outside Lean's trust boundary.

This result preserves and strengthens the original E5 mechanism claim:
staging of the specified isotropic walk produces a uniform quadratic and
axis-additive accounting law and a Euclidean displacement readout. It replaces
the unsupported weak-central-limit shortcut with a derived local estimate.
The predicted universal failure under anisotropy remains false; the matched
control isolates a condition that does fail. Curvature identification,
automatic lens discovery and a Euclidean macro path-metric limit require
separate arguments.

The proof received an adversarial self-review of the Fourier maxima, Taylor
and integration constants, probability lower bounds, logarithm signs,
rounding domains, floor and torus scaling, control covariance and residual
limit. The universal Fourier and limit statements are written mathematics,
not a full Lean mechanization or an independent review. Build, axiom, test
and exact verification receipts accompany the new audit pack.
