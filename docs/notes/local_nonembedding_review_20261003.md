# Exact local nonembedding witnesses for the learned lenses

The original canonical gasket metric has a certified finite patch that cannot
be uniformly approximated in any real inner-product space to error at most
**9.1% of its radius**. The canonical grid also has an obstruction, at **5.6%**.
This supports a finite non-Euclidean local exhibit but demonstrates that this
test alone cannot distinguish a fractal from a grid. It supplies no theorem
about every neighborhood or an asymptotic non-smoothing limit.

This follows the [closure and refinement review](closure_and_refinement_review_20261003.md).
A separate [recursive-cell construction](recursive_gasket_coherence_review_20261003.md)
establishes a coherent fractal family with a genuine limit argument. Its supplied
lenses must not be confused with the learned lenses examined here.

## The obstruction and its coverage

For any four points in a real inner-product space,

```
d01^2 + d23^2 <= d02^2 + d03^2 + d12^2 + d13^2.
```

The difference between the right and left sides is
`||p0+p1-p2-p3||^2`. This identity and inequality are proved in
[EuclideanObstruction.lean](../../lean/GeoSBT/EuclideanObstruction.lean).
The theorem does not assume a dimension or completeness.

Suppose certified lower bounds for the two diagonals are L01, L23, upper
bounds for the other four distances are Uij, and the proposed additive fitting
error is delta >= 0. A candidate fitting all six distances to error at most
delta would have diagonals at least Lij-delta and other distances at most
Uij+delta. If L01,L23 >= delta and

```
(L01-delta)^2 + (L23-delta)^2
  > (U02+delta)^2 + (U03+delta)^2 + (U12+delta)^2 + (U13+delta)^2,
```

no such fit can exist. `no_hilbert_uniform_approximation` mechanizes this
contradiction with the interval and fit premises explicit. Unlike an MDS
reconstruction error, this is a lower bound applying to **all** fits, including
fits in higher dimensions and those found by other algorithms.

## Exact probabilities, protocols and logarithms

[certify_local_nonembedding.py](../../scripts/certify_local_nonembedding.py)
reconstructs the ideal kernel using integers. For the lazy gasket, each step
has denominator 8; for the open grid the common denominator is 24, accounting
for degrees two, three and four. Five integer recurrence steps reconstruct
`P^5 C`. Aggregating over each recorded nonempty fiber and dividing by its
size gives the uniform-prototype macro kernel. Rational weight averaging,
the declared threshold 1e-15, and floor 1e-12 match the mathematical metric
protocol. Integer ranges are checked before any int64 arithmetic.

For positive symmetric edge weights w <= 1, a protocol's negative-log cost is
the negative log of its weight product. A maximizing product is attained by
a simple path: removing a cycle cannot lower the product. The checker uses
exact rational multiplicative Dijkstra. The usual settled-vertex argument
applies because multiplying by an edge cannot increase a product. It rejects
invalid weights and disconnection; the empty path has product one.

Distance logarithms are enclosed exactly. Reduce a rational t >= 1 to
`t = 2^k u`, with 1 <= u < 2, and put z = (u-1)/(u+1). For n = 24,

```
S = 2 sum_{j=0}^n z^(2j+1)/(2j+1),
S <= log(u) <= S + 2 z^(2n+3)/((2n+3)(1-z^2)).
```

Integrating the finite geometric expansion of `1/(1-z^2)` proves the bounds;
0 <= z <= 1/3 keeps the remainder positive and bounded. Enclose log(2) the
same way and add k times its interval. Outward rational rounding to units of
1e-12 keeps witness files small. Every final gap comparison is rational.

## Locality and finite results

The learned labels are recorded explicitly, with seed zero, six spectral
coordinates, uniform prototypes and staging five. The canonical cases use the
paper's complete `[4,8,16,32,64,128]` ladder. The size family uses the earlier
declared `m = min(256,max(16,n//3))` rule. The exact checker treats these labels
as the chosen interface; it does not formalize the floating eigensolver or
prove an optimality property of the learned partition.

Each candidate patch contains a center and `min(24,m-1)` nearest neighbors.
Their order is reconstructed from exact maximum products, with integer label
tie breaking. Thus locality is checked rather than inferred from a plot.
The exact center eccentricity is a lower bound on global diameter. Dividing
the radius upper bound by that eccentricity lower bound certifies the reported
radius/diameter upper bound.

Floating search examines all centers and all four-point subsets of each
candidate patch. It chooses a center from the median candidate score, then
uses exact neighbors to choose and check the witness. Its distributional
statistics are **candidate statistics**, not independently certified population
claims. The following table concerns one exact witness per configuration:

| Configuration | Micro states | Macro states | Excluded relative error | Radius/diameter upper bound |
|---|---:|---:|---:|---:|
| Grid family | 49 | 16 | 8.6% | about 1 |
| Gasket family | 42 | 16 | 10.7% | about 1 |
| Grid family | 121 | 40 | 7.4% | 0.6830 |
| Gasket family | 123 | 41 | 11.3% | 0.6657 |
| Grid family | 361 | 120 | 6.2% | 0.3114 |
| Gasket family | 366 | 122 | 9.4% | 0.3214 |
| Grid family | 1089 | 256 | 6.2% | 0.1939 |
| Gasket family | 1095 | 256 | 10.3% | 0.1600 |
| Canonical grid | 625 | 128 | 5.6% | 0.2763 |
| Canonical gasket | 366 | 128 | 9.1% | 0.2530 |

Errors are relative to the selected patch radius, not its diameter or the
graph diameter. If R is its true radius and U_R its certified upper bound,
the checker excludes absolute fitting error `delta = epsilon U_R`. Since
`epsilon R <= delta`, this rules out every fit with error at most epsilon R.
It gives a lower bound at least epsilon R on the infimum fitting error.
The smallest examples cover nearly the whole macro graph and provide a global
obstruction rather than convincing evidence about small neighborhoods.

The [certificate](local_nonembedding_review_20261003/certificate.json) records
all labels, exact witness vertices, rational intervals, strict gaps and source
hashes. Independent CLI verification reconstructs the macro probabilities,
path optima, nearest-neighbor sets and logarithmic bounds. Exact arithmetic
does not remove the need for a correct checker: the Python checker is outside
Lean. Lean proves the mathematical implication of the interval premises,
not the recurrence, path search or JSON contents.

## Interpretation and review

These certificates repair the earlier absence of a genuine finite lower bound.
They do not prove persistence for all future substrate sizes, all centers,
all scales, or a limiting tangent space. Grid witnesses also obstruct Euclidean
fit, so nonembedding cannot serve as an exclusive fractal test. Dimension and
refinement require separate evidence. High prototype defects in the canonical
gasket likewise remain a separate failure of the current coherence criterion.

Adversarial tests compare exact recurrence weights with the original numerical
macro kernel, exercise a composed-path optimum, reject probability weights
that would invalidate Dijkstra, and distinguish an L1 square from a Euclidean
square. Tampered intervals, duplicate vertices, false error margins and unsafe
integer domains are rejected. The accompanying code repairs also prevent
fractional lens counts and labels from silently passing refinement audits,
validate diffusion-kernel domains, and reject NaN gate strength. Complete gate
removal has an explicitly documented absorbing-row completion rule.

The interval-to-obstruction bridge, normalization, minimum-versus-infimum
wording, nearest-neighbor selection, and finite-versus-limit scope received a
self-review. No independent reviewer was used and the paper was not edited.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/certify_local_nonembedding.py --verify docs/notes/local_nonembedding_review_20261003/certificate.json
```
