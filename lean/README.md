# Lean anchors

Build with the pinned Lean 4.27.0 and mathlib revision in `lake-manifest.json`:

```bash
cd lean
lake build
lake env lean Audit.lean
```

`PathMetric.lean` retains the original unweighted `SimpleGraph.edist` triangle
anchor and adds actual weighted finite protocols. Their costs take values in
`ℝ≥0∞`; infinity represents a missing transition. Concatenation proves the
triangle inequality for the infimum of protocol costs. The empty protocol proves
zero self-distance. Symmetric edge costs give a `PseudoEMetricSpace`; neither
connectivity nor separation is assumed.

`QuotientMetric.lean` covers both ordinary and extended separation quotients,
preserving the source distance. `weightedSeparationMetric` applies that
construction directly to the weighted protocol distance. A disconnected graph
therefore gives an **extended** metric on the quotient. Obtaining an ordinary
finite metric additionally requires finite distances.

`LikelihoodCost.lean` proves that the implemented flooring rule
`-log(max(p, eta))` is nonnegative when `p ≤ 1` and `0 < eta ≤ 1`, and that encoding
this cost into `ℝ≥0∞` preserves its value. Thresholding is a separate choice that
sets absent edges to infinity. Additive smoothing `-log(p + eta)` has no such
general nonnegativity guarantee.

`Pythagoras.lean` is a classical inner-product-space theorem. It does not prove
that a computed cost or path metric is Euclidean, or that stochastic dynamics
produce an inner product. It also proves a conditional readout bridge: a uniform
bound on nonnegative quadratic cost gives a square-root displacement bound.
An RMS fit does not supply the uniform premise.

`MarkovClosure.lean` proves stochasticity of the actual finite closure
`P^tau C U` and macro kernel, the extreme-row TV criterion, the unconditional
closure-defect factorization, and a finite-horizon repetition bound with loss
`k D`. Fiber-supported stochastic prototypes satisfy `U C = I`, preserve signed
L1 discrepancies, and have stability defect exactly equal to macro escape mass.
It does not prove that any experimental ladder has small defects or coherent
geometry.
`closure_defect_le_macro_escape` additionally derives a bound on every
microstate closure-defect row from a bound on macro escape, under those same
stochasticity and fiber-support conditions.

`EuclideanObstruction.lean` proves the four-point squared-distance inequality
in every real inner-product space, and proves that a strict interval gap rules
out any fit with a stated uniform distance error. The recursive gasket's
positive-gap and logarithmic cost bounds are also mechanized. Python exact
counts and paths supply finite witnesses; recursive graph laws and the metric
limit argument are written proofs outside Lean.

`QuadraticLimit.lean` proves quantitative logarithm bounds from relative
probability error, returns both uncentered and centered accounting costs,
and bounds the axis residual under an additive reference law. The original
walk's Fourier local-limit estimate is derived in a written proof outside
Lean; it supplies the applicability premises rather than an RMS fit.

`Connection.lean` defines covariant shifts on a common space of frame fields.
It proves their commutation is equivalent to trivial square holonomy under
commuting base translations, including for abelian coefficient groups. It
also proves the gauge laws and the triangle route criterion. The vector-section
operator norm and spherical parallel-transport calculations are written
arguments outside Lean; they do not establish MDS-estimator consistency.

`PrototypeObstruction.lean` proves a lower TV-defect bound for every stochastic
fiber-supported prototype from a return bound holding throughout a fixed
fiber. Exact canonical grid/gasket counts establish that reweighting prototypes
cannot reduce their worst persistence defect. This does not lower-bound the
closure-idempotence defect or exclude other lenses and stages.

`Audit.lean` prints the transitive axioms of the anchors. The mathematical
declarations should use only Lean's standard foundations (`propext`,
`Classical.choice`, `Quot.sound`), with no `sorryAx` or added axioms. These files do
not verify floating-point algorithms, curvature identification, or continuum
limits. The exact quadratic-walk certificate checker is Python code, outside
the Lean trust boundary.
