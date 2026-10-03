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
produce an inner product.

`Audit.lean` prints the transitive axioms of the anchors. The mathematical
declarations should use only Lean's standard foundations (`propext`,
`Classical.choice`, `Quot.sound`), with no `sorryAx` or added axioms. These files do
not verify floating-point algorithms, Markov closure, curvature identification,
or continuum limits.
