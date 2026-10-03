# Mathematical review and repairs

**Historical checkpoint.** The [current claim ledger](review_claims_20261003.md)
supersedes the open-work status below. Subsequent constructive work supplies a
coherent fractal regime with a different lens; the original learned-lens defects
are retained as contrary evidence.

Follow-up: [Closure proofs and finite refinement](closure_and_refinement_review_20261003.md)
adds mechanized Markov results, exact finite quadratic-cost certificates,
larger-substrate checks and current verification receipts. The receipts below
describe the initial repair checkpoint.

The weighted metric construction and its Lean coverage have been repaired. Corrected experiments retain a grid versus sphere loop-residue separation, an approximately quadratic cost law at large staging, and deformation under directional gating. The stronger interpretation that the canonical refinement ladders satisfy the paper's small-defect coherence criterion is unverified and conflicts with large reported prototype defects. The fractal fixed-point interpretation also lacks the required geometric evidence. Those main-claim questions remain open for discussion with the author; this review does not authorize a downgrade.

The paper and historical run packs were left unchanged. This note records the mathematical issues, repairs, current evidence, and work needed before the later paper revision. The requested baseline checkpoint is commit `4e2e9a0`. The review covered the paper's constructions and exhibits, all four Lean modules, the numerical library, experiment runners, diagnostics, and original tests. A separate adversarial pass over the repairs was a self-review.

## Correct metric construction

For a finite row-stochastic kernel P, a deterministic coarse map C, and distribution-valued prototype rows U, both E = P^tau C U and the macro kernel K = U P^tau C are row-stochastic. If prototypes are supported on their respective nonempty fibers, U C = I. These finite constructions are valid independently of an emergence interpretation.

The paper's formula `-log(p + eta)` is not generally nonnegative: p = 1 and eta > 0 give a negative cost. Negative undirected edges give negative-cost cycles, invalidate Dijkstra, and can make the path infimum minus infinity. The implementation actually used `-log(max(p, eta))`; that rule is valid for probabilities p and 0 < eta <= 1. The implementation now checks these domains, rejects nonfinite kernels, and clips tolerance-sized probability errors before taking logs. It keeps an edge only when its weight exceeds the separate edge threshold. A floor does not add absent transitions. Averaging K and its transpose gives symmetric weights bounded by one but need not give a stochastic matrix; it is an undirected weight array.

Nonnegative additive path costs, with the empty path included, give zero self-distance and the directed triangle inequality. Symmetric weights give an extended pseudometric. Disconnection permits infinite distances; distinct vertices can have zero distance. For example, the two-state deterministic swap has zero transition costs. Identifying zero-distance classes gives an extended metric, and finite distances are an additional requirement for an ordinary metric.

The original SciPy dense graph call treated zero off-diagonal costs as absent edges. It could disconnect precisely the zero-distance classes that the quotient construction must preserve. The repaired implementation uses explicit sparse entries, including zeros; the fallback uses the same finite-edge semantics. Both are tested against a zero-edge counterexample. Negative and NaN costs are rejected before either backend.

The original distortion diagnostic silently omitted pairs whose reachability disagreed. It now records unmatched pairs and connectivity, returns infinity for a failed global finite-metric audit, and reports the finite-pair discrepancy separately. This also avoids subtracting infinity from infinity. A finite discrepancy on common finite pairs is not evidence of global coherence for a disconnected graph. The historical threshold sweeps named `*_eps_edge_disconnect` remain connected; they do not demonstrate a disconnecting failure. A regression test now exercises a threshold that actually disconnects the macro graph.

## Lean coverage

The original `graph_edist_triangle` is correct for unweighted SimpleGraph distance. It did not represent arbitrary weighted negative-log costs. `GeoSBT/PathMetric.lean` now defines actual finite protocols, their extended nonnegative costs, concatenation, reversal, and the infimum over protocols with specified endpoints. It proves weighted zero self-distance and the triangle inequality, and proves symmetry under an explicit symmetric-weight hypothesis. These facts construct a `PseudoEMetricSpace` without assuming connectivity, positive edge costs, or separation.

`GeoSBT/QuotientMetric.lean` retains the ordinary separation result and adds the extended construction and preservation of source distances. `weightedSeparationMetric` applies the quotient to the newly constructed weighted distance, rather than assuming an unrelated pseudometric. Infinite distances remain infinite after quotienting. The old duplicate metric instance was replaced with a named definition using mathlib's existing instance.

`GeoSBT/LikelihoodCost.lean` proves nonnegativity of the flooring rule under explicit probability bounds. It also proves that conversion into extended nonnegative reals preserves the real cost. This prevents relying on `ENNReal.ofReal` to silently truncate an invalid negative ledger.

`pythagoras_real` is a correct imported classical inner-product theorem with an explicit orthogonality hypothesis. It supplies no proof that the experimental cost, or the macro path metric, is Euclidean. No new Six Birds theorem is invoked to establish that missing connection.

Fresh `lake build` succeeds with the pinned Lean 4.27.0 and mathlib revision. `lake env lean Audit.lean` reports only `propext`, `Classical.choice`, and `Quot.sound` for the reviewed anchors. There are no added axioms or `sorryAx`. The receipt is [lean_axioms.txt](math_review_20261003/lean_axioms.txt). Floating-point implementation correctness, manifold limits, Markov closure, and curvature identification remain outside these formal declarations.

## Loop residue and chart orientations

Classical MDS determines each local chart only up to an independent orthogonal transformation, including reflection. The old transport forced a proper rotation between every pair of charts. A counterexample with exact planar coordinates and three different overlapping neighborhoods acquires a residue of about 0.231 radians after reflecting one chart, although the underlying distances are unchanged.

Transport now uses the full orthogonal Procrustes alignment and checks that the centered cross covariance has full rank. Under chart changes Q_i, the transport transforms as `R_ij -> Q_i^T R_ij Q_j`. A loop product therefore transforms by conjugation at its starting chart. Its absolute rotation angle is invariant. Orientation-reversing loop products have no rotation angle and are counted separately, as are missing transports. Tests cover independently rotated, translated, and reflected exact planar charts and gauge changes of distorted charts.

In the corrected canonical configuration the grid median is **0.00169027** and the sphere median is **0.0428365** radians, a ratio of **25.34**. The grid has 789 sampled and evaluated triangles; the sphere has 800. Neither case excludes a loop for missing transport or orientation reversal. This preserves the qualitative separation but replaces the old 0.0479 and 0.5980 values. These are results under the specified finite protocol. Gauge invariance removes a demonstrable artifact; it does not identify the measured residue with a Riemannian curvature tensor or prove that curvature is its only source.

Proper rotations in two dimensions commute: SO(2) is abelian. Nonzero loop residue therefore does not establish algebraic noncommutativity of the fitted rotation matrices. Transports between different fibers are naturally compared as paths with compatible endpoints; treating them as common-domain operators requires an additional construction. The current diagnostic supports path dependence of transport, not the stronger identification of curvature with operator noncommutativity. Exact flat charts also provide a counterexample to the suggestion that patchwise packaging alone forces nontrivial residue.

## Staged costs and Pythagoras

The old FFT calculation assigned roundoff probabilities of order 1e-17 to mathematically unreachable displacements. At tau = 4, for example, a displacement with Manhattan length greater than four has probability exactly zero before torus wraparound. The logarithm amplified those artificial probabilities into finite costs that contaminated fits and sampled residuals. Clipping negative FFT entries did not fix positive roundoff mass.

The distribution now uses nonnegative stencil convolution. It preserves unreachable zeros and the parity constraint of the non-lazy walk at the support level. Fits, axis comparisons, circularity, and separability residuals use probabilities above the declared fitting threshold; sample counts are recorded. An empty supported residual sample produces `null`, rather than an invented finite result. The displacement window must have unique torus representatives. These remain floating-point computations, with no interval or exact-arithmetic certificate for positive probabilities.

The corrected canonical results are:

| Diagnostic | Stage 4 | Stage 128 |
|---|---:|---:|
| Quadratic cost fit RMS | 0.471062 | 0.150864 |
| Fit RMS divided by cost standard deviation | 0.246392 | 0.0240619 |
| Median absolute separability residual | 0.861566 | 0.0585894 |
| Supported residual samples out of 2000 | 307 | 2000 |
| Quadratic axis fit RMS | 0.0541822 | 0.0176878 |

At stage 128 the linear axis fit RMS is 1.05356, so the quadratic model is strongly favored on this window. The corrected residual is not monotone at every stage. The early huge residual and the apparent sharp transition around stage 16 were partly support and numerical artifacts. The comparison changes its supported domain with stage; it is not a uniform theorem or a comparison over identical samples.

The residual in the paper tests **axis separability**, not quadraticity: `|x| + |y|` has residual exactly zero too. The L1 control now reports that fact explicitly, together with its failure of squared-distance Pythagoras, whose right-triangle residual is `2 |x y|`. Quadratic axis fits and radial comparisons provide separate evidence for the quadratic shape. L1 is a geometric control, not a matched stochastic control.

A centered quadratic cost is proportional to squared Euclidean displacement, not to a metric: squared displacement fails the triangle inequality on 0, 1, 2. Taking its shortest-path envelope generally changes the quadratic law. The staged displacement exhibit does not pass through the macro path-metric construction. Comparing diagonal and axis costs at the same tau is a shape identity; it is not an equality of likelihoods for a tau-step protocol and two composed tau-step protocols.

A possible rigorous readout is conditional: if `|(C(z)-C(0))/a - ||z||^2| <= epsilon` uniformly on a domain, a > 0, and the centered cost is nonnegative, its square root approximates Euclidean displacement there with error at most sqrt(epsilon). Current RMS fits do not establish that uniform premise. The defensible current result is a finite approximately quadratic and separable cost exhibit at large staging.

The discussion's prediction that anisotropy must destroy the Pythagorean form is too strong. A centered anisotropic Gaussian gives `C(x,y) = a x^2 + b y^2 + c`, still a separable positive quadratic form and a Pythagorean identity in its associated weighted inner product. It changes circular contours into ellipses. A drift can shift the center while preserving a quadratic form too. Controls must specify which property should fail: isotropy, centered axis scaling, separability, or an inner-product readout.

A Gaussian cost asymptotic requires a local limit estimate for probabilities on a specified domain, with positivity and error bounds sufficient to take logarithms. Weak central-limit convergence alone does not give that estimate. On a fixed finite torus the eventual limit is uniform; retaining a lattice Gaussian regime through increasing stages requires controlling torus size and the displacement window. The current finite experiment supplies no uniform local limit theorem.

## Coherence and prototype persistence

The extreme-row TV formula for delta is correct: a distributional input is a convex combination of rows, so its output discrepancy is at most the largest row discrepancy, and a point mass attains that maximum. But delta is always at most one for stochastic operators. Calling it merely bounded supplies no evidence of closure. The finite-run distortion is also automatically finite on connected finite graphs, so bounds require a declared scale and meaningful tolerance.

For disjoint-fiber prototypes, lifting is an L1 isometry: the supports are disjoint and each row has mass one. Thus prototype stability is exactly `1 - K[x,x]`, the probability of leaving the macro label. This identity follows by applying that isometry to the row `K[x,:] - e_x`; it is also checked numerically in the tests and audit.

The corrected finest grid mean stability defect is **0.62520** with maximum **0.78259**. The sphere mean is **0.85516** with maximum **0.98514**. These do not support the claim that prototype drift is small across the full canonical ladder. Mobility can be expected in a useful random walk, but a criterion requiring almost fixed prototypes cannot be declared satisfied by high mobility.

Small one-versus-two closure defect also does not control arbitrary repetitions for free. TV contraction gives the finite-horizon bound `TV(mu E^k, mu E) <= (k-1) delta`, capped at one; this does not show a uniform small drift as k grows.

The pipeline now actually records route mismatch for adjacent triples. It compares `U_fine P^(2 tau) C_coarse` with `U_fine P^tau C_mid U_mid P^tau C_coarse` by maximum row TV. This is an explicit comparison on finest prototype inputs with equal total staging. It is not the full microstate supremum or a claim that all routes commute. Those fields were absent from the historical canonical packs despite the text saying they were recorded.

Restoring the stronger coherence conclusion requires declared tolerances, a justified scale range and time horizon, and either evidence satisfying the current persistence criterion or a mathematically defended transport-based criterion. Artificially slowing every transition can make escape defects small without establishing the intended geometry, so that alone is not a repair. This main interpretation remains pending discussion.

## Fractal and anisotropic exhibits

The Sierpinski generator produces the intended finite recursively glued gasket graph. Its existence and connected induced metric do not show that the packaged geometry has a non-smoothing or scale-stable fractal fixed point. The current evidence uses one substrate size and a finite lens ladder. A proof or experiment about persistence through substrate refinement, or a direct test of failure of Euclidean local approximation, is missing.

The corrected dimension proxies on the toy configurations are grid ball slope **1.671** and gasket ball slope **1.389**. Information slopes are **4.660** and **2.072**. Replacing d by sqrt(d) approximately doubles these exponents; ball slopes become **3.341** and **2.777**. Such dependence on the metric convention and scale proxy is expected, but it precludes identifying these outputs with the geometric dimensions 2 and log(3)/log(2) without a bridge. Entropy inputs and regressions are now validated, and saturated or degenerate ball-growth fits fail explicitly rather than returning a misleading dimension. The fractal main claim remains pending discussion.

Directional gating changes the kernel, but the canonical comparison also changes its spectral partition. To isolate the dynamical contribution, the audit includes a new comparison holding the baseline grid lens and prototypes fixed. With east gating of strength one, the macro-kernel maximum entry change is **0.23070**, and the induced distance maximum change is **2.42288**. On predefined horizontal and vertical grid pairs separated by four micro lattice steps, the ratio of mean macro distances changes from **0.98405** to **1.08690**. This supports deformation under constraints at a fixed interface. The grid directions are audit readouts supplied by the controlled substrate, not inferred coordinates or a general theorem about anisotropic manifolds.

## Additional numerical repairs

Stationary iteration now uses a lazy iteration preserving stationary distributions, checks its original-kernel residual, and fails on exhaustion. This fixes periodic chains that previously returned an oscillating, nonstationary iterate. K-means now ensures all requested clusters are populated even with duplicate coordinates. Integer lens-level requests are bounded by the substrate size. Labels are validated before integer conversion, so fractional labels cannot silently change the quotient. Kernel validation distinguishes weak from strong directed connectivity. The spectral solver has a fixed initial vector; reproducibility still depends on numerical libraries and eigenspace degeneracies, so cross-environment bitwise identity is not promised.

## Evidence and remaining work

The new [evidence.json](math_review_20261003/evidence.json) records six canonical runs, all fourteen committed sweep configurations, finite metric checks, a fixed-lens gating control, dimension proxies, and hashes of the mathematical source files. Each corrected run has its own summary and plots under `math_review_20261003/`. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python3 scripts/audit_mathematics.py --output results/math_review --sweeps
cd lean
lake build
lake env lean Audit.lean
```

The final Python suite passes **45 tests**, including the mathematical counterexamples and corrected failure controls. The exhaustive metric checks are exhaustive only on their finite test matrices. Neither passing tests nor green Lean builds establish the open coherence and fractal claims.

The remaining choice is whether to pursue additional mathematical criteria and experiments to restore those interpretations or retain the validated finite constructions with weaker conclusions. No paper edits or material claim downgrades have been made. This is a verified repair checkpoint, not completion of the full review goal.
