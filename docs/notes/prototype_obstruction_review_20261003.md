# Exact limits of prototype repair for the canonical partitions

Changing prototype weights cannot make the canonical finest grid or gasket
pass a small prototype-stability tolerance. Even allowing **every** stochastic
prototype supported on its own fiber, the best possible worst defects are
exactly `6411/8192` for the grid and `6957/8192` for the gasket. The original
uniform prototypes already attain these worst-case optima. This rules out one
plausible repair without declaring that all learned lenses or other staging
choices must fail.

## Optimization theorem

Fix the original micro kernel P, the stage tau, and the partition C. Write
`B=P^tau C`. For any stochastic fiber-supported prototypes U, the established
signed-L1 isometry gives the exact prototype defect

```
s_U(x) = TV((U B U)_x,U_x) = 1-(U B)_xx.
```

The diagonal entry is `sum_(z in fiber x) U_xz B_zx`, a convex average of
the fiber's return probabilities. Therefore

```
s_U(x) >= 1-max_(z in fiber x) B_zx = b_x.       (1)
```

Each nonempty finite fiber has a maximizing point. Assigning its point mass
as the prototype attains b_x. These choices can be made independently for
all fibers; disjoint supports and stochasticity are retained. Consequently

```
min_U max_x s_U(x) = max_x b_x.                 (2)
```

The objective is the actual lifted TV defect, not merely an averaged return
surrogate. [PrototypeObstruction.lean](../../lean/GeoSBT/PrototypeObstruction.lean)
proves the universal convex-average upper bound and transfers it to the TV
lower bound through the already mechanized equality. The finite maximizer
construction and exact calculations supply attainment in (2).

## Exact canonical witnesses

The input labels are the previously recorded canonical partitions used in the
[local nonembedding certificate](local_nonembedding_review_20261003/certificate.json):
grid side 25 and gasket level 5, stage five, lazy probability one half,
six eigenvectors, seed zero, and ladder `[4,8,16,32,64,128]`. No partition is
changed to obtain an obstruction. The result is conditional on these recorded
labels; it does not certify eigensolver output across all environments.

[audit_prototype_obstruction.py](../../scripts/audit_prototype_obstruction.py)
reconstructs `P^5 C` using integer recurrence, with step denominator 24 for
the open grid and eight for the gasket. It verifies row mass at every step
and guards int64 ranges. Every fiber return and optimum is a rational number.

| Canonical metric input | Worst fiber | Microstates in that fiber | Best possible return | Minimum worst prototype defect |
|---|---:|---|---:|---:|
| Grid | 6 | 192, 193 | 1781/8192 | 6411/8192 = 0.7825927734375 |
| Gasket | 4 | 299 | 1235/8192 | 6957/8192 = 0.8492431640625 |

Both grid states have the same exact return to their two-state fiber. The
gasket fiber is a singleton. Thus every supported prototype on these particular
fibers has the stated large defect. Across all other fibers, the exact uniform
defects are no larger. This proves that uniform prototypes attain the minimax
values in the table; an alternate choice may improve the mean without improving
the worst case.

The [certificate](prototype_obstruction_review_20261003/evidence.json) contains
all labels, fiber-wise optima, uniform defects, selected optimal point
prototypes and the two witness fibers. Independent CLI verification rebuilds
every staged count from the fixed canonical input and rejects fabricated
improvements, changed counts or fractional integer fields. The Python exact
checker is outside Lean; the theorem's numerical premises are explicit.

## Scope of the failure and the restoration route

This is a lower bound on prototype persistence, not a lower bound on the
idempotence defect delta. One failed coherence requirement suffices to rule out
passing the entire small-defect conjunction at these fixed parameters. It does
not rule out other lenses, stages, substrate sizes or criteria with a different
scientific interpretation. Point-prototype attainment of the scalar objective
also does not prove connectivity or favorable distortion of the resulting
macro metric.

The [recursive-cell gasket construction](recursive_gasket_coherence_review_20261003.md)
and [block-grid construction](block_lens_coherence_review_20261003.md) already
give restoration routes using the original dynamics and stage five with
different supplied lenses and explicit units. These results should be used
for the coherent-regime existence claim; the old finest learned-lens examples
must retain their failing persistence audits. Prototype optimization alone
cannot justify reclassifying them.

The derivation received an adversarial self-review of support, TV versus return,
finite attainment, worst versus mean objectives, and fixed-partition scope.
It is not independent external review.
