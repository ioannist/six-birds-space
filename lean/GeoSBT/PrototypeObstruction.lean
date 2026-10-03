import GeoSBT.MarkovClosure

open scoped BigOperators

namespace GeoSBT

/-- Every supported stochastic prototype averages the micro return
probabilities in its own fiber. A uniform upper return bound cannot be
evaded by learning another prototype. -/
theorem macro_return_le_fiber_bound {X Z : Type*} [Fintype Z]
    (f : Z → X) (B : Matrix Z X ℝ) (U : Matrix X Z ℝ)
    (hU : IsStochastic U) (hs : FiberSupported f U) (x : X) (r : ℝ)
    (hr : ∀ z, f z = x → B z x ≤ r) : (U * B) x x ≤ r := by
  calc
    (U * B) x x = ∑ z, U x z * B z x := by rw [Matrix.mul_apply]
    _ ≤ ∑ z, U x z * r := by
      apply Finset.sum_le_sum
      intro z _
      by_cases hf : f z = x
      · exact mul_le_mul_of_nonneg_left (hr z hf) ((hU x).1 z)
      · simp [hs x z hf]
    _ = r := by rw [← Finset.sum_mul, (hU x).2, one_mul]

/-- A fiber-wise micro return obstruction bounds the actual lifted
prototype TV defect below, for every lawful choice of prototypes. -/
theorem prototype_defect_ge_fiber_escape {X Z : Type*} [Fintype X] [Fintype Z]
    [DecidableEq X] (f : Z → X) (B : Matrix Z X ℝ) (U : Matrix X Z ℝ)
    (hB : IsStochastic B) (hU : IsStochastic U) (hs : FiberSupported f U)
    (x : X) (r : ℝ) (hr : ∀ z, f z = x → B z x ≤ r) :
    1-r ≤ totalVariation ((U * (B * U)) x) (U x) := by
  rw [prototype_stability_eq_escape f B U hB hU hs x]
  have h := macro_return_le_fiber_bound f B U hU hs x r hr
  linarith

end GeoSBT
