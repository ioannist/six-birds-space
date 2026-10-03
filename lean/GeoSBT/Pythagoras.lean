import Mathlib.Analysis.InnerProductSpace.Basic

open scoped InnerProductSpace

/-- Pythagoras in a real inner product space: orthogonal vectors give squared-norm additivity. -/
theorem pythagoras_real {V : Type} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (x y : V)
    (h : ⟪x, y⟫_ℝ = 0) : ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ := by
  simpa using (norm_add_sq_eq_norm_sq_add_norm_sq_real (x := x) (y := y) h)


namespace GeoSBT

/-- On nonnegative readouts, squared error is controlled by squared-cost error. -/
theorem squared_readout_error_le {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (x - y) ^ 2 ≤ |x ^ 2 - y ^ 2| := by
  by_cases h : x ≤ y
  · have hs : x ^ 2 - y ^ 2 ≤ 0 := by nlinarith
    rw [abs_of_nonpos hs]
    nlinarith [mul_nonneg hx (sub_nonneg.mpr h)]
  · have hle : y ≤ x := le_of_not_ge h
    have hs : 0 ≤ x ^ 2 - y ^ 2 := by nlinarith
    rw [abs_of_nonneg hs]
    nlinarith [mul_nonneg hy (sub_nonneg.mpr hle)]

/-- A uniform nonnegative quadratic-cost estimate gives a square-root readout estimate.
This theorem requires the cost estimate; it does not assume that an experimental
RMS fit supplies one or that the resulting pairwise readout is itself a metric. -/
theorem quadratic_cost_readout_error {q r epsilon : ℝ} (hq : 0 ≤ q) (hr : 0 ≤ r)
    (herr : |q - r ^ 2| ≤ epsilon) : |Real.sqrt q - r| ≤ Real.sqrt epsilon := by
  apply Real.abs_le_sqrt
  have h := squared_readout_error_le (Real.sqrt_nonneg q) hr
  rw [Real.sq_sqrt hq] at h
  exact h.trans herr

end GeoSBT
