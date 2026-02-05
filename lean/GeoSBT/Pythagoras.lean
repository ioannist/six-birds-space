import Mathlib.Analysis.InnerProductSpace.Basic

open scoped InnerProductSpace

/-- Pythagoras in a real inner product space: orthogonal vectors give squared-norm additivity. -/
theorem pythagoras_real {V : Type} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (x y : V)
    (h : ⟪x, y⟫_ℝ = 0) : ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ := by
  simpa using (norm_add_sq_eq_norm_sq_add_norm_sq_real (x := x) (y := y) h)
