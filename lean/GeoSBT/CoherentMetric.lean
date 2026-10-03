import GeoSBT.PathMetric
import GeoSBT.LikelihoodCost
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.Linarith

open scoped ENNReal

namespace GeoSBT

/-- Domination of reference distance by every edge passes to every finite
protocol. This prevents coarsening errors accumulating along a long route. -/
theorem Protocol.reference_le_cost {V : Type*} [PseudoEMetricSpace V]
    (w : V → V → ℝ≥0∞) (scale : ℝ≥0∞)
    (hw : ∀ x y, scale * edist x y ≤ w x y)
    {a b : V} (p : Protocol V a b) : scale * edist a b ≤ p.cost w := by
  induction p with
  | nil a => simp [Protocol.cost]
  | @cons a b c tail ih =>
    calc
      _ ≤ scale * (edist a b + edist b c) := mul_le_mul_right (edist_triangle a b c) scale
      _ = scale * edist a b + scale * edist b c := mul_add _ _ _
      _ ≤ w a b + tail.cost w := add_le_add (hw a b) ih
      _ = _ := rfl

/-- Bounds on the implemented edge costs bracket the actual infimum of
protocol costs. The upper bound uses the actual one-edge protocol. -/
theorem weightedEdist_reference_bracket {V : Type*} [PseudoEMetricSpace V]
    (w : V → V → ℝ≥0∞) (scale error : ℝ≥0∞)
    (hlo : ∀ x y, scale * edist x y ≤ w x y)
    (hhi : ∀ x y, w x y ≤ scale * edist x y + error) (a b : V) :
    scale * edist a b ≤ weightedEdist w a b ∧
      weightedEdist w a b ≤ scale * edist a b + error := by
  constructor
  · apply le_iInf
    intro p
    exact p.reference_le_cost w scale hlo
  · exact (weightedEdist_le_cost w (Protocol.cons (Protocol.nil b))).trans (by
      simpa [Protocol.cost] using hhi a b)

/-- Exponential probability brackets and an inactive floor give quantitative
cost brackets for the actual likelihood-cost definition. -/
theorem likelihood_cost_exp_bracket {p eta reference error : ℝ}
    (hp : 0 < p) (hfloor : eta ≤ p)
    (hlo : Real.exp (-(reference+error)) ≤ p)
    (hhi : p ≤ Real.exp (-reference)) :
    reference ≤ flooredLikelihoodCost p eta ∧
      flooredLikelihoodCost p eta ≤ reference+error := by
  have hloglo := Real.log_le_log (Real.exp_pos _) hlo
  have hloghi := Real.log_le_log hp hhi
  rw [Real.log_exp] at hloglo hloghi
  simp only [flooredLikelihoodCost, max_eq_left hfloor]
  constructor <;> linarith

end GeoSBT
