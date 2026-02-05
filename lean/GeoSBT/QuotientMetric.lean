import Mathlib.Topology.MetricSpace.Basic

instance separation_quotient_metric (α : Type) [PseudoMetricSpace α] :
    MetricSpace (SeparationQuotient α) := by
  infer_instance

/-- Distance zero implies equality in the separation quotient. -/
theorem separation_quotient_dist_eq_zero {α : Type} [PseudoMetricSpace α]
    (x y : SeparationQuotient α) : dist x y = 0 ↔ x = y := by
  simpa using (dist_eq_zero : dist x y = 0 ↔ x = y)
