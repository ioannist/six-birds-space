import GeoSBT.PathMetric
import Mathlib.Topology.MetricSpace.Basic

open scoped ENNReal

noncomputable def separation_quotient_metric (α : Type) [PseudoMetricSpace α] :
    MetricSpace (SeparationQuotient α) := by
  infer_instance

/-- Distance zero implies equality in the separation quotient. -/
theorem separation_quotient_dist_eq_zero {α : Type} [PseudoMetricSpace α]
    (x y : SeparationQuotient α) : dist x y = 0 ↔ x = y := by
  exact dist_eq_zero

/-- Extended separation remains valid for disconnected weighted graphs. -/
noncomputable def separation_quotient_emetric (α : Type*) [PseudoEMetricSpace α] :
    EMetricSpace (SeparationQuotient α) := inferInstance

/-- The quotient preserves the source distance, including infinite distances. -/
theorem separation_quotient_edist_mk {α : Type*} [PseudoEMetricSpace α] (x y : α) :
    edist (SeparationQuotient.mk x) (SeparationQuotient.mk y) = edist x y := by
  rfl

/-- Two classes agree exactly when their extended distance vanishes. -/
theorem separation_quotient_edist_eq_zero {α : Type*} [PseudoEMetricSpace α]
    (x y : SeparationQuotient α) : edist x y = 0 ↔ x = y := by
  exact edist_eq_zero

namespace GeoSBT

/-- Apply the separation construction to the actual weighted protocol metric. -/
noncomputable def weightedSeparationMetric {V : Type*} (w : V → V → ℝ≥0∞)
    (hw : ∀ x y, w x y = w y x) :
    letI := weightedPseudoEMetricSpace w hw
    EMetricSpace (SeparationQuotient V) := by
  letI := weightedPseudoEMetricSpace w hw
  infer_instance

/-- Quotient readout is the weighted protocol infimum, with no finite-distance premise. -/
theorem weightedSeparation_edist_mk {V : Type*} (w : V → V → ℝ≥0∞)
    (hw : ∀ x y, w x y = w y x) (a b : V) :
    letI := weightedPseudoEMetricSpace w hw
    edist (SeparationQuotient.mk a) (SeparationQuotient.mk b) = weightedEdist w a b := by
  letI := weightedPseudoEMetricSpace w hw
  rfl

end GeoSBT
