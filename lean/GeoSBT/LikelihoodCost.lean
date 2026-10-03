import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.ENNReal.Real

namespace GeoSBT

/-- The implemented cost uses a probability floor, not additive smoothing. -/
noncomputable def flooredLikelihoodCost (p eta : ℝ) : ℝ := -Real.log (max p eta)

/-- Probability-domain hypotheses ensure nonnegative costs without truncation. -/
theorem flooredLikelihoodCost_nonneg {p eta : ℝ} (hp : p ≤ 1)
    (heta : 0 < eta) (heta_one : eta ≤ 1) : 0 ≤ flooredLikelihoodCost p eta := by
  exact neg_nonneg.mpr (Real.log_nonpos
    (le_trans heta.le (le_max_right p eta)) (max_le hp heta_one))

/-- Encoding a lawful cost as an extended nonnegative number preserves its value. -/
theorem flooredLikelihoodCost_encoding {p eta : ℝ} (hp : p ≤ 1)
    (heta : 0 < eta) (heta_one : eta ≤ 1) :
    (ENNReal.ofReal (flooredLikelihoodCost p eta)).toReal = flooredLikelihoodCost p eta := by
  exact ENNReal.toReal_ofReal (flooredLikelihoodCost_nonneg hp heta heta_one)

end GeoSBT
