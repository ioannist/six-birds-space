import GeoSBT.Pythagoras
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

namespace GeoSBT

/-- Relative error below one controls the logarithm without an assumed
positive lower probability bound independent of stage. -/
theorem abs_log_le_relative_error {x e : ℝ} (hx : 0 < x)
    (he : 0 ≤ e) (he1 : e < 1) (herr : |x - 1| ≤ e) :
    |Real.log x| ≤ e / (1-e) := by
  have hden : 0 < 1-e := by linarith
  have hlo : 1-e ≤ x := by
    have h := (abs_le.mp herr).1
    linarith
  have hup : x-1 ≤ e := (abs_le.mp herr).2
  have hquot : e ≤ e/(1-e) := by
    apply (le_div_iff₀ hden).mpr
    nlinarith
  have hlogup : Real.log x ≤ e/(1-e) :=
    (Real.log_le_sub_one_of_pos hx).trans (hup.trans hquot)
  have hinv : 1/x ≤ 1/(1-e) := one_div_le_one_div_of_le hden hlo
  have hloginv := Real.log_le_sub_one_of_pos (div_pos (by norm_num : (0:ℝ)<1) hx)
  rw [Real.log_div (by norm_num : (1:ℝ) ≠ 0) hx.ne', Real.log_one] at hloginv
  have hfrac : 1/(1-e)-1 = e/(1-e) := by
    field_simp
    ring
  apply abs_le.mpr
  constructor
  · rw [zero_sub] at hloginv
    have hinvsub : 1/x-1 ≤ e/(1-e) := by
      rw [← hfrac]
      linarith
    linarith
  · exact hlogup

/-- A relative probability estimate controls the unfitted negative-log cost. -/
theorem log_cost_error {p g e : ℝ} (hp : 0 < p) (hg : 0 < g)
    (he : 0 ≤ e) (he1 : e < 1) (herr : |p/g-1| ≤ e) :
    |-Real.log p - (-Real.log g)| ≤ e/(1-e) := by
  have h := abs_log_le_relative_error (div_pos hp hg) he he1 herr
  rw [Real.log_div hp.ne' hg.ne'] at h
  have hid : -Real.log p - (-Real.log g) = -(Real.log p - Real.log g) := by ring
  rw [hid, abs_neg]
  exact h

/-- A uniform relative local-limit estimate passes through the centered
negative-log probability ratio. Gaussian applicability is a separate input. -/
theorem centered_log_ratio_error {p0 pz g0 gz e : ℝ}
    (hp0 : 0 < p0) (hpz : 0 < pz) (hg0 : 0 < g0) (hgz : 0 < gz)
    (he : 0 ≤ e) (he1 : e < 1)
    (h0 : |p0/g0-1| ≤ e) (hz : |pz/gz-1| ≤ e) :
    |Real.log (p0/pz) - Real.log (g0/gz)| ≤ 2*e/(1-e) := by
  have h0log := abs_log_le_relative_error (div_pos hp0 hg0) he he1 h0
  have hzlog := abs_log_le_relative_error (div_pos hpz hgz) he he1 hz
  have hid : Real.log (p0/pz) - Real.log (g0/gz) =
      Real.log (p0/g0) - Real.log (pz/gz) := by
    simp only [Real.log_div hp0.ne' hpz.ne', Real.log_div hg0.ne' hgz.ne',
      Real.log_div hp0.ne' hg0.ne', Real.log_div hpz.ne' hgz.ne']
    ring
  rw [hid]
  calc
    _ ≤ |Real.log (p0/g0)| + |Real.log (pz/gz)| := abs_sub _ _
    _ ≤ e/(1-e) + e/(1-e) := add_le_add h0log hzlog
    _ = 2*e/(1-e) := by ring

/-- Three uniform cost errors control the centered axis-separability residual.
The reference costs must actually obey the additive identity. -/
theorem cost_axis_residual_bound {cxy cx cy qxy qx qy e : ℝ}
    (hxy : |cxy-qxy| ≤ e) (hx : |cx-qx| ≤ e) (hy : |cy-qy| ≤ e)
    (hq : qxy=qx+qy) : |cxy-cx-cy| ≤ 3*e := by
  have hidentity : cxy-cx-cy = (cxy-qxy) - (cx-qx) - (cy-qy) := by
    rw [hq]
    ring
  rw [hidentity]
  calc
    _ ≤ |(cxy-qxy)-(cx-qx)| + |cy-qy| := abs_sub _ _
    _ ≤ (|cxy-qxy| + |cx-qx|) + |cy-qy| :=
      add_le_add (abs_sub _ _) le_rfl
    _ ≤ (e+e)+e := add_le_add (add_le_add hxy hx) hy
    _ = 3*e := by ring

end GeoSBT
