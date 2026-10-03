import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

open scoped InnerProductSpace

namespace GeoSBT

/-- Every real inner-product space obeys the four-point squared-distance inequality. -/
theorem hilbert_quadrilateral {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] (a b c d : V) :
    ‖a - b‖ ^ 2 + ‖c - d‖ ^ 2 ≤
      ‖a - c‖ ^ 2 + ‖a - d‖ ^ 2 + ‖b - c‖ ^ 2 + ‖b - d‖ ^ 2 := by
  have hidentity :
      (‖a - c‖ ^ 2 + ‖a - d‖ ^ 2 + ‖b - c‖ ^ 2 + ‖b - d‖ ^ 2) -
      (‖a - b‖ ^ 2 + ‖c - d‖ ^ 2) = ‖a + b - c - d‖ ^ 2 := by
    simp only [norm_sub_sq_real, norm_add_sq_real, inner_add_left,
      inner_sub_left, real_inner_comm b a,
      real_inner_comm c a, real_inner_comm c b]
    ring
  nlinarith [sq_nonneg ‖a + b - c - d‖]

/-- A strict interval-certified quadrilateral gap rules out every Hilbert fit
with the stated uniform additive error, in any dimension. Bounds on the actual
six distances and the proposed fit error are explicit premises. -/
theorem no_hilbert_uniform_approximation {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] (D : Fin 4 → Fin 4 → ℝ)
    (la lb uac uad ubc ubd delta : ℝ)
    (hla : la ≤ D 0 1) (hlb : lb ≤ D 2 3)
    (huac : D 0 2 ≤ uac) (huad : D 0 3 ≤ uad)
    (hubc : D 1 2 ≤ ubc) (hubd : D 1 3 ≤ ubd)
    (hna : delta ≤ la) (hnb : delta ≤ lb)
    (hgap : (uac + delta)^2 + (uad + delta)^2 +
      (ubc + delta)^2 + (ubd + delta)^2 < (la - delta)^2 + (lb - delta)^2)
    (p : Fin 4 → V) (hfit : ∀ i j, |‖p i - p j‖ - D i j| ≤ delta) : False := by
  have hlow01 : la - delta ≤ ‖p 0 - p 1‖ := by
    have h := (abs_le.mp (hfit 0 1)).1
    linarith
  have hlow23 : lb - delta ≤ ‖p 2 - p 3‖ := by
    have h := (abs_le.mp (hfit 2 3)).1
    linarith
  have hu02 : ‖p 0 - p 2‖ ≤ uac + delta := by
    have h := (abs_le.mp (hfit 0 2)).2
    linarith
  have hu03 : ‖p 0 - p 3‖ ≤ uad + delta := by
    have h := (abs_le.mp (hfit 0 3)).2
    linarith
  have hu12 : ‖p 1 - p 2‖ ≤ ubc + delta := by
    have h := (abs_le.mp (hfit 1 2)).2
    linarith
  have hu13 : ‖p 1 - p 3‖ ≤ ubd + delta := by
    have h := (abs_le.mp (hfit 1 3)).2
    linarith
  have hs01 := pow_le_pow_left₀ (sub_nonneg.mpr hna) hlow01 2
  have hs23 := pow_le_pow_left₀ (sub_nonneg.mpr hnb) hlow23 2
  have hs02 := pow_le_pow_left₀ (norm_nonneg (p 0 - p 2)) hu02 2
  have hs03 := pow_le_pow_left₀ (norm_nonneg (p 0 - p 3)) hu03 2
  have hs12 := pow_le_pow_left₀ (norm_nonneg (p 1 - p 2)) hu12 2
  have hs13 := pow_le_pow_left₀ (norm_nonneg (p 1 - p 3)) hu13 2
  have hquad := hilbert_quadrilateral (p 0) (p 1) (p 2) (p 3)
  linarith

/-- A uniform edge-cost bracket on a six-cycle yields a positive four-point
gap at additive error c/4. The relative bracket hypothesis is explicit. -/
theorem gasket_quadrilateral_gap {c e : ℝ} (hc : 0 < c) (he : e ≤ c / 26) :
    2 * (c + c / 4)^2 + 2 * (2*c + c / 4)^2 <
      2 * (3*(c-e) - c/4)^2 := by
  have hbound : 137*c/52 ≤ 3*(c-e)-c/4 := by linarith
  have hp : 0 ≤ 137*c/52 := by linarith
  have hs := pow_le_pow_left₀ hp hbound 2
  nlinarith [sq_pos_of_pos hc]

/-- The fixed limiting-address witness has a strict Hilbert gap at error 1/16.
The graph-to-limit identification is a separate applicability obligation. -/
theorem gasket_limit_quadrilateral_gap :
    2 * ((1:ℝ)/4 + 1/16)^2 + 2 * ((1:ℝ)/2 + 1/16)^2 <
      2 * ((3:ℝ)/4 - 1/16)^2 := by
  norm_num

/-- The recursive-cell cost bracket satisfies the needed relative bound once
the unassigned cell volume is at least 42. No numerical log fit is assumed. -/
theorem gasket_log_bracket {v : ℝ} (hv : 42 ≤ v) :
    0 < Real.log (4*v) ∧ Real.log (v/(v-3)) ≤ Real.log (4*v)/26 := by
  have hv0 : 0 < v := by linarith
  have hv3 : 0 < v-3 := by linarith
  have hlog := Real.log_le_sub_one_of_pos (div_pos hv0 hv3)
  have hfrac : v/(v-3)-1 = 3/(v-3) := by
    field_simp
    ring
  rw [hfrac] at hlog
  have hdiv : 3/(v-3) ≤ (1:ℝ)/13 := by
    apply (div_le_iff₀ hv3).mpr
    linarith
  have hexp : Real.exp (2:ℝ) < 9 := by
    have h := Real.exp_one_lt_three
    have heq : Real.exp (2:ℝ) = Real.exp 1 * Real.exp 1 := by
      rw [show (2:ℝ) = 1+1 by norm_num, Real.exp_add]
    rw [heq]
    nlinarith [Real.exp_pos (1:ℝ)]
  have hc : 2 ≤ Real.log (4*v) := by
    apply (Real.le_log_iff_exp_le (by linarith : 0 < 4*v)).mpr
    linarith
  constructor
  · linarith
  · linarith

/-- The same obstruction bracket applies at a fixed stage with interface flux
in (0,1], including the exact five-step flux 11905/16384. -/
theorem gasket_staged_log_bracket {v k : ℝ} (hv : 42 ≤ v)
    (hk0 : 0 < k) (hk1 : k ≤ 1) :
    0 < Real.log (v/k) ∧ Real.log (v/(v-3)) ≤ Real.log (v/k)/26 := by
  have hv0 : 0 < v := by linarith
  have hv3 : 0 < v-3 := by linarith
  have hlog := Real.log_le_sub_one_of_pos (div_pos hv0 hv3)
  have hfrac : v/(v-3)-1 = 3/(v-3) := by
    field_simp
    ring
  rw [hfrac] at hlog
  have hdiv : 3/(v-3) ≤ (1:ℝ)/13 := by
    apply (div_le_iff₀ hv3).mpr
    linarith
  have hq : v ≤ v/k := by
    apply (le_div_iff₀ hk0).mpr
    simpa using mul_le_mul_of_nonneg_left hk1 hv0.le
  have hexp : Real.exp (2:ℝ) < 9 := by
    have h := Real.exp_one_lt_three
    have heq : Real.exp (2:ℝ) = Real.exp 1 * Real.exp 1 := by
      rw [show (2:ℝ) = 1+1 by norm_num, Real.exp_add]
    rw [heq]
    nlinarith [Real.exp_pos (1:ℝ)]
  have hc : 2 ≤ Real.log (v/k) := by
    apply (Real.le_log_iff_exp_le (div_pos hv0 hk0)).mpr
    linarith
  constructor
  · linarith
  · linarith

end GeoSBT
