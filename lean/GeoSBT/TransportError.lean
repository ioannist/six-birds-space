import Mathlib.Analysis.Normed.Ring.Basic
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace GeoSBT

/-- Three approximate contraction transports control their actual loop product.
For orthogonal matrix transports use a compatible operator norm. -/
theorem triangle_transport_error {R : Type*} [NormedRing R]
    (a b c a' b' c' : R)
    (hb : ‖b‖ ≤ 1) (hc : ‖c‖ ≤ 1) (ha' : ‖a'‖ ≤ 1) (hb' : ‖b'‖ ≤ 1) :
    ‖a*b*c-a'*b'*c'‖ ≤ ‖a-a'‖ + ‖b-b'‖ + ‖c-c'‖ := by
  have hidentity : a*b*c-a'*b'*c' =
      (a-a')*b*c + a'*(b-b')*c + a'*b'*(c-c') := by noncomm_ring
  have hfirst : ‖(a-a')*b*c‖ ≤ ‖a-a'‖ := by
    calc
      _ ≤ ‖a-a'‖ * ‖b‖ * ‖c‖ := (norm_mul_le _ _).trans
        (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _))
      _ ≤ ‖a-a'‖ * 1 * 1 := mul_le_mul
        (mul_le_mul_of_nonneg_left hb (norm_nonneg _)) hc (norm_nonneg _)
        (by positivity)
      _ = _ := by ring
  have hsecond : ‖a'*(b-b')*c‖ ≤ ‖b-b'‖ := by
    calc
      _ ≤ ‖a'‖ * ‖b-b'‖ * ‖c‖ := (norm_mul_le _ _).trans
        (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _))
      _ ≤ 1 * ‖b-b'‖ * 1 := mul_le_mul
        (mul_le_mul_of_nonneg_right ha' (norm_nonneg _)) hc (norm_nonneg _)
        (by positivity)
      _ = _ := by ring
  have hthird : ‖a'*b'*(c-c')‖ ≤ ‖c-c'‖ := by
    calc
      _ ≤ ‖a'‖ * ‖b'‖ * ‖c-c'‖ := (norm_mul_le _ _).trans
        (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _))
      _ ≤ 1 * 1 * ‖c-c'‖ := mul_le_mul_of_nonneg_right
        (mul_le_mul ha' hb' (norm_nonneg _) (by norm_num)) (norm_nonneg _)
      _ = _ := by ring
  rw [hidentity]
  exact ((norm_add_le _ _).trans (add_le_add (norm_add_le _ _) le_rfl)).trans
    (add_le_add (add_le_add hfirst hsecond) hthird)

/-- A quantified angle error must be divided by the shrinking area. An
unnormalized vanishing error does not supply curvature consistency. -/
theorem normalized_angle_error {angle area error : ℝ} (ha : 0 < area)
    (he : |angle-area| ≤ error) : |angle/area-1| ≤ error/area := by
  have hidentity : angle/area-1 = (angle-area)/area := by field_simp
  rw [hidentity, abs_div, abs_of_pos ha]
  exact div_le_div_of_nonneg_right he ha.le

end GeoSBT
