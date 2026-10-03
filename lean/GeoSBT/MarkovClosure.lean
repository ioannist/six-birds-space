import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Order.BigOperators.Group.Finset

open scoped BigOperators

namespace GeoSBT

/-- A finite probability vector, with no positivity or normalization hidden. -/
def IsDistribution {I : Type*} [Fintype I] (mu : I → ℝ) : Prop :=
  (∀ i, 0 ≤ mu i) ∧ ∑ i, mu i = 1

/-- Row stochasticity also applies to rectangular packaging and lifting maps. -/
def IsStochastic {I J : Type*} [Fintype J] (K : Matrix I J ℝ) : Prop :=
  ∀ i, IsDistribution (K i)

theorem stochastic_mul {I J K : Type*} [Fintype J] [Fintype K]
    (A : Matrix I J ℝ) (B : Matrix J K ℝ)
    (hA : IsStochastic A) (hB : IsStochastic B) : IsStochastic (A * B) := by
  intro i
  constructor
  · intro k
    exact Finset.sum_nonneg fun j _ => mul_nonneg ((hA i).1 j) ((hB j).1 k)
  · simp only [Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, (hB _).2, mul_one]
    exact (hA i).2

/-- Deterministic quotient labels induce a stochastic coarse map. -/
noncomputable def coarseMatrix {Z X : Type*} (f : Z → X) : Matrix Z X ℝ := by
  classical
  exact fun z x => if f z = x then 1 else 0

theorem coarseMatrix_stochastic {Z X : Type*} [Fintype X] (f : Z → X) :
    IsStochastic (coarseMatrix f) := by
  classical
  intro z
  constructor
  · intro x
    simp only [coarseMatrix]
    split_ifs <;> norm_num
  · simp [coarseMatrix]

theorem stochastic_identity {Z : Type*} [Fintype Z] [DecidableEq Z] :
    IsStochastic (1 : Matrix Z Z ℝ) := by
  intro z
  constructor
  · intro x
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · simp [Matrix.one_apply]

theorem stochastic_power {Z : Type*} [Fintype Z] [DecidableEq Z]
    (P : Matrix Z Z ℝ) (hP : IsStochastic P) (tau : ℕ) : IsStochastic (P ^ tau) := by
  induction tau with
  | zero => simpa using (stochastic_identity (Z := Z))
  | succ tau ih => simpa [pow_succ] using stochastic_mul (P ^ tau) P ih hP

/-- The actual closure and macro kernel preserve distributions by construction. -/
theorem closure_and_macro_stochastic {Z X : Type*} [Fintype Z] [Fintype X]
    [DecidableEq Z] (P : Matrix Z Z ℝ) (f : Z → X) (U : Matrix X Z ℝ)
    (hP : IsStochastic P) (hU : IsStochastic U) (tau : ℕ) :
    IsStochastic (P ^ tau * coarseMatrix f * U) ∧
    IsStochastic (U * P ^ tau * coarseMatrix f) := by
  have hC := coarseMatrix_stochastic f
  have hPt := stochastic_power P hP tau
  exact ⟨stochastic_mul _ U (stochastic_mul _ _ hPt hC) hU,
         stochastic_mul _ _ (stochastic_mul U _ hU hPt) hC⟩

/-- A factorization of the actual closure defect, valid without support hypotheses. -/
theorem closure_defect_factorization {Z X : Type*} [Fintype Z] [Fintype X]
    (B : Matrix Z X ℝ) (U : Matrix X Z ℝ) :
    (B * U) * (B * U) - B * U = (B * (U * B) - B) * U := by
  rw [Matrix.sub_mul]
  simp only [Matrix.mul_assoc]

/-- Prototypes carry mass only on their own quotient fibers. -/
def FiberSupported {Z X : Type*} (f : Z → X) (U : Matrix X Z ℝ) : Prop :=
  ∀ x z, f z ≠ x → U x z = 0

/-- Normalized fiber-supported prototypes are a right inverse to packaging. -/
theorem lift_coarse_identity {Z X : Type*} [Fintype Z] [DecidableEq X]
    (f : Z → X) (U : Matrix X Z ℝ) (hU : IsStochastic U)
    (hs : FiberSupported f U) : U * coarseMatrix f = (1 : Matrix X X ℝ) := by
  classical
  ext x y
  rw [Matrix.mul_apply, Matrix.one_apply]
  by_cases hxy : x = y
  · subst y
    rw [if_pos rfl]
    calc
      ∑ z, U x z * coarseMatrix f z x = ∑ z, U x z := by
        apply Finset.sum_congr rfl
        intro z _
        by_cases hf : f z = x
        · simp [coarseMatrix, hf]
        · simp [coarseMatrix, hf, hs x z hf]
      _ = 1 := (hU x).2
  · simp only [if_neg hxy]
    apply Finset.sum_eq_zero
    intro z _
    by_cases hf : f z = y
    · have hx : f z ≠ x := by simpa [hf] using Ne.symm hxy
      simp [coarseMatrix, hf, hs x z hx]
    · simp [coarseMatrix, hf]

noncomputable def push {I J : Type*} [Fintype I] (mu : I → ℝ)
    (K : Matrix I J ℝ) : J → ℝ := fun j => ∑ i, mu i * K i j

noncomputable def totalVariation {I : Type*} [Fintype I] (mu nu : I → ℝ) : ℝ :=
  (∑ i, |mu i - nu i|) / 2

/-- The row supremum controls every probability input, not just sampled inputs. -/
theorem totalVariation_push_row_bound {I J : Type*} [Fintype I] [Fintype J]
    (mu : I → ℝ) (A B : Matrix I J ℝ) (D : ℝ)
    (hmu : IsDistribution mu) (hrow : ∀ i, totalVariation (A i) (B i) ≤ D) :
    totalVariation (push mu A) (push mu B) ≤ D := by
  have hpoint : ∀ j, |push mu A j - push mu B j| ≤
      ∑ i, mu i * |A i j - B i j| := by
    intro j
    simp only [push, ← Finset.sum_sub_distrib, ← mul_sub]
    calc
      |∑ i, mu i * (A i j - B i j)| ≤ ∑ i, |mu i * (A i j - B i j)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ i, mu i * |A i j - B i j| := by
        simp only [abs_mul, abs_of_nonneg (hmu.1 _)]
  have hsum : (∑ j, |push mu A j - push mu B j|) ≤ 2 * D := by
    calc
      ∑ j, |push mu A j - push mu B j| ≤ ∑ j, ∑ i, mu i * |A i j - B i j| :=
        Finset.sum_le_sum fun j _ => hpoint j
      _ = ∑ i, mu i * ∑ j, |A i j - B i j| := by
        rw [Finset.sum_comm]
        simp only [Finset.mul_sum]
      _ ≤ ∑ i, mu i * (2 * D) := by
        apply Finset.sum_le_sum
        intro i _
        apply mul_le_mul_of_nonneg_left _ (hmu.1 i)
        have hi := hrow i
        unfold totalVariation at hi
        linarith
      _ = 2 * D := by rw [← Finset.sum_mul, hmu.2, one_mul]
  unfold totalVariation
  linarith

/-- Point masses certify that the row bound is sharp over the probability simplex. -/
theorem totalVariation_extreme_point_iff {I J : Type*} [Fintype I] [Fintype J]
    (A B : Matrix I J ℝ) (D : ℝ) :
    (∀ mu, IsDistribution mu → totalVariation (push mu A) (push mu B) ≤ D) ↔
    (∀ i, totalVariation (A i) (B i) ≤ D) := by
  classical
  constructor
  · intro h i
    let mu : I → ℝ := fun k => if k = i then 1 else 0
    have hmu : IsDistribution mu := by
      constructor
      · intro k
        simp only [mu]
        split_ifs <;> norm_num
      · simp [mu]
    have ha : push mu A = A i := by ext j; simp [push, mu]
    have hb : push mu B = B i := by ext j; simp [push, mu]
    simpa only [ha, hb] using h mu hmu
  · intro h mu hmu
    exact totalVariation_push_row_bound mu A B D hmu h

theorem push_distribution {I J : Type*} [Fintype I] [Fintype J]
    (mu : I → ℝ) (K : Matrix I J ℝ) (hmu : IsDistribution mu)
    (hK : IsStochastic K) : IsDistribution (push mu K) := by
  have h := stochastic_mul (fun (_ : Unit) i => mu i) K (fun _ => hmu) hK
  exact h ()

theorem push_push {I J K : Type*} [Fintype I] [Fintype J]
    (mu : I → ℝ) (A : Matrix I J ℝ) (B : Matrix J K ℝ) :
    push (push mu A) B = push mu (A * B) := by
  ext k
  exact congrArg (fun M => M () k) (Matrix.mul_assoc (fun (_ : Unit) i => mu i) A B)

theorem totalVariation_self {I : Type*} [Fintype I] (mu : I → ℝ) :
    totalVariation mu mu = 0 := by simp [totalVariation]

theorem totalVariation_triangle {I : Type*} [Fintype I] (mu nu rho : I → ℝ) :
    totalVariation mu rho ≤ totalVariation mu nu + totalVariation nu rho := by
  have h : (∑ i, |mu i - rho i|) ≤ (∑ i, |mu i - nu i|) + ∑ i, |nu i - rho i| := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun i _ => abs_sub_le (mu i) (nu i) (rho i)
  unfold totalVariation
  linarith

/-- A one-versus-two-step defect gives a finite-horizon bound with its actual loss. -/
theorem closure_repetition_bound {Z : Type*} [Fintype Z] [DecidableEq Z]
    (E : Matrix Z Z ℝ) (mu : Z → ℝ) (D : ℝ) (hE : IsStochastic E)
    (hmu : IsDistribution mu)
    (hdef : ∀ z, totalVariation ((E ^ 2) z) (E z) ≤ D) (k : ℕ) :
    totalVariation (push mu (E ^ (k + 1))) (push mu E) ≤ (k : ℝ) * D := by
  induction k with
  | zero => simp [totalVariation_self]
  | succ k ih =>
    have hstep := totalVariation_push_row_bound (push mu (E ^ k)) (E ^ 2) E D
      (push_distribution mu (E ^ k) hmu (stochastic_power E hE k)) hdef
    rw [push_push, push_push] at hstep
    have hpow2 : E ^ k * E ^ 2 = E ^ (k + 2) := (pow_add E k 2).symm
    have hpow1 : E ^ k * E = E ^ (k + 1) := by simp [pow_succ]
    rw [hpow2, hpow1] at hstep
    have htri := totalVariation_triangle (push mu (E ^ (k + 2)))
      (push mu (E ^ (k + 1))) (push mu E)
    have hcast : ((k + 1 : ℕ) : ℝ) = (k : ℝ) + 1 := by simp
    simp only [Nat.add_assoc, hcast]
    linarith

/-- Exact signed L1 accounting for lifting with disjoint row supports. -/
theorem disjoint_push_l1 {X Z : Type*} [Fintype X] [Fintype Z]
    (a : X → ℝ) (U : Matrix X Z ℝ)
    (hd : ∀ z x y, U x z ≠ 0 → U y z ≠ 0 → x = y) :
    (∑ z, |push a U z|) = ∑ x, |a x| * ∑ z, |U x z| := by
  classical
  have hcol : ∀ z, |push a U z| = ∑ x, |a x| * |U x z| := by
    intro z
    by_cases h : ∃ x, U x z ≠ 0
    · obtain ⟨x, hx⟩ := h
      have hzero : ∀ y, y ≠ x → U y z = 0 := by
        intro y hy
        by_contra hne
        exact hy (hd z y x hne hx)
      have hl : (∑ y, a y * U y z) = a x * U x z := by
        apply Finset.sum_eq_single x
        · intro y _ hy
          simp [hzero y hy]
        · simp
      have hr : (∑ y, |a y| * |U y z|) = |a x| * |U x z| := by
        apply Finset.sum_eq_single x
        · intro y _ hy
          simp [hzero y hy]
        · simp
      rw [push, hl, hr, abs_mul]
    · have hzero : ∀ x, U x z = 0 := by simpa using h
      simp [push, hzero]
  simp_rw [hcol]
  rw [Finset.sum_comm]
  simp only [Finset.mul_sum]

/-- Fiber-supported normalized prototypes preserve every signed L1 discrepancy. -/
theorem fiber_push_l1_isometry {X Z : Type*} [Fintype X] [Fintype Z]
    (f : Z → X) (U : Matrix X Z ℝ) (hU : IsStochastic U)
    (hs : FiberSupported f U) (a : X → ℝ) :
    (∑ z, |push a U z|) = ∑ x, |a x| := by
  have hd : ∀ z x y, U x z ≠ 0 → U y z ≠ 0 → x = y := by
    intro z x y hx hy
    have hfx : f z = x := by
      by_contra hn
      exact hx (hs x z hn)
    have hfy : f z = y := by
      by_contra hn
      exact hy (hs y z hn)
    exact hfx.symm.trans hfy
  rw [disjoint_push_l1 a U hd]
  simp only [abs_of_nonneg ((hU _).1 _), (hU _).2, mul_one]

theorem totalVariation_fiber_lift {X Z : Type*} [Fintype X] [Fintype Z]
    (f : Z → X) (U : Matrix X Z ℝ) (hU : IsStochastic U)
    (hs : FiberSupported f U) (a b : X → ℝ) :
    totalVariation (push a U) (push b U) = totalVariation a b := by
  have h := fiber_push_l1_isometry f U hU hs (fun x => a x - b x)
  unfold totalVariation
  simp only [push, sub_mul, Finset.sum_sub_distrib] at h
  exact congrArg (fun r : ℝ => r / 2) h

/-- TV from a probability row to its own point mass is exactly its escape mass. -/
theorem totalVariation_point_eq_escape {X : Type*} [Fintype X] [DecidableEq X]
    (p : X → ℝ) (hp : IsDistribution p) (x : X) :
    totalVariation p ((1 : Matrix X X ℝ) x) = 1 - p x := by
  have hx : p x ≤ 1 := by
    have h := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) => hp.1 j)
      (Finset.mem_univ x)
    simpa only [hp.2] using h
  have hentry : ∀ j, |p j - (1 : Matrix X X ℝ) x j| =
      p j + if x = j then 1 - 2 * p x else 0 := by
    intro j
    by_cases h : x = j
    · subst j
      simp only [Matrix.one_apply, if_true]
      rw [abs_of_nonpos (sub_nonpos.mpr hx)]
      ring
    · simp [h, abs_of_nonneg (hp.1 j)]
  unfold totalVariation
  simp_rw [hentry]
  rw [Finset.sum_add_distrib]
  simp only [hp.2, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  ring

/-- Exact stability of the actual lifted closure, with all fiber hypotheses exposed. -/
theorem prototype_stability_eq_escape {X Z : Type*} [Fintype X] [Fintype Z]
    [DecidableEq X] (f : Z → X) (B : Matrix Z X ℝ) (U : Matrix X Z ℝ)
    (hB : IsStochastic B) (hU : IsStochastic U) (hs : FiberSupported f U) (x : X) :
    totalVariation ((U * (B * U)) x) (U x) = 1 - (U * B) x x := by
  have hp : push ((1 : Matrix X X ℝ) x) U = U x := by
    ext z
    simp [push, Matrix.one_apply]
  rw [← Matrix.mul_assoc]
  change totalVariation (push ((U * B) x) U) (U x) = _
  rw [← hp, totalVariation_fiber_lift f U hU hs]
  exact totalVariation_point_eq_escape ((U * B) x) (stochastic_mul U B hU hB x) x

end GeoSBT
