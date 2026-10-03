import Mathlib.Algebra.Group.Basic
import Mathlib.Logic.Function.Basic

namespace GeoSBT

/-- A connection coefficient transports the shifted frame back to the current
fiber. This is an operator on one common space of group-valued frame fields. -/
def covariantShift {X G : Type*} [Group G] (a : X → G) (u : X → X)
    (s : X → G) : X → G := fun x => a x * s (u x)

/-- Ratio of the two connection routes around a commuting translation square. -/
def plaquetteHolonomy {X G : Type*} [Group G] (a b : X → G)
    (u v : X → X) (x : X) : G :=
  (a x * b (u x)) * (b x * a (v x))⁻¹

/-- Common-space covariant shifts commute exactly when every square holonomy
is trivial. This holds even for an abelian coefficient group. -/
theorem covariantShift_commute_iff {X G : Type*} [Group G]
    (a b : X → G) (u v : X → X) (huv : ∀ x, v (u x) = u (v x)) :
    Function.Commute (covariantShift a u) (covariantShift b v) ↔
      ∀ x, plaquetteHolonomy a b u v x = 1 := by
  constructor
  · intro h x
    have hcoeff := congrFun (h (fun _ => 1)) x
    simp only [covariantShift, mul_one] at hcoeff
    simp only [plaquetteHolonomy, mul_inv_eq_one]
    exact hcoeff
  · intro h s
    funext x
    have hcoeff := h x
    simp only [plaquetteHolonomy, mul_inv_eq_one] at hcoeff
    simp only [covariantShift]
    rw [← mul_assoc, ← mul_assoc, hcoeff, huv x]

def connectionGauge {X G : Type*} [Group G] (q a : X → G) (u : X → X) : X → G :=
  fun x => q x * a x * (q (u x))⁻¹

def frameGauge {X G : Type*} [Group G] (q s : X → G) : X → G :=
  fun x => q x * s x

/-- The transformed shift intertwines the frame change exactly. -/
theorem covariantShift_gauge {X G : Type*} [Group G]
    (q a : X → G) (u : X → X) (s : X → G) :
    covariantShift (connectionGauge q a u) u (frameGauge q s) =
      frameGauge q (covariantShift a u s) := by
  funext x
  simp [covariantShift, connectionGauge, frameGauge, mul_assoc]

/-- A loop changes by conjugation at its base fiber. No common global frame
or orientation choice is assumed. -/
theorem plaquetteHolonomy_gauge {X G : Type*} [Group G]
    (q a b : X → G) (u v : X → X) (huv : ∀ x, v (u x) = u (v x)) (x : X) :
    plaquetteHolonomy (connectionGauge q a u) (connectionGauge q b v) u v x =
      q x * plaquetteHolonomy a b u v x * (q x)⁻¹ := by
  simp [plaquetteHolonomy, connectionGauge, huv x, mul_inv_rev, mul_assoc]

/-- A triangle loop is trivial exactly when the two routes between its
endpoints agree. This is route dependence, not noncommutation of two matrices. -/
theorem triangleHolonomy_eq_one_iff {G : Type*} [Group G] (a b c : G) :
    a * b * c⁻¹ = 1 ↔ a * b = c := by
  exact mul_inv_eq_one

end GeoSBT
