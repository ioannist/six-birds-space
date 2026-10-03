import Mathlib.Combinatorics.SimpleGraph.Metric
import Mathlib.Data.ENNReal.Operations
import Mathlib.Topology.EMetricSpace.Defs

open SimpleGraph
open scoped ENNReal

/-- Triangle inequality for graph distance (path cost). -/
theorem graph_edist_triangle {V : Type} (G : SimpleGraph V) (a b c : V) :
    G.edist a c ≤ G.edist a b + G.edist b c := by
  simpa using G.edist_triangle (u := a) (v := b) (w := c)

/-! Weighted protocol distances. An infinite edge weight represents an absent
transition. The empty protocol is included, and no finiteness or separation
assumption is hidden in the construction. -/

namespace GeoSBT

/-- Finite vertex protocols with prescribed endpoints (cycles are permitted). -/
inductive Protocol (V : Type*) : V → V → Type _
  | nil (a : V) : Protocol V a a
  | cons {a b c : V} (tail : Protocol V b c) : Protocol V a c

namespace Protocol

variable {V : Type*} {a b c : V}

noncomputable def cost {a b : V} (w : V → V → ℝ≥0∞) : Protocol V a b → ℝ≥0∞
  | .nil _ => 0
  | @cons _ x y _ tail => w x y + cost w tail

def append {a b c : V} : Protocol V a b → Protocol V b c → Protocol V a c
  | .nil _, q => q
  | .cons tail, q => .cons (append tail q)

@[simp] theorem cost_append (w : V → V → ℝ≥0∞)
    (p : Protocol V a b) (q : Protocol V b c) :
    cost w (append p q) = cost w p + cost w q := by
  induction p with
  | nil => simp [append, cost]
  | cons tail ih => simp [append, cost, ih, add_assoc]

def reverse {a b : V} : Protocol V a b → Protocol V b a
  | .nil x => .nil x
  | .cons tail => append (reverse tail) (.cons (.nil _))

@[simp] theorem cost_reverse (w : V → V → ℝ≥0∞) (hw : ∀ x y, w x y = w y x)
    (p : Protocol V a b) : cost w (reverse p) = cost w p := by
  induction p with
  | nil => rfl
  | @cons x y z tail ih => simp [reverse, cost, ih, hw y x, add_comm]

end Protocol

/-- Infimum of nonnegative weighted finite protocol costs, possibly infinity. -/
noncomputable def weightedEdist {V : Type*} (w : V → V → ℝ≥0∞) (a b : V) : ℝ≥0∞ :=
  ⨅ p : Protocol V a b, p.cost w

theorem weightedEdist_le_cost {V : Type*} (w : V → V → ℝ≥0∞)
    {a b : V} (p : Protocol V a b) : weightedEdist w a b ≤ p.cost w :=
  iInf_le _ p

@[simp] theorem weightedEdist_self {V : Type*} (w : V → V → ℝ≥0∞) (a : V) :
    weightedEdist w a a = 0 := by
  apply le_antisymm _ (zero_le _)
  simpa [Protocol.cost] using weightedEdist_le_cost w (Protocol.nil a)

/-- Weighted triangle inequality by actual protocol concatenation. -/
theorem weightedEdist_triangle {V : Type*} (w : V → V → ℝ≥0∞) (a b c : V) :
    weightedEdist w a c ≤ weightedEdist w a b + weightedEdist w b c := by
  apply ENNReal.le_iInf_add_iInf
  intro p q
  simpa using weightedEdist_le_cost w (p.append q)

theorem weightedEdist_symm {V : Type*} (w : V → V → ℝ≥0∞)
    (hw : ∀ x y, w x y = w y x) (a b : V) : weightedEdist w a b = weightedEdist w b a := by
  apply le_antisymm
  · apply le_iInf
    intro p
    simpa only [Protocol.cost_reverse w hw] using weightedEdist_le_cost w p.reverse
  · apply le_iInf
    intro p
    simpa only [Protocol.cost_reverse w hw] using weightedEdist_le_cost w p.reverse

/-- Symmetric nonnegative edge costs induce a pseudo extended metric. -/
noncomputable def weightedPseudoEMetricSpace {V : Type*} (w : V → V → ℝ≥0∞)
    (hw : ∀ x y, w x y = w y x) : PseudoEMetricSpace V where
  edist := weightedEdist w
  edist_self := weightedEdist_self w
  edist_comm := weightedEdist_symm w hw
  edist_triangle := weightedEdist_triangle w

end GeoSBT
