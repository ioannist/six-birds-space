import Mathlib.Combinatorics.SimpleGraph.Metric

open SimpleGraph

/-- Triangle inequality for graph distance (path cost). -/
theorem graph_edist_triangle {V : Type} (G : SimpleGraph V) (a b c : V) :
    G.edist a c ≤ G.edist a b + G.edist b c := by
  simpa using G.edist_triangle (u := a) (v := b) (w := c)
