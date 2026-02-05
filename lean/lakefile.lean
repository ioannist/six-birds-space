import Lake
open Lake DSL

package GeoSBT where
  moreServerArgs := #[]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.27.0"

@[default_target]
lean_lib GeoSBT
