from __future__ import annotations

import numpy as np

from geo_sbt.lenses.ladder import (
    check_refinement_consistency,
    hierarchical_diffusion_partition,
    max_macro_identity_deviation,
)
from geo_sbt.lenses.prototypes import prototypes_uniform
from geo_sbt.packaging import make_C
from geo_sbt.substrates.grid import grid_2d


def main() -> None:
    P = grid_2d(10, lazy=0.5)
    ladder = hierarchical_diffusion_partition(
        P, levels=[2, 4, 8, 16], n_eigs=4, method="kmeans_merge", seed=0
    )

    labels_list = ladder["labels_list"]
    refine_maps = ladder["refine_maps"]
    counts = ladder["cluster_counts"]

    for idx, labels in enumerate(labels_list):
        U = prototypes_uniform(labels)
        C = make_C(labels)
        max_dev = max_macro_identity_deviation(C, U)
        print(f"level {idx} | m={counts[idx]} | max_macro_identity_dev={max_dev:.2e}")

    consistent = True
    for i in range(len(refine_maps)):
        if not check_refinement_consistency(labels_list[i], labels_list[i + 1], refine_maps[i]):
            consistent = False
            break
    print(f"refinement_consistent={consistent}")


if __name__ == "__main__":
    main()
