from __future__ import annotations

import numpy as np

from geo_sbt.packaging import (
    idempotence_defect_delta,
    make_C,
    prototype_stabilities,
)


def main() -> None:
    labels = np.array([0, 0, 1, 1], dtype=int)
    C = make_C(labels, m=2)
    U = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])

    P_id = np.eye(4, dtype=np.float64)
    delta_id = idempotence_defect_delta(P_id, tau=1, C=C, U=U)
    stabilities_id = prototype_stabilities(P_id, tau=1, C=C, U=U)

    print("Idempotent case:")
    print(f"delta={delta_id:.16e}")
    print("stabilities=", stabilities_id)

    P_non = np.array(
        [
            [0.6, 0.4, 0.0, 0.0],
            [0.2, 0.6, 0.2, 0.0],
            [0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.4, 0.6],
        ],
        dtype=np.float64,
    )
    delta_non = idempotence_defect_delta(P_non, tau=2, C=C, U=U)
    print("Non-idempotent case:")
    print(f"delta={delta_non:.16e}")


if __name__ == "__main__":
    main()
