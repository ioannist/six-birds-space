from __future__ import annotations

import numpy as np

from geo_sbt.packaging import make_C
from geo_sbt.route_mismatch import rm_closure_two_step


def main() -> None:
    P = np.eye(4, dtype=np.float64)
    labels0 = np.array([0, 0, 1, 1], dtype=int)
    C0 = make_C(labels0, m=2)
    U0 = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])

    labels1 = np.array([0, 1, 2, 3], dtype=int)
    C1 = make_C(labels1, m=4)
    U1 = np.eye(4, dtype=np.float64)

    rm_commuting_tv = rm_closure_two_step(P, 1, 1, C0, U0, C1, U1, distance="tv_sup")
    rm_commuting_fro = rm_closure_two_step(P, 1, 1, C0, U0, C1, U1, distance="fro")

    labels1b = np.array([0, 1, 0, 1], dtype=int)
    C1b = make_C(labels1b, m=2)
    U1b = np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5]])

    rm_non_tv = rm_closure_two_step(P, 1, 1, C0, U0, C1b, U1b, distance="tv_sup")
    rm_non_fro = rm_closure_two_step(P, 1, 1, C0, U0, C1b, U1b, distance="fro")

    print(f"RM commuting (tv_sup): {rm_commuting_tv:.16e}")
    print(f"RM noncommuting (tv_sup): {rm_non_tv:.16e}")
    print(f"RM commuting (fro): {rm_commuting_fro:.16e}")
    print(f"RM noncommuting (fro): {rm_non_fro:.16e}")


if __name__ == "__main__":
    main()
