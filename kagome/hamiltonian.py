"""Single-particle Hamiltonian and QuSpin coupling lists."""

import numpy as np
from dataclasses import dataclass

# Sublattice pairs with +phi phase: A->B, B->C, C->A
PLUS_PAIRS = {(0, 1), (1, 2), (2, 0)}


@dataclass
class ModelParams:
    """Kagome FCI model parameters."""
    t: float = 1.0
    tp: float = -0.19
    phi: float = 0.22 * np.pi
    V_trap: float = 0.005


def build_single_particle_H(lattice, params):
    """Build the n_sites x n_sites single-particle Hamiltonian.

    Includes NN hopping with Peierls phase, NNN hopping (real), and
    harmonic trap V_trap * |r|^2 (r in units of a/2).
    """
    n = lattice.n_sites
    H = np.zeros((n, n), dtype=complex)

    # Trap: |r| in units of a/2; a = NN distance = 0.5, a/2 = 0.25
    # so |r|_{a/2} = d / 0.25 = 4d
    for i in range(n):
        d = np.linalg.norm(lattice.positions[i])
        H[i, i] = params.V_trap * (4.0 * d) ** 2

    tol = 1e-6
    for i in range(n):
        for j in range(i + 1, n):
            d = lattice.D[i, j]
            if abs(d - 0.5) < tol:
                p = params.phi if (lattice.sublats[i], lattice.sublats[j]) in PLUS_PAIRS else -params.phi
                H[j, i] = -params.t * np.exp(1j * p)
                H[i, j] = np.conj(H[j, i])
            elif abs(d - np.sqrt(3) / 2.0) < tol:
                H[j, i] = -params.tp
                H[i, j] = -params.tp

    return H


def quspin_static_lists(lattice, params, V_nn=0.0):
    """Build QuSpin static operator lists for many-body Hamiltonian.

    Returns list of [op_string, coupling_list] entries for QuSpin's
    ``hamiltonian(static, ...)``.
    """
    n = lattice.n_sites
    tol = 1e-6

    # Trap: "n" operator
    trap_list = []
    for i in range(n):
        d = np.linalg.norm(lattice.positions[i])
        trap_list.append([params.V_trap * (4.0 * d) ** 2, i])

    # Hopping: all directed bonds in "+-" (b_i^dag b_j)
    # Iterating both (i,j) and (j,i) makes the sum manifestly Hermitian.
    hop_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = lattice.D[i, j]
            if abs(d - 0.5) < tol:
                p = params.phi if (lattice.sublats[j], lattice.sublats[i]) in PLUS_PAIRS else -params.phi
                hop_list.append([-params.t * np.exp(1j * p), i, j])
            elif abs(d - np.sqrt(3) / 2.0) < tol:
                hop_list.append([-params.tp, i, j])

    static = [["n", trap_list], ["+-", hop_list]]

    # NN density-density interaction
    if V_nn != 0.0:
        nn_int = []
        for i, j in lattice.nn_pairs:
            nn_int.append([V_nn, i, j])
        static.append(["nn", nn_int])

    return static
