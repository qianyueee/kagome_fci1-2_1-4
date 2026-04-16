"""C6 symmetry operators and angular momentum classification."""

import numpy as np


def c6_site_matrix(lattice):
    """C6 rotation operator in the single-particle site basis.

    (C6)_{i,j} = 1 iff site j maps to site i under CW 60 deg rotation.
    """
    n = lattice.n_sites
    C6 = np.zeros((n, n), dtype=complex)
    for j in range(n):
        C6[lattice.sigma[j], j] = 1.0
    return C6


def classify_angular_momentum(evals, evecs, C6, deg_tol=1e-7):
    """Assign angular momentum quantum number L mod 6 to each eigenstate.

    Within degenerate subspaces, diagonalises C6 to extract the phase
    exp(i 2pi L / 6).
    """
    n = len(evals)
    L = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        j = i
        while j < n and abs(evals[j] - evals[i]) < deg_tol:
            j += 1
        V = evecs[:, i:j]
        S = V.conj().T @ C6 @ V
        cv, _ = np.linalg.eig(S)
        for k, c in enumerate(cv):
            L[i + k] = int(round(np.angle(c) * 6 / (2 * np.pi))) % 6
        i = j
    return L
