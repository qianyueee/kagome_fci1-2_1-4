"""Many-body exact diagonalization using QuSpin with C6 block decomposition."""

import time
import numpy as np
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian

from .hamiltonian import quspin_static_lists


def run_ed(lattice, params, Nb, V_nn=0.0, V_nnn=0.0, n_states=30):
    """Block-diagonalize Nb hard-core bosons on the Kagome disk by C6 sector.

    Args:
        lattice  : DiskLattice instance.
        params   : ModelParams instance.
        Nb       : number of bosons.
        V_nn     : nearest-neighbour repulsion strength.
        V_nnn    : next-nearest-neighbour repulsion strength.
        n_states : eigenvalues per sector.  None -> all (dense diag).

    Returns:
        (energies, L_values) sorted by energy.
    """
    static = quspin_static_lists(lattice, params, V_nn=V_nn, V_nnn=V_nnn)
    n = lattice.n_sites
    sigma = lattice.sigma
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

    all_evals, all_Ls = [], []
    t0 = time.time()

    for L in range(6):
        basis = boson_basis_general(n, Nb=Nb, sps=2, C6=(sigma, L))
        H = hamiltonian(static, [], basis=basis, dtype=np.complex128,
                        **no_checks)
        ts = time.time()

        if n_states is None or basis.Ns <= 500:
            # Dense for small sectors or when all eigenvalues requested
            evals = np.sort(np.linalg.eigvalsh(H.toarray()))
            if n_states is not None:
                evals = evals[:n_states]
        else:
            k = min(n_states, basis.Ns - 2)
            ncv = min(max(3 * k, 2 * k + 1), basis.Ns - 1)
            evals = np.sort(H.eigsh(k=k, which='SA', tol=1e-8, ncv=ncv,
                                    return_eigenvectors=False))

        print(f"  L={L}  Ns={basis.Ns:6d}  diag {time.time()-ts:5.1f}s  "
              f"Emin={evals.min():.6f}", flush=True)
        all_evals.append(evals)
        all_Ls.append(np.full_like(evals, L, dtype=int))

    print(f"Total diag: {time.time()-t0:.1f}s")

    E = np.concatenate(all_evals)
    Lv = np.concatenate(all_Ls)
    srt = np.argsort(E)
    return E[srt], Lv[srt]
