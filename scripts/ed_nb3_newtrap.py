"""Nb=3 hard-core boson ED on 72-site Kagome disk with the discrete trap

    E_trap(r) = round((x^2 + y^2) * 9) * 0.02

Replaces the continuous V_trap * |r/(a/2)|^2 used elsewhere.
Runs two cases:
  (a) free hard-core   (V_nn = V_nnn = 0)
  (b) V_nn = 1, V_nnn = 1.5
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')

import time
import numpy as np
import matplotlib.pyplot as plt

from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian as qhamiltonian

from kagome import DiskLattice, ModelParams
from kagome.hamiltonian import PLUS_PAIRS
from kagome.plot import plot_low_energy_degen


def build_static_newtrap(lattice, params, V_nn=0.0, V_nnn=0.0):
    n = lattice.n_sites
    tol = 1e-6

    trap_list = []
    for i in range(n):
        pos = lattice.positions[i]
        e = round((pos[0] ** 2 + pos[1] ** 2) * 9) * 0.02
        trap_list.append([e, i])

    hop_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = lattice.D[i, j]
            if abs(d - 0.5) < tol:
                p = (params.phi if (lattice.sublats[j], lattice.sublats[i]) in PLUS_PAIRS
                     else -params.phi)
                hop_list.append([-params.t * np.exp(1j * p), i, j])
            elif abs(d - np.sqrt(3) / 2.0) < tol:
                hop_list.append([-params.tp, i, j])

    static = [["n", trap_list], ["+-", hop_list]]
    if V_nn != 0.0:
        static.append(["nn", [[V_nn, i, j] for i, j in lattice.nn_pairs]])
    if V_nnn != 0.0:
        static.append(["nn", [[V_nnn, i, j] for i, j in lattice.nnn_pairs]])
    return static


def run_ed_newtrap(lattice, params, Nb, V_nn=0.0, V_nnn=0.0, n_states=30):
    static = build_static_newtrap(lattice, params, V_nn=V_nn, V_nnn=V_nnn)
    sigma = lattice.sigma
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

    all_evals, all_Ls = [], []
    t0 = time.time()
    for L in range(6):
        basis = boson_basis_general(lattice.n_sites, Nb=Nb, sps=2, C6=(sigma, L))
        H = qhamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
        ts = time.time()
        if n_states is None or basis.Ns <= 500:
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


lat = DiskLattice()
params = ModelParams()
Nb = 3
n_show = 100

cases = [
    ('free',          0.0, 0.0, 'free hard-core (V_nn=V_nnn=0)'),
    ('vnn1_vnnn1p5',  1.0, 1.5, 'V_nn=1, V_nnn=1.5'),
]

for tag, V_nn, V_nnn, title in cases:
    print(f"\n=== {title} ===")
    E, L = run_ed_newtrap(lat, params, Nb=Nb, V_nn=V_nn, V_nnn=V_nnn,
                          n_states=n_show)
    k = min(n_show, len(E))
    E, L = E[:k], L[:k]
    E0 = E[0]
    print(f"\nLowest {k} (Nb={Nb}, newtrap, V_nn={V_nn}, V_nnn={V_nnn}):")
    for j in range(k):
        print(f"  k={j:3d}  E={E[j]: .6f}  dE={E[j]-E0: .6f}  L_tot={L[j]}")

    npz = f'data/nb3/kagome_disk_nb3_newtrap_{tag}_n{n_show}.npz'
    np.savez(npz, V_nn=V_nn, V_nnn=V_nnn, E=E, L=L, n_show=n_show,
             trap_formula='round((x^2+y^2)*9)*0.02')
    print(f"Saved data: {npz}")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_low_energy_degen(L, E, n_low=n_show, ax=ax)
    ax.set_title(f'Nb={Nb}, newtrap, {title}')
    plt.tight_layout()
    out = f'figures/nb3/kagome_disk_nb3_newtrap_{tag}_n{n_show}.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
