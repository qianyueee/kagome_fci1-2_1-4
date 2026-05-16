"""Nb=3 newtrap V_nnn = 1.5 V_nn scan, V_nn from 2.2 to 3.0 (step 0.2, 5 points).
Extends ed_nb3_trapcompare_vnnn_1p5vnn_0to2.py; saves to its own npz and also
produces a merged 0..3 plot using both files.
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


def run_ed_newtrap(lattice, params, Nb, V_nn, V_nnn, n_states):
    static = build_static_newtrap(lattice, params, V_nn=V_nn, V_nnn=V_nnn)
    sigma = lattice.sigma
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    all_evals, all_Ls = [], []
    t0 = time.time()
    for L in range(6):
        basis = boson_basis_general(lattice.n_sites, Nb=Nb, sps=2, C6=(sigma, L))
        H = qhamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
        if n_states is None or basis.Ns <= 500:
            evals = np.sort(np.linalg.eigvalsh(H.toarray()))
            if n_states is not None:
                evals = evals[:n_states]
        else:
            k = min(n_states, basis.Ns - 2)
            ncv = min(max(3 * k, 2 * k + 1), basis.Ns - 1)
            evals = np.sort(H.eigsh(k=k, which='SA', tol=1e-8, ncv=ncv,
                                    return_eigenvectors=False))
        all_evals.append(evals)
        all_Ls.append(np.full_like(evals, L, dtype=int))
    print(f"  diag {time.time()-t0:.1f}s", flush=True)
    E = np.concatenate(all_evals)
    Lv = np.concatenate(all_Ls)
    srt = np.argsort(E)
    return E[srt], Lv[srt]


lat = DiskLattice()
params = ModelParams()
Nb = 3
n_show = 20
V_nn_list = np.linspace(2.2, 3.0, 5)
V_nnn_list = 1.5 * V_nn_list
tag = 'newtrap_vnnn_1p5vnn_2to3'
npz_out = f'data/nb3/kagome_disk_nb{Nb}_{tag}.npz'

records_E = np.full((len(V_nn_list), n_show), np.nan)
records_L = np.full((len(V_nn_list), n_show), -1, dtype=int)

start_idx = 0
if os.path.exists(npz_out):
    prev = np.load(npz_out)
    if (prev['V_nn'].shape == V_nn_list.shape and
            np.allclose(prev['V_nn'], V_nn_list) and
            np.allclose(prev['V_nnn'], V_nnn_list)):
        start_idx = int(prev['done'])
        records_E[:start_idx] = prev['E'][:start_idx]
        records_L[:start_idx] = prev['L'][:start_idx]
        print(f"Resume: {start_idx}/{len(V_nn_list)}", flush=True)

t0 = time.time()
for idx, (v, vp) in enumerate(zip(V_nn_list, V_nnn_list)):
    if idx < start_idx:
        continue
    print(f"[{idx+1}/{len(V_nn_list)}] V_nn={v:.3f}  V_nnn={vp:.3f}", flush=True)
    E, L = run_ed_newtrap(lat, params, Nb=Nb, V_nn=v, V_nnn=vp, n_states=n_show)
    k = min(n_show, len(E))
    records_E[idx, :k] = E[:k]
    records_L[idx, :k] = L[:k]
    print(f"  Emin={E[0]: .6f}  L_ground={L[0]}", flush=True)
    np.savez(npz_out, V_nn=V_nn_list, V_nnn=V_nnn_list,
             E=records_E, L=records_L, done=np.array(idx + 1), trap='new')
print(f"Total: {time.time()-t0:.1f}s")

# Merged 0..3 plot using existing 0to2 + this 2to3 npz
prev_npz = f'data/nb3/kagome_disk_nb{Nb}_newtrap_vnnn_1p5vnn_0to2.npz'
prev = np.load(prev_npz)
V_all = np.concatenate([prev['V_nn'][:int(prev['done'])], V_nn_list])
E_all = np.concatenate([prev['E'][:int(prev['done'])], records_E], axis=0)
L_all = np.concatenate([prev['L'][:int(prev['done'])], records_L], axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for i, v in enumerate(V_all):
        m = L_all[i] == Ltag
        xs.extend([v] * int(m.sum()))
        ys.extend(E_all[i, m].tolist())
    ax.scatter(xs, ys, s=20, alpha=0.85, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')
ax.set_xlabel(r'$V_{nn}$  ($V_{nnn}=1.5\,V_{nn}$, newtrap)')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} spectrum vs V (newtrap, $V_{{nnn}}=1.5V_{{nn}}$, '
             f'{len(V_all)} pts, lowest {n_show})')
ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = f'figures/nb3/kagome_disk_nb{Nb}_newtrap_vnnn_1p5vnn_0to3.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved merged plot: {out}")
print(f"Saved data: {npz_out}")
