"""Nb=5 V_nnn = V_nn scan on 72-site Kagome disk with the discrete trap

    E_trap(r) = round((x^2 + y^2) * 9) * 0.02

V_nn from 0 to 2, 11 points (step 0.2), lowest 10 states per (L sector, V) point.
Resumes from existing npz if compatible.
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
Nb = 5
n_show = 10

V_nn_list = np.linspace(0.0, 2.0, 11)
V_nnn_list = V_nn_list.copy()
tag = 'newtrap_vnnn_eq_vnn_0to2'
npz_out = f'data/nb5/kagome_disk_nb{Nb}_{tag}.npz'

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
        print(f"Resume: {start_idx}/{len(V_nn_list)} done.", flush=True)

t0 = time.time()
for idx, (v, vp) in enumerate(zip(V_nn_list, V_nnn_list)):
    if idx < start_idx:
        continue
    print(f"\n=== [{idx+1}/{len(V_nn_list)}] V_nn={v:.4f}  V_nnn={vp:.4f} ===",
          flush=True)
    E, L = run_ed_newtrap(lat, params, Nb=Nb, V_nn=v, V_nnn=vp, n_states=n_show)
    k = min(n_show, len(E))
    records_E[idx, :k] = E[:k]
    records_L[idx, :k] = L[:k]
    print(f"  Emin={E[0]: .6f}  L_ground={L[0]}  elapsed={time.time()-t0:.0f}s",
          flush=True)
    np.savez(npz_out, V_nn=V_nn_list, V_nnn=V_nnn_list,
             E=records_E, L=records_L, done=np.array(idx + 1),
             trap_formula='round((x^2+y^2)*9)*0.02')

print(f"\nTotal scan time: {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for i, v in enumerate(V_nn_list):
        mask = records_L[i] == Ltag
        xs.extend([v] * int(mask.sum()))
        ys.extend(records_E[i, mask].tolist())
    ax.scatter(xs, ys, s=18, alpha=0.8, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')

ax.set_xlabel(r'$V_{nn}$  ($V_{nnn}=V_{nn}$, newtrap)')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} spectrum vs V (newtrap, '
             f'$V_{{nnn}}=V_{{nn}}$, {len(V_nn_list)} pts, lowest {n_show})')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'figures/nb5/kagome_disk_nb{Nb}_{tag}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")
print(f"Saved data: {npz_out}")
