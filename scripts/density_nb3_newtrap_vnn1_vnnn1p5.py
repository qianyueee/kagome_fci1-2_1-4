"""Nb=3 ground-state site density at V_nn=1, V_nnn=1.5 (newtrap).

Discrete trap E_trap(r) = round((x^2+y^2)*9) * 0.02.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')

import time
import numpy as np
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian

from kagome import DiskLattice, ModelParams
from kagome.hamiltonian import PLUS_PAIRS


def build_static_newtrap(lattice, params, V_nn, V_nnn):
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
    if V_nn != 0:
        static.append(["nn", [[V_nn, i, j] for i, j in lattice.nn_pairs]])
    if V_nnn != 0:
        static.append(["nn", [[V_nnn, i, j] for i, j in lattice.nnn_pairs]])
    return static


lat = DiskLattice()
params = ModelParams()
Nb = 3
V_nn = 1.0
V_nnn = 1.5

n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
static = build_static_newtrap(lat, params, V_nn, V_nnn)

print(f"Nb={Nb}, newtrap, V_nn={V_nn}, V_nnn={V_nnn}")

best = {'L': None, 'E': np.inf, 'psi': None, 'basis': None}
t0 = time.time()
for L in range(6):
    basis = boson_basis_general(n, Nb=Nb, sps=2, C6=(sigma, L))
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
    ts = time.time()
    k = 4
    ncv = min(max(3 * k, 2 * k + 1), basis.Ns - 1)
    E_k, V_k = H.eigsh(k=k, which='SA', tol=1e-8, ncv=ncv,
                       return_eigenvectors=True)
    idx0 = int(np.argmin(E_k))
    print(f"  L={L}  Ns={basis.Ns}  diag {time.time()-ts:5.1f}s  "
          f"Emin={E_k[idx0]:.6f}", flush=True)
    if E_k[idx0] < best['E']:
        best['E'] = E_k[idx0]
        best['L'] = L
        best['psi'] = V_k[:, idx0].copy()
        best['basis'] = basis

print(f"\nGround state in L={best['L']}, E={best['E']:.6f}, "
      f"total {time.time()-t0:.1f}s")

basis_gs = best['basis']
psi = best['psi']

print("\nComputing site densities via C6-symmetrised density operator ...")
density = np.zeros(n)
ts = time.time()
for i in range(n):
    orb = [i]
    j = int(sigma[i])
    while j != i:
        orb.append(j); j = int(sigma[j])
    coup = [[1.0 / len(orb), s] for s in orb]
    n_sym = hamiltonian([['n', coup]], [], basis=basis_gs,
                        dtype=np.complex128, **no_checks)
    density[i] = n_sym.expt_value(psi).real
print(f"  done in {time.time()-ts:.1f}s, sum(n_i) = {density.sum():.4f} "
      f"(expect {Nb})")

out_npz = f'data/nb3/kagome_disk_nb{Nb}_newtrap_density_vnn{V_nn:g}_vnnn{V_nnn:g}.npz'
np.savez(out_npz, density=density, positions=lat.positions,
         sublats=lat.sublats, V_nn=V_nn, V_nnn=V_nnn,
         L_ground=best['L'], E_ground=best['E'])
print(f"Saved: {out_npz}")
