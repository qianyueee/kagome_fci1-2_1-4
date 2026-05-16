"""Compute Nb=5 ground-state site density at V_nn=6, V_nnn=V_nn/1.5=4.

Re-runs ED with eigenvectors for the cheapest path:
  - For each L sector: eigsh k=4 (just need the lowest), keep eigenvector.
  - Pick the global ground state across L sectors.
  - Compute n_i = <psi|n_i|psi> for all 72 sites (operator built per-site,
    expectation value automatically projects within the L block).
  - Plot density as colored disks on the lattice.

OMP=2 to coexist with the v6 scan (uses 6) without thrashing the CPU.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import time
import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian

from kagome import DiskLattice, ModelParams
from kagome.hamiltonian import quspin_static_lists

lat = DiskLattice()
params = ModelParams()
Nb = 5
V_nn = 6.0
V_nnn = V_nn / 1.5  # = 4.0

n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
static = quspin_static_lists(lat, params, V_nn=V_nn, V_nnn=V_nnn)

print(f"Nb={Nb}, V_nn={V_nn}, V_nnn={V_nnn}")

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

print("\nComputing site densities ...")
density = np.zeros(n)
ts = time.time()
for i in range(n):
    n_op = hamiltonian([['n', [[1.0, i]]]], [], basis=basis_gs,
                       dtype=np.complex128, **no_checks)
    density[i] = n_op.expt_value(psi).real
print(f"  done in {time.time()-ts:.1f}s, sum(n_i) = {density.sum():.4f} "
      f"(expect {Nb})")

out_npz = (f'kagome_disk_nb{Nb}_density_vnnn_1p5_vnn{V_nn:g}.npz')
np.savez(out_npz, density=density, positions=lat.positions,
         sublats=lat.sublats, V_nn=V_nn, V_nnn=V_nnn,
         L_ground=best['L'], E_ground=best['E'])
print(f"Saved: {out_npz}")

fig, ax = plt.subplots(figsize=(7, 7))
xs, ys = lat.positions[:, 0], lat.positions[:, 1]
sc = ax.scatter(xs, ys, c=density, s=140, cmap='viridis',
                edgecolor='k', linewidth=0.4)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8, label=r'$\langle n_i \rangle$')
ax.set_aspect('equal')
ax.set_title(f'Nb={Nb} ground-state density   '
             f'$V_{{nn}}={V_nn:g}$, $V_{{nnn}}=V_{{nn}}/1.5={V_nnn:g}$  '
             f'($L_g={best["L"]}$, $E={best["E"]:.4f}$)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.25)

plt.tight_layout()
out_png = f'kagome_disk_nb{Nb}_density_vnnn_1p5_vnn{V_nn:g}.png'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f"Saved: {out_png}")
