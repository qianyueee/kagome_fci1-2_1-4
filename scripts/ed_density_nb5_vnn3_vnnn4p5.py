"""Nb=5, V_nn=3, V_nnn=4.5 (=1.5*V_nn): lowest-100 spectrum + GS density.

For each L sector eigsh k=30 with eigenvectors -> 180 candidates,
keep global lowest 100 for the spectrum plot; pick the global GS to
compute site densities n_i = <psi|n_i|psi>.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')

import time
import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian

from kagome import DiskLattice, ModelParams
from kagome.hamiltonian import quspin_static_lists
from kagome.plot import plot_low_energy_degen

lat = DiskLattice()
params = ModelParams()
Nb = 5
V_nn = 3.0
V_nnn = 4.5
k_per_sector = 30

n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
static = quspin_static_lists(lat, params, V_nn=V_nn, V_nnn=V_nnn)

print(f"Nb={Nb}, V_nn={V_nn}, V_nnn={V_nnn},  k_per_sector={k_per_sector}")

all_E, all_L, all_V = [], [], []
basis_per_L = {}
t0 = time.time()
for L in range(6):
    basis = boson_basis_general(n, Nb=Nb, sps=2, C6=(sigma, L))
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
    ts = time.time()
    k = min(k_per_sector, basis.Ns - 2)
    ncv = min(max(3 * k, 2 * k + 1), basis.Ns - 1)
    E_k, V_k = H.eigsh(k=k, which='SA', tol=1e-8, ncv=ncv,
                       return_eigenvectors=True)
    order = np.argsort(E_k)
    E_k = E_k[order]
    V_k = V_k[:, order]
    print(f"  L={L}  Ns={basis.Ns}  diag {time.time()-ts:5.1f}s  "
          f"Emin={E_k[0]:.6f}", flush=True)
    all_E.append(E_k)
    all_L.append(np.full(len(E_k), L, dtype=int))
    all_V.append(V_k)
    basis_per_L[L] = basis

print(f"\nED total: {time.time()-t0:.1f}s")

E = np.concatenate(all_E)
Lv = np.concatenate(all_L)
order = np.argsort(E)
E = E[order]
Lv = Lv[order]
# Map back to (L_sector, intra-sector index)
sector_index = []
for L in range(6):
    sector_index.extend([(L, j) for j in range(len(all_E[L]))])
sector_index = [sector_index[i] for i in order]

n_keep = min(100, len(E))
E100 = E[:n_keep]
L100 = Lv[:n_keep]
print(f"\nLowest {n_keep} states:")
for k in range(n_keep):
    print(f"  k={k:3d}  E={E100[k]:+.6f}  dE={E100[k]-E100[0]:+.6f}  "
          f"L={L100[k]}")

gs_L, gs_j = sector_index[0]
psi_gs = all_V[gs_L][:, gs_j]
basis_gs = basis_per_L[gs_L]
print(f"\nGS in L={gs_L}, E={E100[0]:.6f}")

# Site densities
print("Computing site densities ...")
density = np.zeros(n)
ts = time.time()
for i in range(n):
    n_op = hamiltonian([['n', [[1.0, i]]]], [], basis=basis_gs,
                       dtype=np.complex128, **no_checks)
    density[i] = n_op.expt_value(psi_gs).real
print(f"  done in {time.time()-ts:.1f}s, sum(n_i)={density.sum():.4f} "
      f"(expect {Nb})")

tag = 'nb5_vnn3_vnnn4p5'
npz_out = f'data/nb5/kagome_density_{tag}.npz'
os.makedirs(os.path.dirname(npz_out), exist_ok=True)
np.savez(npz_out,
         V_nn=V_nn, V_nnn=V_nnn,
         E=E100, L=L100,
         density=density,
         positions=lat.positions, sublats=lat.sublats,
         L_ground=gs_L, E_ground=E100[0])
print(f"Saved data: {npz_out}")

# Spectrum: hline style (kagome_disk_nb3_new format)
fig, ax = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L100, E100, n_low=n_keep, ax=ax)
ax.set_title(rf'Nb={Nb}  $V_{{nn}}={V_nn:g}$, $V_{{nnn}}={V_nnn:g}$  '
             f'(lowest {n_keep})')
plt.tight_layout()
out_sp = f'figures/nb5/kagome_disk_{tag}_spectrum_n{n_keep}.png'
os.makedirs(os.path.dirname(out_sp), exist_ok=True)
plt.savefig(out_sp, dpi=180, bbox_inches='tight')
print(f"Saved spectrum: {out_sp}")

# Density: scatter colored by <n_i>
fig2, ax2 = plt.subplots(figsize=(7, 7))
xs, ys = lat.positions[:, 0], lat.positions[:, 1]
sc = ax2.scatter(xs, ys, c=density, s=160, cmap='viridis',
                 edgecolor='k', linewidth=0.4)
plt.colorbar(sc, ax=ax2, shrink=0.8, label=r'$\langle n_i \rangle$')
ax2.set_aspect('equal')
ax2.set_title(rf'Nb={Nb} GS density   '
              rf'$V_{{nn}}={V_nn:g}$, $V_{{nnn}}={V_nnn:g}$  '
              rf'($L_g={gs_L}$, $E={E100[0]:.4f}$)')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.grid(True, alpha=0.25)
plt.tight_layout()
out_d = f'figures/nb5/kagome_density_{tag}.png'
plt.savefig(out_d, dpi=180, bbox_inches='tight')
print(f"Saved density: {out_d}")
