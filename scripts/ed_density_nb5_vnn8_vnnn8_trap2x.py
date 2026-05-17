"""Nb=5 trap×2 (V_trap=0.010), V_nn=V_nnn=7: lowest-100 + density."""

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
params = ModelParams(V_trap=0.010)
Nb = 5
V_nn = 8.0
V_nnn = 8.0
k_per_sector = 30

n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
static = quspin_static_lists(lat, params, V_nn=V_nn, V_nnn=V_nnn)

seen = np.zeros(n, dtype=bool); orbits = []
for i in range(n):
    if seen[i]:
        continue
    orb = [i]; j = int(sigma[i])
    while j != i:
        orb.append(j); j = int(sigma[j])
    orbits.append(orb); seen[orb] = True

print(f"Nb={Nb}, V_nn={V_nn}, V_nnn={V_nnn}, V_trap={params.V_trap} (2x)")

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
    E_k = E_k[order]; V_k = V_k[:, order]
    print(f"  L={L}  Ns={basis.Ns}  diag {time.time()-ts:5.1f}s  "
          f"Emin={E_k[0]:.6f}", flush=True)
    all_E.append(E_k); all_L.append(np.full(len(E_k), L, dtype=int))
    all_V.append(V_k); basis_per_L[L] = basis
print(f"ED total: {time.time()-t0:.1f}s")

E = np.concatenate(all_E); Lv = np.concatenate(all_L)
order = np.argsort(E)
E_sorted = E[order]; L_sorted = Lv[order]
sector_index = []
for L in range(6):
    sector_index.extend([(L, j) for j in range(len(all_E[L]))])
sector_index = [sector_index[i] for i in order]

n_keep = min(100, len(E_sorted))
E100 = E_sorted[:n_keep]; L100 = L_sorted[:n_keep]
print(f"\nLowest 12:")
for k in range(12):
    print(f"  k={k:2d}  E={E100[k]:+.6f}  dE={E100[k]-E100[0]:+.6f}  L={L100[k]}")

gs_L, gs_j = sector_index[0]
psi_gs = all_V[gs_L][:, gs_j]
basis_gs = basis_per_L[gs_L]
print(f"\nGS: L={gs_L}, E={E100[0]:.6f}, gap={E100[1]-E100[0]:.4f}")

density = np.zeros(n); ts = time.time()
for orb in orbits:
    coup = [[1.0 / len(orb), s] for s in orb]
    nop = hamiltonian([['n', coup]], [], basis=basis_gs,
                      dtype=np.complex128, **no_checks)
    density[orb] = nop.expt_value(psi_gs).real
print(f"density done in {time.time()-ts:.1f}s, sum={density.sum():.4f}")

out_npz = 'data/nb5/kagome_density_nb5_vnn8_vnnn8_trap2x.npz'
np.savez(out_npz, V_nn=V_nn, V_nnn=V_nnn, V_trap=params.V_trap,
         E=E100, L=L100, density=density,
         positions=lat.positions, sublats=lat.sublats,
         L_ground=gs_L, E_ground=E100[0])
print(f"saved npz: {out_npz}")

out_dir = 'figures/nb5/trap2x_eq_diag'
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L100, E100, n_low=n_keep, ax=ax)
ax.set_title(rf'Nb={Nb}  $V_{{nn}}={V_nn:g}$, $V_{{nnn}}={V_nnn:g}$, '
             rf'$V_{{trap}}={params.V_trap:g}$ (2$\times$)  '
             f'(lowest {n_keep})\n'
             rf'$L_{{GS}}={gs_L}$, $E_{{GS}}={E100[0]:.4f}$, '
             rf'gap={E100[1]-E100[0]:.4f}')
plt.tight_layout()
out_sp = f'{out_dir}/spectrum_vnn8_vnnn8_trap2x_n{n_keep}.png'
plt.savefig(out_sp, dpi=150, bbox_inches='tight')
print(f"saved spectrum: {out_sp}")

fig2, axes = plt.subplots(2, 1, figsize=(6.5, 11))
xs, ys = lat.positions[:, 0], lat.positions[:, 1]
ax0 = axes[0]
sc = ax0.scatter(xs, ys, c=density, s=160, cmap='viridis',
                 vmin=0.0, vmax=density.max(),
                 edgecolor='k', linewidth=0.4)
plt.colorbar(sc, ax=ax0, shrink=0.85, label=r'$\langle n_i \rangle$')
ax0.set_aspect('equal')
ax0.set_title(rf"Nb={Nb}, $V_{{nn}}={V_nn:g},\ V_{{nnn}}={V_nnn:g}$, "
              rf"$V_{{trap}}={params.V_trap:g}$ (2$\times$)"
              f"\n$L_{{gs}}={gs_L}$, $E_{{gs}}={E100[0]:.4f}$",
              fontsize=12)
ax0.set_xlabel('x'); ax0.set_ylabel('y'); ax0.grid(True, alpha=0.25)

ax1 = axes[1]
r = np.linalg.norm(lat.positions, axis=1)
order_r = np.argsort(r)
ax1.plot(r[order_r], density[order_r], 'o-', ms=5, lw=0.8,
         color='C0', alpha=0.85)
ax1.set_xlabel(r'$|r|$ (site distance from center)')
ax1.set_ylabel(r'$\langle n_i \rangle$')
ax1.set_title(rf"$V_{{trap}}={params.V_trap:g}$ (2$\times$ default)", fontsize=12)
ax1.grid(True, alpha=0.3); ax1.set_ylim(0, density.max() * 1.05)
plt.tight_layout()
out_d = f'{out_dir}/density_vnn8_vnnn8_trap2x.png'
plt.savefig(out_d, dpi=180, bbox_inches='tight')
print(f"saved density: {out_d}")

print(f"\nSublattice means: "
      f"A={density[lat.sublats==0].mean():.4f}, "
      f"B={density[lat.sublats==1].mean():.4f}, "
      f"C={density[lat.sublats==2].mean():.4f}")
print(f"density: max={density.max():.4f} min={density.min():.4f} "
      f"std={density.std():.4f}")
