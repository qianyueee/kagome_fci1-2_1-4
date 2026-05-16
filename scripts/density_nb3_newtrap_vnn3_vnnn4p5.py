"""Nb=3 ground-state site density at V_nn=3, V_nnn=4.5 (newtrap).

Uses the discrete trap E_trap(r) = round((x^2+y^2)*9) * 0.02.
Finds the global ground state across L sectors, then computes
<n_i> for all 72 sites and plots a colored disk.
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
V_nn = 3.0
V_nnn = 4.5

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
# For each site i, compute <psi| (1/6) sum_k n_{sigma^k(i)} |psi>
# which equals <n_j> for every j in the C6 orbit of i (orbit averaging is
# trivial here because all 12 orbits are length 6 on this disk).
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

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

xs, ys = lat.positions[:, 0], lat.positions[:, 1]
sc = axes[0].scatter(xs, ys, c=density, s=140, cmap='viridis',
                     edgecolor='k', linewidth=0.4)
plt.colorbar(sc, ax=axes[0], shrink=0.8, label=r'$\langle n_i \rangle$')
axes[0].set_aspect('equal')
axes[0].set_title(f'Nb={Nb} ground density (newtrap, $V_{{nn}}={V_nn:g}$, '
                  f'$V_{{nnn}}={V_nnn:g}$, $L={best["L"]}$)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].grid(True, alpha=0.25)

# Radial profile: average <n_i> in bins of |r|
r = np.linalg.norm(lat.positions, axis=1)
order = np.argsort(r)
axes[1].plot(r[order], density[order], 'o-', ms=4, lw=0.6, alpha=0.6,
             label='per-site')
# Bin-averaged
bins = np.linspace(0, r.max() + 0.01, 12)
binc = 0.5 * (bins[:-1] + bins[1:])
mean_n = np.zeros_like(binc)
for k in range(len(binc)):
    m = (r >= bins[k]) & (r < bins[k + 1])
    mean_n[k] = density[m].mean() if m.any() else np.nan
axes[1].plot(binc, mean_n, 'r-s', lw=2, ms=7, label='bin-avg')
# Reference: ν=1/2 hard-core boson on Kagome lowest band -> 1 boson per 2 unit cells.
# Per site density would be (1/2)/3 = 1/6 ≈ 0.167.  Average per site of full filling = Nb/n.
axes[1].axhline(Nb / n, ls='--', color='gray', label=f'avg = {Nb/n:.3f}')
axes[1].axhline(1.0 / 6, ls=':', color='magenta',
                label=r'$\nu=1/2$ Laughlin plateau (1/6)')
axes[1].set_xlabel(r'$|r|$ (site distance from center)')
axes[1].set_ylabel(r'$\langle n_i \rangle$')
axes[1].set_title('Radial density profile')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_png = f'figures/nb3/kagome_disk_nb{Nb}_newtrap_density_vnn{V_nn:g}_vnnn{V_nnn:g}.png'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f"Saved: {out_png}")
