"""Nb=5 trap×1.4 (V_trap=0.007), V_nn=V_nnn=6 and 7: lowest-100 + density.
Probe whether the L=5↔L=0 transition at V_nn=V_nnn=5 (trap×~1.31) shifts
with V_nn=V_nnn.
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
params = ModelParams(V_trap=0.007)   # trap×1.4
Nb = 5
k_per_sector = 30
points = [(6.0, 6.0), (7.0, 7.0)]

n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

seen = np.zeros(n, dtype=bool); orbits = []
for i in range(n):
    if seen[i]:
        continue
    orb = [i]; j = int(sigma[i])
    while j != i:
        orb.append(j); j = int(sigma[j])
    orbits.append(orb); seen[orb] = True

out_dir = 'figures/nb5/vnn_eq_diag_trap1p4x'
os.makedirs(out_dir, exist_ok=True)

T0 = time.time()
for V_nn, V_nnn in points:
    print(f"\n{'='*70}")
    print(f"Nb={Nb}, V_nn={V_nn}, V_nnn={V_nnn}, V_trap={params.V_trap} "
          f"(trap×1.4).  Total elapsed so far: {time.time()-T0:.0f}s")
    print('='*70)
    static = quspin_static_lists(lat, params, V_nn=V_nn, V_nnn=V_nnn)

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
    print(f"  ED total: {time.time()-t0:.1f}s")

    E = np.concatenate(all_E); Lv = np.concatenate(all_L)
    order = np.argsort(E)
    E_sorted = E[order]; L_sorted = Lv[order]
    sector_index = []
    for L in range(6):
        sector_index.extend([(L, j) for j in range(len(all_E[L]))])
    sector_index = [sector_index[i] for i in order]

    n_keep = min(100, len(E_sorted))
    E100 = E_sorted[:n_keep]; L100 = L_sorted[:n_keep]
    gs_L, gs_j = sector_index[0]
    psi_gs = all_V[gs_L][:, gs_j]
    basis_gs = basis_per_L[gs_L]
    print(f"  GS: L={gs_L}, E={E100[0]:.6f}, gap={E100[1]-E100[0]:.4f}, "
          f"L_1st={L100[1]}")
    print("  Lowest 6: " + ", ".join(
        f"dE={E100[k]-E100[0]:+.4f}(L={L100[k]})" for k in range(6)))

    density = np.zeros(n)
    for orb in orbits:
        coup = [[1.0 / len(orb), s] for s in orb]
        nop = hamiltonian([['n', coup]], [], basis=basis_gs,
                          dtype=np.complex128, **no_checks)
        density[orb] = nop.expt_value(psi_gs).real
    print(f"  density: sum={density.sum():.4f}, max={density.max():.4f}, "
          f"min={density.min():.4f}, std={density.std():.4f}")

    tag = f'vnn{V_nn:g}_vnnn{V_nnn:g}_trap1p4x'.replace('.', 'p')
    out_npz = f'data/nb5/kagome_density_nb5_{tag}.npz'
    np.savez(out_npz, V_nn=V_nn, V_nnn=V_nnn, V_trap=params.V_trap,
             E=E100, L=L100, density=density,
             positions=lat.positions, sublats=lat.sublats,
             L_ground=gs_L, E_ground=E100[0])
    print(f"  saved npz: {out_npz}")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_low_energy_degen(L100, E100, n_low=n_keep, ax=ax)
    ax.set_title(rf'Nb={Nb}  $V_{{nn}}=V_{{nnn}}={V_nn:g}$, '
                 rf'trap1.4$\times$ ($V_{{trap}}={params.V_trap:g}$)  '
                 f'(lowest {n_keep})\n'
                 rf'$L_{{GS}}={gs_L}$, $E_{{GS}}={E100[0]:.4f}$, '
                 rf'gap={E100[1]-E100[0]:.4f}, $L_{{1st}}={L100[1]}$')
    plt.tight_layout()
    out_sp = f'{out_dir}/spectrum_{tag}_n{n_keep}.png'
    plt.savefig(out_sp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved spectrum: {out_sp}")

    fig2, axes = plt.subplots(2, 1, figsize=(6.5, 11))
    xs, ys = lat.positions[:, 0], lat.positions[:, 1]
    ax0 = axes[0]
    sc = ax0.scatter(xs, ys, c=density, s=160, cmap='viridis',
                     vmin=0.0, vmax=density.max(),
                     edgecolor='k', linewidth=0.4)
    plt.colorbar(sc, ax=ax0, shrink=0.85, label=r'$\langle n_i \rangle$')
    ax0.set_aspect('equal')
    ax0.set_title(rf"Nb={Nb}, $V_{{nn}}=V_{{nnn}}={V_nn:g}$, "
                  rf"trap1.4$\times$ ($V_{{trap}}={params.V_trap:g}$)"
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
    ax1.set_title(rf"trap1.4$\times$ ($V_{{trap}}={params.V_trap:g}$)",
                  fontsize=12)
    ax1.grid(True, alpha=0.3); ax1.set_ylim(0, density.max() * 1.05)
    plt.tight_layout()
    out_d = f'{out_dir}/density_{tag}.png'
    plt.savefig(out_d, dpi=180, bbox_inches='tight')
    plt.close(fig2)
    print(f"  saved density: {out_d}")

print(f"\nTotal scan time: {(time.time()-T0)/60:.1f} min")
