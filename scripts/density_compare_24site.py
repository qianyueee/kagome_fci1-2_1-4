"""Compare ground-state densities on 24-site Kagome disk:
  panel a) Nb=2  ν_band=1/4  V_nn=0       (target: FCI 1/4 -- but band not flat)
  panel b) Nb=2  ν_band=1/4  V_nn=0.5     (with weak NN repulsion)
  panel c) Nb=4  ν_band=1/2  V_nn=0       (textbook bosonic Laughlin)

Shows what 'FCI-like density' really looks like on a small disk.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'scripts')
from density_fci_quarter_24site import (
    build_disk, build_c6, find_neighbors, static_lists)
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian


def ground_density(pos, sub, sigma, nn, nnn, Nb, V_nn, V_nnn):
    n = len(pos)
    static = static_lists(pos, sub, nn, nnn, V_nn, V_nnn)
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    best = {'E': np.inf}
    for L in range(6):
        basis = boson_basis_general(n, Nb=Nb, sps=2, C6=(sigma, L))
        if basis.Ns == 0:
            continue
        H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
        E_all, V_all = np.linalg.eigh(H.toarray())
        if E_all[0] < best['E']:
            best = dict(L=L, E=E_all[0], psi=V_all[:, 0].copy(), basis=basis,
                        gap=E_all[1] - E_all[0])
    rho = np.zeros(n)
    for i in range(n):
        op = hamiltonian([['n', [[1.0, i]]]], [], basis=best['basis'],
                         dtype=np.complex128, **no_checks)
        rho[i] = op.expt_value(best['psi']).real
    return rho, best


pos, sub = build_disk(24)
sigma = build_c6(pos)
nn, nnn = find_neighbors(pos)

cases = [
    ('a) Nb=2, V_nn=0   (ν_band=1/4)', 2, 0.0, 0.0),
    ('b) Nb=2, V_nn=0.5 (ν_band=1/4)', 2, 0.5, 0.0),
    ('c) Nb=4, V_nn=0   (ν_band=1/2 Laughlin)', 4, 0.0, 0.0),
]

results = []
for label, Nb, Vn, Vnnn in cases:
    rho, best = ground_density(pos, sub, sigma, nn, nnn, Nb, Vn, Vnnn)
    print(f"{label}: L_g={best['L']}, E={best['E']:.4f}, "
          f"gap={best['gap']:.4f}, ⟨n⟩={rho.mean():.3f}, "
          f"σ/⟨n⟩={rho.std()/rho.mean():.3f}")
    for s, name in enumerate(['A', 'B', 'C']):
        m = sub == s
        print(f"    {name}: {rho[m].mean():.3f}")
    results.append((label, Nb, rho, best))

vmax = max(r[2].max() for r in results)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (label, Nb, rho, best) in zip(axes, results):
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=rho, s=350, cmap='viridis',
                    edgecolor='k', linewidth=0.5, vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_aspect('equal')
    ax.set_title(f'{label}\n$L_g$={best["L"]},  gap={best["gap"]:.3f},  '
                 f'σ/⟨n⟩={rho.std()/rho.mean():.2f}')
    ax.grid(True, alpha=0.25)

plt.tight_layout()
out = 'kagome_disk24_density_compare.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nSaved: {out}")
