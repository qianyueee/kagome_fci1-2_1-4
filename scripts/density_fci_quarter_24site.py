"""FCI ν=1/4 boson ground state on a 24-site Kagome disk.

24 sites is the smallest C6 disk that admits exact ν_band=1/4 (with
N_band=8 lowest-band states, Nb=2 gives 2/8=1/4). 72-site Nb=6 needs
~25-30GB RAM and won't fit on a 16GB system.

Computes ground state at moderate V_nn, V_nnn (probe FCI vs CDW),
extracts site density n_i, plots on the disk lattice.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2.0])
ORB_POS = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [0.25, np.sqrt(3) / 4],
])
C0 = np.array([0.75, np.sqrt(3) / 4.0])
NN_DIST = 0.5
NNN_DIST = np.sqrt(3) / 2.0

t_hop = 1.0
tp_hop = -0.19
phi = 0.22 * np.pi
V_trap = 0.005


def build_disk(N_target):
    N_max = 8
    pos, sub = [], []
    for n1 in range(-N_max, N_max + 1):
        for n2 in range(-N_max, N_max + 1):
            for s in range(3):
                pos.append(n1 * A1 + n2 * A2 + ORB_POS[s] - C0)
                sub.append(s)
    pos = np.array(pos)
    sub = np.array(sub)
    dist = np.linalg.norm(pos, axis=1)
    order = np.argsort(dist)
    ds = dist[order]
    assert ds[N_target - 1] + 1e-6 < ds[N_target], \
        f"disk boundary ambiguous at N={N_target}: " \
        f"d[{N_target-1}]={ds[N_target-1]:.6f}, d[{N_target}]={ds[N_target]:.6f}"
    R_cut = 0.5 * (ds[N_target - 1] + ds[N_target])
    sel = dist < R_cut
    return pos[sel], sub[sel].astype(int)


def build_c6(positions):
    theta = -np.pi / 3
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rotated = positions @ R.T
    tree = cKDTree(positions)
    err, sigma = tree.query(rotated, k=1)
    assert err.max() < 1e-8, f"C6 broken err={err.max():.2e}"
    return sigma.astype(np.int32)


def find_neighbors(positions):
    n = len(positions)
    D = cdist(positions, positions)
    nn, nnn = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if abs(d - NN_DIST) < 1e-6:
                nn.append((i, j, d))
            elif abs(d - NNN_DIST) < 1e-6:
                nnn.append((i, j, d))
    return nn, nnn


def static_lists(positions, sublats, nn, nnn, V_nn, V_nnn):
    n = len(positions)
    static = []

    nn_hop_pos, nn_hop_neg = [], []
    for i, j, _ in nn:
        nn_hop_pos.append([-t_hop * np.exp(1j * phi), i, j])
        nn_hop_neg.append([-t_hop * np.exp(-1j * phi), j, i])
    static.append(['+-', nn_hop_pos])
    static.append(['+-', nn_hop_neg])

    nnn_hop = []
    for i, j, _ in nnn:
        nnn_hop.append([-tp_hop, i, j])
        nnn_hop.append([-tp_hop, j, i])
    static.append(['+-', nnn_hop])

    trap = []
    for i in range(n):
        r2 = (positions[i] ** 2).sum() / 0.25  # in units of (a/2)^2
        trap.append([V_trap * r2, i])
    static.append(['n', trap])

    if V_nn != 0:
        nn_int = [[V_nn, i, j] for i, j, _ in nn]
        static.append(['nn', nn_int])
    if V_nnn != 0:
        nnn_int = [[V_nnn, i, j] for i, j, _ in nnn]
        static.append(['nn', nnn_int])
    return static


pos, sub = build_disk(24)
n = len(pos)
sigma = build_c6(pos)
nn, nnn = find_neighbors(pos)
print(f"24-site disk built: n={n}, NN pairs={len(nn)}, NNN pairs={len(nnn)}")

Nb = 2  # ν_band = 2 / 8 = 1/4
V_nn = 0.5
V_nnn = 0.0

print(f"\nNb={Nb}, V_nn={V_nn}, V_nnn={V_nnn}")
static = static_lists(pos, sub, nn, nnn, V_nn, V_nnn)
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

best = {'L': None, 'E': np.inf, 'psi': None, 'basis': None}
for L in range(6):
    basis = boson_basis_general(n, Nb=Nb, sps=2, C6=(sigma, L))
    if basis.Ns == 0:
        continue
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
    Hd = H.toarray()
    E_all, V_all = np.linalg.eigh(Hd)
    print(f"  L={L}  Ns={basis.Ns}  Emin={E_all[0]:.6f}  E1-E0={E_all[1]-E_all[0]:.4f}")
    if E_all[0] < best['E']:
        best.update(L=L, E=E_all[0], psi=V_all[:, 0].copy(), basis=basis)

print(f"\nGround state in L={best['L']}, E={best['E']:.6f}")

basis_gs = best['basis']
psi = best['psi']
density = np.zeros(n)
for i in range(n):
    n_op = hamiltonian([['n', [[1.0, i]]]], [], basis=basis_gs,
                       dtype=np.complex128, **no_checks)
    density[i] = n_op.expt_value(psi).real
print(f"sum(n_i) = {density.sum():.4f} (expect {Nb})")
print(f"⟨n⟩ = {density.mean():.4f}, σ(n) = {density.std():.4f}, "
      f"σ/⟨n⟩ = {density.std()/density.mean():.3f}")

print(f"\nPer-sublattice density:")
for s, name in enumerate(['A', 'B', 'C']):
    mask = sub == s
    print(f"  {name}: ⟨n⟩ = {density[mask].mean():.4f}  ({mask.sum()} sites)")

out_npz = f'kagome_disk24_density_fci_quarter_Nb{Nb}.npz'
np.savez(out_npz, density=density, positions=pos, sublats=sub,
         V_nn=V_nn, V_nnn=V_nnn, L_ground=best['L'], E_ground=best['E'])
print(f"\nSaved: {out_npz}")

fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(pos[:, 0], pos[:, 1], c=density, s=400, cmap='viridis',
                edgecolor='k', linewidth=0.6, vmin=0)
plt.colorbar(sc, ax=ax, shrink=0.8, label=r'$\langle n_i \rangle$')
ax.set_aspect('equal')
ax.set_title(f'24-site disk  Nb={Nb}  (ν_band=1/4)\n'
             f'$V_{{nn}}={V_nn}$, $V_{{nnn}}={V_nnn}$   '
             f'$L_g={best["L"]}$,  $E={best["E"]:.4f}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.25)
plt.tight_layout()
out = f'kagome_disk24_density_fci_quarter_Nb{Nb}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")
