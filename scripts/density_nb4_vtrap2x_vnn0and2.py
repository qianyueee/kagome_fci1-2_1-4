"""Plot ground-state density profile for Nb=4 on the 72-site Kagome disk
at V_trap = 2x default (0.01) for V_nn = 0 and V_nn = 2.

Uses C6 block-diagonalization to find the global ground state, then computes
<n_i> for every site via per-site number operators (in the symmetric basis;
since psi has definite C6 quantum number, the result equals the orbit-averaged
density, which is also the per-site density).
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

lat = DiskLattice()
n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

V_trap = 0.01      # 2x default
Nb = 4
# (V_nn, V_nnn) tuples
runs = [(5.0, 5.0 / 3.0)]


def find_ground_state_full(V_nn, V_nnn):
    """ED in the full (un-symmetrized) Nb-particle Hilbert space."""
    params = ModelParams(V_trap=V_trap)
    static = quspin_static_lists(lat, params, V_nn=V_nn, V_nnn=V_nnn)
    basis = boson_basis_general(n, Nb=Nb, sps=2)
    print(f"  full-basis Ns = {basis.Ns}", flush=True)
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
    ts = time.time()
    E, V = H.eigsh(k=1, which='SA', tol=1e-8, return_eigenvectors=True)
    print(f"  eigsh: {time.time()-ts:.1f}s", flush=True)
    return float(E[0]), V[:, 0], basis


def density_from_psi(psi, basis):
    density = np.zeros(n)
    for i in range(n):
        n_op = hamiltonian([['n', [[1.0, i]]]], [], basis=basis,
                           dtype=np.float64, **no_checks)
        density[i] = float(n_op.expt_value(psi).real)
    return density


def plot_density(positions, density, V_nn, V_nnn, V_trap, out):
    fig, ax = plt.subplots(figsize=(8, 8))
    norm = max(density.max(), 1e-12)
    sizes = 60 + 1500 * density / norm
    sc = ax.scatter(positions[:, 0], positions[:, 1], c=density,
                    s=sizes, cmap='viridis',
                    edgecolor='black', linewidth=0.4)
    plt.colorbar(sc, ax=ax, label=r'$\langle n_i \rangle$',
                 fraction=0.046, pad=0.04)
    ax.set_aspect('equal')
    ax.set_title(f'Nb={Nb}  $V_{{trap}}={V_trap:g}$  '
                 f'$V_{{nn}}={V_nn:g}$  $V_{{nnn}}={V_nnn:g}$  '
                 f'(sum={density.sum():.4f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out}", flush=True)


for V_nn, V_nnn in runs:
    print(f"\n=== V_nn={V_nn}  V_nnn={V_nnn:.4f}  V_trap={V_trap} ===",
          flush=True)
    t0 = time.time()
    E0, psi, basis = find_ground_state_full(V_nn, V_nnn)
    print(f"  global GS: E={E0:.6f}  (ED total {time.time()-t0:.1f}s)",
          flush=True)

    t1 = time.time()
    density = density_from_psi(psi, basis)
    print(f"  density sum = {density.sum():.6f}  (target Nb = {Nb})  "
          f"(density {time.time()-t1:.1f}s)", flush=True)
    print(f"  density min/max: {density.min():.4f} / {density.max():.4f}",
          flush=True)

    tag = f'vnn{V_nn:g}_vnnn{V_nnn:.3f}'
    out = f'kagome_density_nb{Nb}_vtrap2x_{tag}.png'
    plot_density(lat.positions, density, V_nn, V_nnn, V_trap, out)

    npz_out = f'kagome_density_nb{Nb}_vtrap2x_{tag}.npz'
    np.savez(npz_out, positions=lat.positions, sublats=lat.sublats,
             density=density, V_nn=np.array(V_nn), V_nnn=np.array(V_nnn),
             V_trap=np.array(V_trap), E_ground=np.array(E0))
    print(f"  saved data: {npz_out}", flush=True)
