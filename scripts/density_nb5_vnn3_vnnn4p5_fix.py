"""Fix: Nb=5 V_nn=3 V_nnn=4.5 ground-state density using C6-symmetrised
density operator.  Re-runs ED only in L=5 (known GS sector) with k=2,
then computes density two ways for symmetry validation:

  raw      <n_i>  using single-site operator (incorrect in C6 block basis)
  symm     <(1/6) sum_k n_{sigma^k(i)}>  (correct, C6-invariant)

Saves orbit-averaged density and writes a compare3-style 2-row plot.
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
params = ModelParams()
Nb = 5
V_nn = 3.0
V_nnn = 4.5
L_GS_known = 5

n = lat.n_sites
sigma = lat.sigma
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
static = quspin_static_lists(lat, params, V_nn=V_nn, V_nnn=V_nnn)

# Orbit structure
print("Verifying C6 orbits ...")
seen = np.zeros(n, dtype=bool)
orbits = []
for i in range(n):
    if seen[i]:
        continue
    orb = [i]
    j = int(sigma[i])
    while j != i:
        orb.append(j); j = int(sigma[j])
    orbits.append(orb)
    seen[orb] = True
orbit_lengths = [len(o) for o in orbits]
print(f"  {len(orbits)} orbits, lengths: {set(orbit_lengths)}")
assert all(L == 6 for L in orbit_lengths), "Expected all length-6 orbits"
site_to_orbit = np.full(n, -1, dtype=int)
for k, o in enumerate(orbits):
    for s in o:
        site_to_orbit[s] = k

# ED in L=5 only, k=2
print(f"\nED in L={L_GS_known} sector ...")
basis = boson_basis_general(n, Nb=Nb, sps=2, C6=(sigma, L_GS_known))
H = hamiltonian(static, [], basis=basis, dtype=np.complex128, **no_checks)
t0 = time.time()
k = 2
ncv = min(max(3 * k, 2 * k + 1), basis.Ns - 1)
E_k, V_k = H.eigsh(k=k, which='SA', tol=1e-10, ncv=ncv,
                   return_eigenvectors=True)
order = np.argsort(E_k)
E_k = E_k[order]; V_k = V_k[:, order]
print(f"  Ns={basis.Ns}  diag {time.time()-t0:.1f}s")
print(f"  E0 = {E_k[0]:.8f}  E1 = {E_k[1]:.8f}")
psi = V_k[:, 0]
E_GS = float(E_k[0])

# Cross-check with stored value
prev = np.load('data/nb5/kagome_density_nb5_vnn3_vnnn4p5.npz')
print(f"  prev E_GS = {float(prev['E_ground']):.8f}  (delta = "
      f"{abs(float(prev['E_ground'])-E_GS):.2e})")

# Raw density (incorrect in C6 basis but kept for comparison)
print("\nComputing raw <n_i> (one operator per site, no orbit average) ...")
density_raw = np.zeros(n)
ts = time.time()
for i in range(n):
    nop = hamiltonian([['n', [[1.0, i]]]], [], basis=basis,
                      dtype=np.complex128, **no_checks)
    density_raw[i] = nop.expt_value(psi).real
print(f"  done in {time.time()-ts:.1f}s, sum = {density_raw.sum():.4f}")

# Orbit-averaged density (correct)
print("\nComputing orbit-averaged <(1/6) sum_k n_{sigma^k(i)}> ...")
density_sym = np.zeros(n)
ts = time.time()
# One operator per *orbit* (saves 6x work over per-site)
for orb in orbits:
    coup = [[1.0 / len(orb), s] for s in orb]
    nop = hamiltonian([['n', coup]], [], basis=basis,
                      dtype=np.complex128, **no_checks)
    val = nop.expt_value(psi).real
    density_sym[orb] = val
print(f"  done in {time.time()-ts:.1f}s, sum = {density_sym.sum():.4f}")

# Symmetry diagnostics on the raw density
print("\nC6 symmetry diagnostic on raw density:")
print(f"  {'orbit':>6}  {'min':>10}  {'max':>10}  {'spread':>10}  {'mean':>10}")
total_spread = 0.0
for k, o in enumerate(orbits):
    vs = density_raw[o]
    spread = vs.max() - vs.min()
    total_spread += spread
    print(f"  {k:6d}  {vs.min():10.6f}  {vs.max():10.6f}  "
          f"{spread:10.6f}  {vs.mean():10.6f}")
print(f"  total spread sum: {total_spread:.6f}  "
      f"(should be 0 for true C6-invariant density)")

# Compare raw vs symm site-by-site (sites in same orbit)
print(f"\nmax|raw - orbit_mean| = "
      f"{np.max(np.abs(density_raw - density_sym)):.6f}")
print(f"orbit_mean equals symm: "
      f"{np.allclose(density_sym, np.array([density_raw[o].mean() for o in orbits])[site_to_orbit])}")

# Save corrected data
out_npz = 'data/nb5/kagome_density_nb5_vnn3_vnnn4p5_symm.npz'
os.makedirs(os.path.dirname(out_npz), exist_ok=True)
np.savez(out_npz,
         V_nn=V_nn, V_nnn=V_nnn,
         density=density_sym,
         density_raw=density_raw,
         positions=lat.positions, sublats=lat.sublats,
         L_ground=L_GS_known, E_ground=E_GS,
         orbit_id=site_to_orbit)
print(f"\nSaved data: {out_npz}")

# Plot: 2 rows x 1 col (compare3-style)
fig, axes = plt.subplots(2, 1, figsize=(6.5, 11))

xs, ys = lat.positions[:, 0], lat.positions[:, 1]
ax0 = axes[0]
sc = ax0.scatter(xs, ys, c=density_sym, s=160, cmap='viridis',
                 vmin=0.0, vmax=density_sym.max(),
                 edgecolor='k', linewidth=0.4)
plt.colorbar(sc, ax=ax0, shrink=0.85, label=r'$\langle n_i \rangle$')
ax0.set_aspect('equal')
ax0.set_title(rf"Nb={Nb}, $V_{{nn}}={V_nn:g},\ V_{{nnn}}={V_nnn:g}$"
              f"\n$L_{{gs}}={L_GS_known}$, $E_{{gs}}={E_GS:.4f}$",
              fontsize=12)
ax0.set_xlabel('x'); ax0.set_ylabel('y')
ax0.grid(True, alpha=0.25)

ax1 = axes[1]
r = np.linalg.norm(lat.positions, axis=1)
order = np.argsort(r)
ax1.plot(r[order], density_sym[order], 'o-', ms=5, lw=0.8,
         color='C0', alpha=0.85)
ax1.set_xlabel(r'$|r|$ (site distance from center)')
ax1.set_ylabel(r'$\langle n_i \rangle$')
ax1.set_title(rf"$V_{{nn}}={V_nn:g},\ V_{{nnn}}={V_nnn:g}$", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, density_sym.max() * 1.05)

plt.tight_layout()
out_png = 'figures/nb5/kagome_disk_nb5_density_vnn3_vnnn4p5_compare3style.png'
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f"Saved plot: {out_png}")

# Sublattice stats for the corrected density
print("\nSublattice statistics (corrected, symm-averaged):")
for s, name in enumerate(['A', 'B', 'C']):
    m = lat.sublats == s
    print(f"  sublat {name}: n_sites={m.sum()}, mean<n>={density_sym[m].mean():.4f}, "
          f"max={density_sym[m].max():.4f}, sum={density_sym[m].sum():.4f}")
