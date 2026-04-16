"""Single-particle spectrum of the 72-site Kagome disk.

Replaces kagome_disk.py — same physics, uses kagome/ library.
"""

import numpy as np
import matplotlib.pyplot as plt

from kagome import DiskLattice, ModelParams
from kagome.hamiltonian import build_single_particle_H
from kagome.symmetry import c6_site_matrix, classify_angular_momentum
from kagome.plot import plot_disk_geometry, plot_spectrum, plot_low_energy

lat = DiskLattice()
params = ModelParams()

print(f"N = {lat.n_sites}, sublattice counts: "
      f"A={np.sum(lat.sublats==0)}, B={np.sum(lat.sublats==1)}, "
      f"C={np.sum(lat.sublats==2)}")

H = build_single_particle_H(lat, params)
C6 = c6_site_matrix(lat)
print(f"Hermiticity error: {np.max(np.abs(H - H.conj().T)):.2e}")
print(f"[H, C6] error:     {np.max(np.abs(H @ C6 - C6 @ H)):.2e}")

evals, evecs = np.linalg.eigh(H)
L = classify_angular_momentum(evals, evecs, C6)

print("\nLowest 12 states:")
for k in range(12):
    print(f"  k={k:2d}  E={evals[k]: .6f}  L={L[k]}")

# --- plot ---
fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])

ax0 = fig.add_subplot(gs[0, 0])
plot_disk_geometry(lat, ax=ax0)
ax0.set_title(f'(a) Disk lattice, N={lat.n_sites}')

ax1 = fig.add_subplot(gs[0, 1])
plot_spectrum(L, evals, ax=ax1)
ax1.set_title('(b) single-particle spectrum')

ax2 = fig.add_subplot(gs[0, 2])
plot_low_energy(L, evals, ax=ax2)
ax2.set_title('(c) low-energy states')

plt.tight_layout()
out = 'kagome_disk_new.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
