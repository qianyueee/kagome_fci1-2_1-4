"""One-off: Nb=5, V_nn=5, V_nnn=V_nn/1.5=10/3, lowest 30 eigenstates per sector.
Save spectrum and plot E vs L.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')

import time
import numpy as np
import matplotlib.pyplot as plt

from kagome import DiskLattice, ModelParams
from kagome.ed import run_ed

lat = DiskLattice()
params = ModelParams()
Nb = 5
n_show = 30

V_nn = 5.0
V_nnn = V_nn / 1.5

t0 = time.time()
E, L = run_ed(lat, params, Nb=Nb, V_nn=V_nn, V_nnn=V_nnn, n_states=n_show)
print(f"ED done in {time.time()-t0:.1f}s, got {len(E)} levels")

# Keep lowest 30 globally (run_ed already sorts).
k = min(n_show, len(E))
E = E[:k]
L = L[:k]

npz_out = f'kagome_disk_nb{Nb}_vnnn1p5_vnn5_n30.npz'
np.savez(npz_out, V_nn=V_nn, V_nnn=V_nnn, E=E, L=L)
print(f"Saved data: {npz_out}")

E_rel = E - E.min()

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    mask = L == Ltag
    if mask.any():
        ax.scatter([Ltag] * int(mask.sum()), E_rel[mask],
                   s=70, color=cmap(Ltag), label=f'L={Ltag}',
                   edgecolor='k', linewidth=0.5, zorder=3)

ax.set_xlabel('Angular momentum  L (mod 6)')
ax.set_ylabel(r'$E - E_0$')
ax.set_xticks(range(6))
ax.set_title(f'Nb={Nb} disk spectrum   '
             f'$V_{{nn}}={V_nn:g}$,  $V_{{nnn}}=V_{{nn}}/1.5={V_nnn:.4f}$\n'
             f'lowest {k} states')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_vnnn1p5_spectrum_vnn5_n30.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved plot: {out}")

print("\nAll 30 lowest levels (E - E0, L):")
for j in range(k):
    print(f"  E-E0 = {E_rel[j]:+.6f}   L = {L[j]}")
