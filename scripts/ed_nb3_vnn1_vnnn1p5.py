"""Nb=3 hard-core boson ED on 72-site Kagome disk at V_nn=1, V_nnn=1.5.
Plot lowest-100 E vs L horizontal-line style (kagome_disk_nb3_new format).
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')

import numpy as np
import matplotlib.pyplot as plt

from kagome import DiskLattice, ModelParams
from kagome.ed import run_ed
from kagome.plot import plot_low_energy_degen

lat = DiskLattice()
params = ModelParams()
Nb = 3
V_nn = 1.0
V_nnn = 1.5
n_show = 100

E, L = run_ed(lat, params, Nb=Nb, V_nn=V_nn, V_nnn=V_nnn, n_states=n_show)
k = min(n_show, len(E))
E = E[:k]
L = L[:k]

E0 = E[0]
print(f"\nLowest {k} states (Nb={Nb}, V_nn={V_nn}, V_nnn={V_nnn}):")
for j in range(k):
    print(f"  k={j:3d}  E={E[j]: .6f}  dE={E[j]-E0: .6f}  L_tot={L[j]}")

npz_out = f'kagome_disk_nb{Nb}_vnn1_vnnn1p5_n{n_show}.npz'
np.savez(npz_out, V_nn=V_nn, V_nnn=V_nnn, E=E, L=L)
print(f"\nSaved data: {npz_out}")

fig, ax = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L, E, n_low=n_show, ax=ax)
ax.set_title(f'Nb={Nb}, V_nn={V_nn:g}, V_nnn={V_nnn:g}')

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_vnn1_vnnn1p5_n{n_show}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")
