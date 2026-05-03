"""Nb=3 hard-core boson ED on 72-site Kagome disk at V_nn=2."""

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
V_nn = 2.0

E, L = run_ed(lat, params, Nb=Nb, V_nn=V_nn, n_states=50)

E0 = E[0]
print(f"\nLowest 30 states (Nb={Nb}, V_nn={V_nn}):")
for k in range(min(30, len(E))):
    print(f"  k={k:2d}  E={E[k]: .6f}  dE={E[k]-E0: .6f}  L_tot={L[k]}")

fig, ax = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L, E, n_low=50, ax=ax)
ax.set_title(f'Nb={Nb}, V_nn={V_nn}')

plt.tight_layout()
out = f'kagome_disk_nb3_vnn{V_nn:g}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nSaved: {out}")
