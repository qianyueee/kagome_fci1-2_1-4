"""Nb=2 hard-core boson ED on 72-site Kagome disk."""

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
Nb = 2

E, L = run_ed(lat, params, Nb=Nb, V_nn=0.0, n_states=None)

E0 = E[0]
print(f"\nLowest 12 states (Nb={Nb}):")
for k in range(12):
    print(f"  k={k:2d}  E={E[k]: .6f}  dE={E[k]-E0: .6f}  L_tot={L[k]}")

fig, ax = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L, E, n_low=50, ax=ax)
ax.set_title(f'Nb={Nb}, V_nn=0')

plt.tight_layout()
out = 'kagome_disk_nb2_new.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nSaved: {out}")
