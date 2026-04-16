"""Nb=4 hard-core boson ED on 72-site Kagome disk."""

import numpy as np
import matplotlib.pyplot as plt

from kagome import DiskLattice, ModelParams
from kagome.ed import run_ed
from kagome.plot import plot_low_energy_degen

lat = DiskLattice()
params = ModelParams()
Nb = 4

E, L = run_ed(lat, params, Nb=Nb, n_states=50)

E0 = E[0]
print(f"\nLowest 30 states (Nb={Nb}):")
for k in range(min(30, len(E))):
    print(f"  k={k:2d}  E={E[k]: .6f}  dE={E[k]-E0: .6f}  L_tot={L[k]}")

fig, ax = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L, E, n_low=50, ax=ax)
ax.set_title(f'Nb={Nb}, V_nn=0')

plt.tight_layout()
out = 'kagome_disk_nb4.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nSaved: {out}")
