"""Nb=3 V_nn scan on 72-site Kagome disk: lowest 30 states vs V_nn."""

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
Nb = 3
n_show = 30

V_nn_list = np.linspace(0.0, 5.0, 26)

records = []  # list of (V_nn, E_arr, L_arr) truncated to n_show
t0 = time.time()
for v in V_nn_list:
    print(f"\n=== V_nn={v:.3f} ===", flush=True)
    E, L = run_ed(lat, params, Nb=Nb, V_nn=v, n_states=n_show)
    E = E[:n_show]
    L = L[:n_show]
    records.append((v, E, L))
    print(f"  Emin={E[0]: .6f}  L_ground={L[0]}", flush=True)

print(f"\nTotal scan time: {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for v, E, L in records:
        mask = L == Ltag
        xs.extend([v] * int(mask.sum()))
        ys.extend(E[mask].tolist())
    ax.scatter(xs, ys, s=14, alpha=0.75, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')

ax.set_xlabel(r'$V_{nn}$')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} low-energy spectrum vs $V_{{nn}}$  (lowest {n_show} states)')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'kagome_disk_nb3_vnn_scan_0to5.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")
