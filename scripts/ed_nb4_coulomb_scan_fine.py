"""Nb=4 V_nn + V_nnn coupled fine scan (1/r^2 ratio) on 72-site Kagome disk.

V_nnn = V_nn / 3, i.e. interaction strength ~ 1/r^2 truncated to NNN.
Scan V_nn from 0 to 2, 101 points.
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
Nb = 4
n_show = 20

V_nn_list = np.linspace(0.0, 2.0, 101)
V_nnn_list = V_nn_list / 3.0
npz_out = f'kagome_disk_nb{Nb}_coulomb_scan_fine_0to2.npz'

records_E = np.full((len(V_nn_list), n_show), np.nan)
records_L = np.full((len(V_nn_list), n_show), -1, dtype=int)

start_idx = 0
if os.path.exists(npz_out):
    prev = np.load(npz_out)
    if (prev['V_nn'].shape == V_nn_list.shape and
            np.allclose(prev['V_nn'], V_nn_list)):
        start_idx = int(prev['done'])
        records_E[:start_idx] = prev['E'][:start_idx]
        records_L[:start_idx] = prev['L'][:start_idx]
        print(f"Resume: {start_idx}/{len(V_nn_list)} points already done.",
              flush=True)
    else:
        print("Existing npz has different V_nn grid; restarting from 0.",
              flush=True)

t0 = time.time()
for idx, (v, vp) in enumerate(zip(V_nn_list, V_nnn_list)):
    if idx < start_idx:
        continue
    print(f"\n=== [{idx+1}/{len(V_nn_list)}] V_nn={v:.4f}  V_nnn={vp:.4f} ===",
          flush=True)
    E, L = run_ed(lat, params, Nb=Nb, V_nn=v, V_nnn=vp, n_states=n_show)
    k = min(n_show, len(E))
    records_E[idx, :k] = E[:k]
    records_L[idx, :k] = L[:k]
    print(f"  Emin={E[0]: .6f}  L_ground={L[0]}  elapsed={time.time()-t0:.0f}s",
          flush=True)
    np.savez(npz_out, V_nn=V_nn_list, V_nnn=V_nnn_list,
             E=records_E, L=records_L, done=np.array(idx + 1))

print(f"\nTotal scan time: {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for i, v in enumerate(V_nn_list):
        mask = records_L[i] == Ltag
        xs.extend([v] * int(mask.sum()))
        ys.extend(records_E[i, mask].tolist())
    ax.scatter(xs, ys, s=10, alpha=0.75, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')

ax.set_xlabel(r'$V_{nn}$   ($V_{nnn}=V_{nn}/3$, i.e. $V\propto 1/r^2$)')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} low-energy spectrum vs Coulomb-like V  '
             f'(fine scan 0 to 2, lowest {n_show} states)')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_coulomb_scan_fine_0to2.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")
print(f"Saved data: {npz_out}")
