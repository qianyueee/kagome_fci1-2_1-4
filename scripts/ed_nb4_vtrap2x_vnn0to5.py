"""Nb=4, V_trap = 2x default = 0.01, V_nn scan from 0 to 5, 30 points, V_nnn=0."""

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
Nb = 4
n_show = 20
V_trap = 0.005 * 2.0
V_nn_list = np.linspace(0.0, 5.0, 30)

params = ModelParams(V_trap=V_trap)
npz_out = f'kagome_disk_nb{Nb}_vnn_scan_0to5_vtrap2x.npz'
png_out = f'kagome_disk_nb{Nb}_vnn_scan_0to5_vtrap2x.png'

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
        print(f"Resume: {start_idx}/{len(V_nn_list)} done.", flush=True)

t0 = time.time()
for idx, v in enumerate(V_nn_list):
    if idx < start_idx:
        continue
    print(f"\n=== [{idx+1}/{len(V_nn_list)}] V_nn={v:.4f} ===", flush=True)
    E, L = run_ed(lat, params, Nb=Nb, V_nn=v, n_states=n_show)
    k = min(n_show, len(E))
    records_E[idx, :k] = E[:k]
    records_L[idx, :k] = L[:k]
    print(f"  Emin={E[0]: .6f}  L_ground={L[0]}  elapsed={time.time()-t0:.0f}s",
          flush=True)
    np.savez(npz_out, V_nn=V_nn_list, V_trap=np.array(V_trap),
             E=records_E, L=records_L, done=np.array(idx + 1))
print(f"\nTotal scan time: {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(9, 6))
cmap = plt.get_cmap('tab10')
for Lt in range(6):
    xs, ys = [], []
    for i, v in enumerate(V_nn_list):
        mask = records_L[i] == Lt
        xs.extend([v] * int(mask.sum()))
        ys.extend(records_E[i, mask].tolist())
    ax.scatter(xs, ys, s=18, alpha=0.8, color=cmap(Lt),
               label=f'L={Lt}', edgecolor='none')

ax.set_xlabel(r'$V_{nn}$')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} spectrum vs $V_{{nn}}$  '
             f'($V_{{trap}}={V_trap:g}$ = 2× default, '
             f'{len(V_nn_list)} pts, lowest {n_show})')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(png_out, dpi=180, bbox_inches='tight')
print(f"Saved: {png_out}")
