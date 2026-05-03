"""Nb=5 V_nnn scan on 72-site Kagome disk: V_nn=2.0 fixed, V_nnn from 0 to 2, 11 points (step 0.2)."""

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
n_show = 20

V_nn_fix = 2.0
V_nnn_list = np.linspace(0.0, 2.0, 11)
tag = 'vnn2_vnnn0to2'
npz_out = f'kagome_disk_nb{Nb}_{tag}.npz'

records_E = np.full((len(V_nnn_list), n_show), np.nan)
records_L = np.full((len(V_nnn_list), n_show), -1, dtype=int)

start_idx = 0
if os.path.exists(npz_out):
    prev = np.load(npz_out)
    if (prev['V_nnn'].shape == V_nnn_list.shape and
            np.allclose(prev['V_nnn'], V_nnn_list)
            and float(prev['V_nn']) == V_nn_fix):
        start_idx = int(prev['done'])
        records_E[:start_idx] = prev['E'][:start_idx]
        records_L[:start_idx] = prev['L'][:start_idx]
        print(f"Resume: {start_idx}/{len(V_nnn_list)} done.", flush=True)
    else:
        print("Existing npz has different grid; restarting from 0.", flush=True)

t0 = time.time()
for idx, vp in enumerate(V_nnn_list):
    if idx < start_idx:
        continue
    print(f"\n=== [{idx+1}/{len(V_nnn_list)}] V_nn={V_nn_fix:.3f}  V_nnn={vp:.4f} ===",
          flush=True)
    E, L = run_ed(lat, params, Nb=Nb, V_nn=V_nn_fix, V_nnn=vp, n_states=n_show)
    k = min(n_show, len(E))
    records_E[idx, :k] = E[:k]
    records_L[idx, :k] = L[:k]
    print(f"  Emin={E[0]: .6f}  L_ground={L[0]}  elapsed={time.time()-t0:.0f}s",
          flush=True)
    np.savez(npz_out, V_nn=np.array(V_nn_fix), V_nnn=V_nnn_list,
             E=records_E, L=records_L, done=np.array(idx + 1))

print(f"\nTotal scan time: {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(9, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for i, vp in enumerate(V_nnn_list):
        mask = records_L[i] == Ltag
        xs.extend([vp] * int(mask.sum()))
        ys.extend(records_E[i, mask].tolist())
    ax.scatter(xs, ys, s=18, alpha=0.8, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')

ax.set_xlabel(r'$V_{nnn}$')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} spectrum vs $V_{{nnn}}$  '
             f'($V_{{nn}}={V_nn_fix:g}$ fixed, '
             f'{len(V_nnn_list)} pts, lowest {n_show})')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_{tag}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")
print(f"Saved data: {npz_out}")
