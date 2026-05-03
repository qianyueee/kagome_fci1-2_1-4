"""Nb=4 V_nn scan (0 to 2, 51 points, V_nnn=0) at three V_trap values:
1.5x, 2x, 2.5x of the default V_trap=0.005.

Per-V_trap npz/png; resume via the saved 'done' counter.
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
Nb = 4
n_show = 20
V_nn_list = np.linspace(0.0, 2.0, 51)
V_trap_default = 0.005
trap_multipliers = [1.5, 2.0, 2.5]


def run_one_vtrap(mult):
    V_trap = V_trap_default * mult
    tag = f'vtrap{mult:g}x'
    npz_out = f'kagome_disk_nb{Nb}_vnn_scan_fine_0to2_{tag}.npz'
    png_out = f'kagome_disk_nb{Nb}_vnn_scan_fine_0to2_{tag}.png'

    params = ModelParams(V_trap=V_trap)

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
            print(f"[{tag}] resume: {start_idx}/{len(V_nn_list)} done.",
                  flush=True)

    print(f"\n########## {tag}  V_trap={V_trap:.4f} "
          f"({len(V_nn_list)} points) ##########", flush=True)

    t0 = time.time()
    for idx, v in enumerate(V_nn_list):
        if idx < start_idx:
            continue
        print(f"\n=== [{tag}  {idx+1}/{len(V_nn_list)}] V_nn={v:.4f} ===",
              flush=True)
        E, L = run_ed(lat, params, Nb=Nb, V_nn=v, n_states=n_show)
        k = min(n_show, len(E))
        records_E[idx, :k] = E[:k]
        records_L[idx, :k] = L[:k]
        print(f"  Emin={E[0]: .6f}  L_ground={L[0]}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
        np.savez(npz_out, V_nn=V_nn_list, V_trap=np.array(V_trap),
                 E=records_E, L=records_L, done=np.array(idx + 1))
    print(f"[{tag}] total scan time: {time.time()-t0:.1f}s", flush=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10')
    for Lt in range(6):
        xs, ys = [], []
        for i, v in enumerate(V_nn_list):
            mask = records_L[i] == Lt
            xs.extend([v] * int(mask.sum()))
            ys.extend(records_E[i, mask].tolist())
        ax.scatter(xs, ys, s=14, alpha=0.8, color=cmap(Lt),
                   label=f'L={Lt}', edgecolor='none')

    ax.set_xlabel(r'$V_{nn}$')
    ax.set_ylabel('E')
    ax.set_title(f'Nb={Nb} spectrum vs $V_{{nn}}$  '
                 f'($V_{{trap}}={V_trap:g}$ = {mult:g}× default, '
                 f'{len(V_nn_list)} pts, lowest {n_show})')
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[{tag}] saved: {png_out}", flush=True)


for mult in trap_multipliers:
    run_one_vtrap(mult)

print("\nALL DONE.")
