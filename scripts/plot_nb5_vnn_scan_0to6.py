"""Merge Nb=5 V_nn 0-2, 2.2-4, 4.2-6 scans into a single 0-6 plot."""

import os
import numpy as np
import matplotlib.pyplot as plt

Nb = 5
n_show = 20

segments = [
    f'kagome_disk_nb{Nb}_vnn_scan_0to2.npz',
    f'kagome_disk_nb{Nb}_vnn_scan_2p2to4.npz',
    f'kagome_disk_nb{Nb}_vnn_scan_4p2to6.npz',
]

V_nn_all, E_all, L_all = [], [], []
for path in segments:
    if not os.path.exists(path):
        print(f"  skip (missing): {path}")
        continue
    seg = np.load(path)
    done = int(seg['done'])
    V_nn_all.append(seg['V_nn'][:done])
    E_all.append(seg['E'][:done])
    L_all.append(seg['L'][:done])
    print(f"  loaded {path}: {done} points")

V_nn = np.concatenate(V_nn_all)
E = np.concatenate(E_all, axis=0)
L = np.concatenate(L_all, axis=0)

order = np.argsort(V_nn)
V_nn = V_nn[order]
E = E[order]
L = L[order]

print(f"Combined: {len(V_nn)} points, V_nn range [{V_nn.min():g}, {V_nn.max():g}]")

fig, ax = plt.subplots(figsize=(13, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for i, v in enumerate(V_nn):
        mask = L[i] == Ltag
        xs.extend([v] * int(mask.sum()))
        ys.extend(E[i, mask].tolist())
    ax.scatter(xs, ys, s=18, alpha=0.8, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')

ax.set_xlabel(r'$V_{nn}$')
ax.set_ylabel('E')
ax.set_title(f'Nb={Nb} low-energy spectrum vs $V_{{nn}}$  '
             f'(combined 0 to 6, {len(V_nn)} pts, lowest {n_show})')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_vnn_scan_0to6.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")

npz_out = f'kagome_disk_nb{Nb}_vnn_scan_0to6.npz'
np.savez(npz_out, V_nn=V_nn, E=E, L=L)
print(f"Saved: {npz_out}")
