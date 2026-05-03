"""Merge Nb=5 V_nn 0-2 (11 pts) and 2.2-4 (10 pts) scans into a single 0-4 plot."""

import numpy as np
import matplotlib.pyplot as plt

Nb = 5
n_show = 20

low = np.load(f'kagome_disk_nb{Nb}_vnn_scan_0to2.npz')
high = np.load(f'kagome_disk_nb{Nb}_vnn_scan_2p2to4.npz')

# Use only completed rows from each segment
done_low = int(low['done'])
done_high = int(high['done'])

V_nn = np.concatenate([low['V_nn'][:done_low], high['V_nn'][:done_high]])
E = np.concatenate([low['E'][:done_low], high['E'][:done_high]], axis=0)
L = np.concatenate([low['L'][:done_low], high['L'][:done_high]], axis=0)

order = np.argsort(V_nn)
V_nn = V_nn[order]
E = E[order]
L = L[order]

print(f"Combined: {len(V_nn)} points, V_nn range [{V_nn.min():g}, {V_nn.max():g}]")

fig, ax = plt.subplots(figsize=(11, 6))
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
             f'(combined 0 to 4, {len(V_nn)} pts, lowest {n_show})')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_vnn_scan_0to4.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")

npz_out = f'kagome_disk_nb{Nb}_vnn_scan_0to4.npz'
np.savez(npz_out, V_nn=V_nn, E=E, L=L)
print(f"Saved: {npz_out}")
