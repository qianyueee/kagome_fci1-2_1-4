"""Plot Nb=5, V_nnn = V_nn scan (first 20 completed points, V_nn=2..5.8)."""

import os
import numpy as np
import matplotlib.pyplot as plt

src = 'data/nb5/kagome_disk_nb5_vnnn_eq_vnn_2to6.npz'
d = np.load(src)
done = int(d['done'])
n_show = min(20, done)
V_nn = d['V_nn'][:n_show]
E = d['E'][:n_show]
L = d['L'][:n_show]
print(f'Plotting first {n_show} points (done={done}), V_nn={V_nn[0]}..{V_nn[-1]}')

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    xs, ys = [], []
    for i, v in enumerate(V_nn):
        mask = L[i] == Ltag
        xs.extend([v] * int(mask.sum()))
        ys.extend(E[i, mask].tolist())
    ax.scatter(xs, ys, s=20, alpha=0.85, color=cmap(Ltag),
               label=f'L={Ltag}', edgecolor='none')

E_gs = E[:, 0]
ax.plot(V_nn, E_gs, '-', color='k', lw=0.8, alpha=0.5, zorder=1)

ax.set_xlabel(r'$V_{nn}\;(=V_{nnn})$')
ax.set_ylabel('E')
ax.set_title(f'Nb=5  $V_{{nnn}}=V_{{nn}}$ scan, first {n_show} pts '
             rf'($V_{{nn}}={V_nn[0]:g}\to{V_nn[-1]:g}$, lowest 10)')
ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'figures/nb5/kagome_disk_nb5_vnnn_eq_vnn_2to5p8_first20.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved: {out}')
