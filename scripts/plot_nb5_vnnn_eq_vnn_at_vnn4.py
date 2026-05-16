"""Single-point E-vs-L spectrum at V_nn = V_nnn = 4 from the
vnnn_eq_vnn_2to6 scan (10 lowest states)."""

import os
import numpy as np
import matplotlib.pyplot as plt

from kagome.plot import plot_low_energy_degen

src = 'data/nb5/kagome_disk_nb5_vnnn_eq_vnn_2to6.npz'
d = np.load(src)
V_nn_arr = d['V_nn']
V_nnn_arr = d['V_nnn']

target = 4.0
i = int(np.argmin(np.abs(V_nn_arr - target)))
assert abs(V_nn_arr[i] - target) < 1e-9, f'V_nn={target} not in scan'

E = d['E'][i]
L = d['L'][i]
valid = L != -1
E = E[valid]
L = L[valid]
order = np.argsort(E)
E = E[order]
L = L[order]
E0 = E[0]
print(f'Nb=5 V_nn={V_nn_arr[i]:g} V_nnn={V_nnn_arr[i]:g}, {len(E)} levels:')
for k in range(len(E)):
    print(f'  k={k:2d}  E={E[k]:+.6f}  dE={E[k]-E0:+.6f}  L={L[k]}')

E_rel = E - E0
fig, ax = plt.subplots(figsize=(6, 5))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    mask = L == Ltag
    if mask.any():
        ax.scatter([Ltag] * int(mask.sum()), E_rel[mask],
                   s=80, color=cmap(Ltag), label=f'L={Ltag}',
                   edgecolor='k', linewidth=0.5, zorder=3)
ax.set_xlabel('Angular momentum  L (mod 6)')
ax.set_ylabel(r'$E - E_0$')
ax.set_xticks(range(6))
ax.set_title(rf'Nb=5  $V_{{nn}}=V_{{nnn}}={V_nn_arr[i]:g}$  (lowest {len(E)})')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = f'figures/nb5/kagome_disk_nb5_vnnn_eq_vnn_at_vnn{int(target)}_n{len(E)}.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved scatter: {out}')

fig2, ax2 = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L, E, n_low=len(E), ax=ax2)
ax2.set_title(rf'Nb=5  $V_{{nn}}=V_{{nnn}}={V_nn_arr[i]:g}$  (lowest {len(E)})')
plt.tight_layout()
out2 = (f'figures/nb5/kagome_disk_nb5_vnnn_eq_vnn_at_vnn{int(target)}'
        f'_n{len(E)}_hline.png')
plt.savefig(out2, dpi=180, bbox_inches='tight')
print(f'Saved hline:   {out2}')
