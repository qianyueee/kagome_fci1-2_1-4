"""Single-point spectrum at Nb=5, V_nn=1.8, V_nnn=4
from the running vnnn4_vnn_1to5 scan."""

import os
import numpy as np
import matplotlib.pyplot as plt
from kagome.plot import plot_low_energy_degen

src = 'data/nb5/kagome_disk_nb5_vnnn4_vnn_1to5_disc.npz'
d = np.load(src)
target = 1.8
i = int(np.argmin(np.abs(d['V_nn'] - target)))
assert abs(d['V_nn'][i] - target) < 1e-9

E = d['E'][i]
L = d['L'][i]
mask = L != -1
E = E[mask]; L = L[mask]
order = np.argsort(E)
E = E[order]; L = L[order]
E0 = E[0]

print(f'Nb=5 V_nn={d["V_nn"][i]:g} V_nnn={d["V_nnn"][i]:g}, {len(E)} levels:')
for k in range(len(E)):
    print(f'  k={k:2d}  E={E[k]:+.6f}  dE={E[k]-E0:+.6f}  L={L[k]}')

E_rel = E - E0
cmap = plt.get_cmap('tab10')

fig, ax = plt.subplots(figsize=(6, 5))
for Ltag in range(6):
    m = L == Ltag
    if m.any():
        ax.scatter([Ltag] * int(m.sum()), E_rel[m],
                   s=70, color=cmap(Ltag), label=f'L={Ltag}',
                   edgecolor='k', linewidth=0.4, zorder=3)
ax.set_xlabel('Angular momentum  L (mod 6)')
ax.set_ylabel(r'$E - E_0$')
ax.set_xticks(range(6))
ax.set_title(rf'Nb=5  $V_{{nn}}={d["V_nn"][i]:g}$, '
             rf'$V_{{nnn}}={d["V_nnn"][i]:g}$  (lowest {len(E)})')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = f'figures/nb5/kagome_disk_nb5_vnnn4_at_vnn1p8_n{len(E)}.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved scatter: {out}')

fig2, ax2 = plt.subplots(figsize=(6, 5))
plot_low_energy_degen(L, E, n_low=len(E), ax=ax2)
ax2.set_title(rf'Nb=5  $V_{{nn}}={d["V_nn"][i]:g}$, '
              rf'$V_{{nnn}}={d["V_nnn"][i]:g}$  (lowest {len(E)})')
plt.tight_layout()
out2 = f'figures/nb5/kagome_disk_nb5_vnnn4_at_vnn1p8_n{len(E)}_hline.png'
plt.savefig(out2, dpi=180, bbox_inches='tight')
print(f'Saved hline:   {out2}')
