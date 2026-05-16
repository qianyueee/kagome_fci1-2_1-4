"""Quick E-vs-L plot for Nb=5, V_nn=4.6, V_nnn=3.0667 (V_nn/1.5),
using sector-minimum energies parsed from the log.
"""

import numpy as np
import matplotlib.pyplot as plt

V_nn = 4.6
V_nnn = V_nn / 1.5

# From logs/nb5_vnnn1p5_vnn_4to6_disc.log lines 32-39
L_arr = np.arange(6)
E_min = np.array([-10.235152, -10.196902, -10.174680,
                  -10.167260, -10.171506, -10.240172])
E_rel = E_min - E_min.min()
L_ground = int(L_arr[np.argmin(E_min)])

cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(6, 5))
for L in L_arr:
    ax.scatter([L], [E_rel[L]], s=120, color=cmap(L),
               edgecolor='k', linewidth=0.6, zorder=3)

ax.set_xlabel('Angular momentum  L (mod 6)')
ax.set_ylabel(r'$E - E_0$')
ax.set_xticks(L_arr)
ax.set_title(f'Nb=5 sector minima  '
             rf'$V_{{nn}}={V_nn:g}$, $V_{{nnn}}=V_{{nn}}/1.5={V_nnn:.4f}$'
             '\n(parsed from log; one state per sector)')
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.97, f'GS at L={L_ground}', transform=ax.transAxes,
        va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
out = 'figures/nb5/kagome_disk_nb5_vnn4p6_sector_minima.png'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")

print("\nSector minima (V_nn=4.6, V_nnn=3.0667):")
for L in L_arr:
    print(f"  L={L}  E_min={E_min[L]:+.6f}  E-E0={E_rel[L]:+.6f}")
