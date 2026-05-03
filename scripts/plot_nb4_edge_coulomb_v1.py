"""Edge excitation plot for Nb=4 at V_nn=1, V_nnn=1/3 from the
existing Coulomb-like scan npz.  Same style as the V_nn=2 / V_nn=10 plots.
"""

import numpy as np
import matplotlib.pyplot as plt

NPZ = 'kagome_disk_nb4_coulomb_scan_0to20.npz'
TARGET = 1.0

d = np.load(NPZ)
V_nn = d['V_nn']
V_nnn = d['V_nnn']
E_all = d['E']
L_all = d['L']

idx = int(np.argmin(np.abs(V_nn - TARGET)))
v = float(V_nn[idx])
vp = float(V_nnn[idx])
print(f"V_nn={v}  V_nnn={vp:.4f}  (idx={idx})")

E = E_all[idx]
L = L_all[idx]
mask = np.isfinite(E) & (L >= 0)
E = E[mask]
L = L[mask]
order = np.argsort(E)
E = E[order]
L = L[order]

E0 = E[0]
print(f"\nLowest {min(20, len(E))} states:")
for k in range(min(20, len(E))):
    print(f"  k={k:2d}  E={E[k]: .6f}  dE={E[k]-E0: .6f}  L={L[k]}")

fig, ax = plt.subplots(figsize=(7, 6))
hw = 0.3
n_show = min(20, len(E))
for k in range(n_show):
    ax.hlines(E[k], L[k] - hw, L[k] + hw, colors='k', linewidths=1.6)

n_label = min(12, n_show)
for k in range(n_label):
    ax.annotate(str(k), (L[k] + hw + 0.02, E[k]),
                fontsize=8, va='center', color='gray')

ax.set_xlabel('L')
ax.set_ylabel('E')
ax.set_xticks(range(6))
ax.set_xlim(-0.6, 5.6)
ax.set_title(f'Nb=4 edge spectrum  V_nn={v}, V_nnn={vp:.3f}')
ax.grid(True, alpha=0.3)

e_lo = E[0] - 0.005
e_hi = E[min(11, n_show - 1)] + 0.01
ax.set_ylim(e_lo, e_hi)

plt.tight_layout()
out = f'kagome_disk_nb4_edge_vnn1_coulomb.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nSaved: {out}")
