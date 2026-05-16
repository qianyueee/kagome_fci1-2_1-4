"""Plot E vs L scatter (energy spectrum) at the V_nn point closest to a target,
using the sqrt3 coulomb (V_nnn = V_nn/sqrt(3)) data we have.

Target: V_nn = 5. Closest available in 0-4 scan: V_nn = 4.0.
Once the 4.2-6 ext scan reaches V_nn=5.0 we can rerun.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
Nb = 5
target_vnn = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

segments = [
    'kagome_disk_nb5_coulomb_vnnn_sqrt3_vnn_0to4.npz',
    'kagome_disk_nb5_coulomb_vnnn_sqrt3_vnn_4p2to6.npz',
]

V_nn_all, V_nnn_all, E_all, L_all = [], [], [], []
for path in segments:
    if not os.path.exists(path):
        continue
    seg = np.load(path)
    done = int(seg['done']) if 'done' in seg.files else len(seg['V_nn'])
    if done == 0:
        continue
    V_nn_all.append(seg['V_nn'][:done])
    V_nnn_all.append(seg['V_nnn'][:done])
    E_all.append(seg['E'][:done])
    L_all.append(seg['L'][:done])
    print(f"loaded {path}: {done} points, V_nn in [{seg['V_nn'][:done].min():g}, {seg['V_nn'][:done].max():g}]")

V_nn = np.concatenate(V_nn_all)
V_nnn = np.concatenate(V_nnn_all)
E = np.concatenate(E_all, axis=0)
L = np.concatenate(L_all, axis=0)

order = np.argsort(V_nn)
V_nn = V_nn[order]
V_nnn = V_nnn[order]
E = E[order]
L = L[order]

idx = int(np.argmin(np.abs(V_nn - target_vnn)))
v_pick = V_nn[idx]
vp_pick = V_nnn[idx]
E_pick = E[idx]
L_pick = L[idx]
print(f"\nTarget V_nn = {target_vnn}, picked V_nn = {v_pick:g}, V_nnn = {vp_pick:.4f}")

E_rel = E_pick - E_pick.min()

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap('tab10')
for Ltag in range(6):
    mask = L_pick == Ltag
    if mask.any():
        ax.scatter([Ltag] * int(mask.sum()), E_rel[mask],
                   s=80, color=cmap(Ltag), label=f'L={Ltag}',
                   edgecolor='k', linewidth=0.5, zorder=3)

ax.set_xlabel('Angular momentum  L (mod 6)')
ax.set_ylabel(r'$E - E_0$')
ax.set_xticks(range(6))
close_note = '' if abs(v_pick - target_vnn) < 1e-9 else f'  (target {target_vnn:g}, closest available)'
ax.set_title(f'Nb={Nb} disk spectrum   '
             f'$V_{{nn}}={v_pick:g}$,  $V_{{nnn}}=V_{{nn}}/\\sqrt{{3}}={vp_pick:.4f}$\n'
             f'lowest 20 states{close_note}')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_sqrt3_spectrum_vnn{v_pick:g}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")

print("\nLow-lying levels (E - E0, L):")
order2 = np.argsort(E_pick)
for j in order2[:12]:
    print(f"  E-E0 = {E_pick[j]-E_pick.min():+.6f}   L = {L_pick[j]}")
