"""Plot E vs L scatter at a chosen V_nn from the
kagome_disk_nb5_vnnn_1p5_vnn_2to6.npz scan (V_nnn = V_nn / 1.5).

Usage:
    python scripts/plot_nb5_vnnn1p5_spectrum_at_vnn.py [target_vnn]
Default target_vnn = 5.0.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

Nb = 5
target_vnn = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

path = 'kagome_disk_nb5_vnnn_1p5_vnn_2to6.npz'
seg = np.load(path)
done = int(seg['done']) if 'done' in seg.files else len(seg['V_nn'])
V_nn = seg['V_nn'][:done]
V_nnn = seg['V_nnn'][:done]
E = seg['E'][:done]
L = seg['L'][:done]
print(f"loaded {path}: {done} points, V_nn in [{V_nn.min():g}, {V_nn.max():g}]")

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
n_state = E_pick.shape[0]
ax.set_title(f'Nb={Nb} disk spectrum   '
             f'$V_{{nn}}={v_pick:g}$,  $V_{{nnn}}=V_{{nn}}/1.5={vp_pick:.4f}$\n'
             f'lowest {n_state} states{close_note}')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'kagome_disk_nb{Nb}_vnnn1p5_spectrum_vnn{v_pick:g}.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved: {out}")

print("\nLow-lying levels (E - E0, L):")
order2 = np.argsort(E_pick)
for j in order2:
    print(f"  E-E0 = {E_pick[j]-E_pick.min():+.6f}   L = {L_pick[j]}")
