"""Plot the partial Nb=4 V_nn fine-scan in V_nn ∈ [1.0, 1.5] from the
incrementally-saved npz (read-only — safe to run while the scan is still going).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

NPZ = 'kagome_disk_nb4_vnn_scan_fine_0to2.npz'
V_LO, V_HI = 0.0, 1.5

assert os.path.exists(NPZ), f'missing {NPZ}'
d = np.load(NPZ)
V = d['V_nn']
E = d['E']
L = d['L']
done = int(d['done']) if 'done' in d.files else len(V)
print(f'done = {done}/{len(V)}')

idx_in_range = np.where((V >= V_LO) & (V <= V_HI) & (np.arange(len(V)) < done))[0]
print(f'points in [{V_LO}, {V_HI}]: {len(idx_in_range)}')
if len(idx_in_range) == 0:
    raise SystemExit('No completed points in window yet.')
print(f'  V_nn covered: {V[idx_in_range[0]]:.4f} … {V[idx_in_range[-1]]:.4f}')

fig, ax = plt.subplots(figsize=(9, 6))
cmap = plt.get_cmap('tab10')
for Lt in range(6):
    xs, ys = [], []
    for i in idx_in_range:
        m = L[i] == Lt
        xs.extend([V[i]] * int(m.sum()))
        ys.extend(E[i, m].tolist())
    ax.scatter(xs, ys, s=14, alpha=0.8, color=cmap(Lt),
               label=f'L={Lt}', edgecolor='none')

ax.set_xlabel(r'$V_{nn}$')
ax.set_ylabel('E')
ax.set_title(f'Nb=4 spectrum vs $V_{{nn}}$  '
             f'(partial fine scan, $V_{{nn}}\\in[{V_LO},{V_HI}]$, '
             f'{len(idx_in_range)} points)')
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'kagome_disk_nb4_vnn_scan_fine_0to1p5.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved: {out}')
