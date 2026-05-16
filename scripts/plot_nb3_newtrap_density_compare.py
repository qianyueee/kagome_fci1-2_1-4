"""Plot Nb=3 ground-state site density for three interaction strengths.

Cases (all newtrap, hardcore bosons):
  (i)   V_nn=0,   V_nnn=0     (pure hardcore)
  (ii)  V_nn=1,   V_nnn=1.5
  (iii) V_nn=3,   V_nnn=4.5

Top row: lattice-coloured density map for each case.
Bottom row: per-site density vs |r| (no averaging overlays).
"""

import numpy as np
import matplotlib.pyplot as plt

cases = [
    ('data/nb3/kagome_disk_nb3_newtrap_density_freehc.npz',
     r'$V_{nn}=0,\ V_{nnn}=0$ (pure hard-core)'),
    ('data/nb3/kagome_disk_nb3_newtrap_density_vnn1_vnnn1.5.npz',
     r'$V_{nn}=1,\ V_{nnn}=1.5$'),
    ('data/nb3/kagome_disk_nb3_newtrap_density_vnn3_vnnn4.5.npz',
     r'$V_{nn}=3,\ V_{nnn}=4.5$'),
]

data = []
for path, title in cases:
    d = np.load(path, allow_pickle=True)
    data.append({
        'density': d['density'],
        'positions': d['positions'],
        'sublats': d['sublats'],
        'L_ground': int(d['L_ground']),
        'E_ground': float(d['E_ground']),
        'title': title,
    })

# Shared color scale across the three site maps.
vmax = max(d['density'].max() for d in data)
vmin = 0.0

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for k, d in enumerate(data):
    xs, ys = d['positions'][:, 0], d['positions'][:, 1]
    ax = axes[0, k]
    sc = ax.scatter(xs, ys, c=d['density'], s=160, cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    edgecolor='k', linewidth=0.4)
    plt.colorbar(sc, ax=ax, shrink=0.85, label=r'$\langle n_i \rangle$')
    ax.set_aspect('equal')
    ax.set_title(f"Nb=3, {d['title']}\n"
                 f"$L_{{gs}}={d['L_ground']}$,  $E_{{gs}}={d['E_ground']:.4f}$",
                 fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.25)

for k, d in enumerate(data):
    r = np.linalg.norm(d['positions'], axis=1)
    order = np.argsort(r)
    ax = axes[1, k]
    ax.plot(r[order], d['density'][order], 'o-', ms=5, lw=0.8,
            color='C0', alpha=0.85)
    ax.set_xlabel(r'$|r|$ (site distance from center)')
    ax.set_ylabel(r'$\langle n_i \rangle$')
    ax.set_title(d['title'], fontsize=11)
    ax.grid(True, alpha=0.3)

# Share y-axis on the bottom row for a fair visual comparison.
ymax = max(d['density'].max() for d in data) * 1.05
for k in range(3):
    axes[1, k].set_ylim(0, ymax)

plt.tight_layout()
out_png = 'figures/nb3/kagome_disk_nb3_newtrap_density_compare3.png'
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f"Saved: {out_png}")
