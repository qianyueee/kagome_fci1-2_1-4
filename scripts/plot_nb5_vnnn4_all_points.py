"""Generate all 21 single-point E-vs-L spectra from the
Nb=5 V_nnn=4 V_nn=1..5 scan into a dedicated folder."""

import os
import numpy as np
import matplotlib.pyplot as plt

from kagome.plot import plot_low_energy_degen

src = 'data/nb5/kagome_disk_nb5_vnnn4_vnn_1to5_disc.npz'
out_dir = 'figures/nb5/vnnn4_vnn_1to5_spectra'
os.makedirs(out_dir, exist_ok=True)

d = np.load(src)
V_nn_arr = d['V_nn']
V_nnn_arr = d['V_nnn']
E_arr = d['E']
L_arr = d['L']

for i in range(len(V_nn_arr)):
    E = E_arr[i]; L = L_arr[i]
    mask = L != -1
    E = E[mask]; L = L[mask]
    order = np.argsort(E)
    E = E[order]; L = L[order]
    E0 = E[0]

    v_str = f'{V_nn_arr[i]:.1f}'.replace('.', 'p')

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_low_energy_degen(L, E, n_low=len(E), ax=ax)
    ax.set_title(rf'Nb=5  $V_{{nn}}={V_nn_arr[i]:g}$, '
                 rf'$V_{{nnn}}={V_nnn_arr[i]:g}$  '
                 f'(lowest {len(E)})\n'
                 rf'$L_{{GS}}={L[0]}$, $E_{{GS}}={E0:.4f}$, '
                 rf'gap = {E[1]-E0:.4f}')
    plt.tight_layout()
    out = f'{out_dir}/spectrum_vnn{v_str}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[{i+1:2d}/{len(V_nn_arr)}] {out}')

print(f'\nAll {len(V_nn_arr)} spectra in: {out_dir}/')
