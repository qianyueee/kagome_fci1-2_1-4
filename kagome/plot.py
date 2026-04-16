"""Plotting utilities for Kagome disk spectra."""

import numpy as np
import matplotlib.pyplot as plt

SUBLAT_COLORS = ['#d62728', '#2ca02c', '#1f77b4']
SUBLAT_NAMES = ['A', 'B', 'C']


def plot_disk_geometry(lattice, ax=None):
    """Plot disk lattice with sublattice colours and NN bonds."""
    if ax is None:
        _, ax = plt.subplots()
    for i, j in lattice.nn_pairs:
        ax.plot([lattice.positions[i, 0], lattice.positions[j, 0]],
                [lattice.positions[i, 1], lattice.positions[j, 1]],
                'k-', lw=0.4, alpha=0.5)
    for s in range(3):
        m = lattice.sublats == s
        ax.scatter(lattice.positions[m, 0], lattice.positions[m, 1],
                   c=SUBLAT_COLORS[s], s=38, label=SUBLAT_NAMES[s],
                   edgecolor='k', linewidth=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    return ax


def _draw_levels(ax, L_values, energies, hw=0.3, lw=1.2, color='k'):
    """Draw each energy level as a short horizontal line centred on its L."""
    for L, E in zip(L_values, energies):
        ax.hlines(E, L - hw, L + hw, colors=color, linewidths=lw)


def plot_spectrum(L_values, energies, ax=None, **kwargs):
    """E vs angular momentum — horizontal-line style."""
    if ax is None:
        _, ax = plt.subplots()
    _draw_levels(ax, L_values, energies, hw=0.3, lw=1.0,
                 color=kwargs.get('color', 'k'))
    ax.set_xlabel('L (mod 6)')
    ax.set_ylabel('E')
    ax.set_xticks(range(6))
    ax.grid(True, alpha=0.3)
    return ax


def plot_low_energy(L_values, energies, n_low=12, ax=None, **kwargs):
    """Low-energy zoom with state index labels — horizontal-line style."""
    if ax is None:
        _, ax = plt.subplots()
    Ls = L_values[:n_low]
    Es = energies[:n_low]
    _draw_levels(ax, Ls, Es, hw=0.3, lw=1.8,
                 color=kwargs.get('color', 'k'))
    for k in range(n_low):
        ax.annotate(str(k), (Ls[k] + 0.32, Es[k]),
                    fontsize=8, va='center')
    ax.set_xlabel('L (mod 6)')
    ax.set_ylabel('E')
    ax.set_xticks(range(6))
    ax.grid(True, alpha=0.3)
    return ax


def _group_by_degeneracy(energies_sorted, indices_sorted, tol):
    """Group sorted energies into quasi-degenerate clusters.

    Args:
        energies_sorted: energies sorted ascending.
        indices_sorted:  original indices corresponding to energies_sorted.
        tol: energy tolerance for grouping.

    Returns list of (E_center, count, member_indices) tuples.
    """
    if len(energies_sorted) == 0:
        return []
    groups = []
    cur_E = [energies_sorted[0]]
    cur_idx = [indices_sorted[0]]
    for e, idx in zip(energies_sorted[1:], indices_sorted[1:]):
        if e - cur_E[-1] < tol:
            cur_E.append(e)
            cur_idx.append(idx)
        else:
            groups.append((np.mean(cur_E), len(cur_E), list(cur_idx)))
            cur_E = [e]
            cur_idx = [idx]
    groups.append((np.mean(cur_E), len(cur_E), list(cur_idx)))
    return groups


def plot_low_energy_degen(L_values, energies, n_low=40, ax=None,
                          hw=0.3, lw=1.2, color='k'):
    """Low-energy spectrum — horizontal-line style with grid."""
    if ax is None:
        _, ax = plt.subplots()
    Ls = L_values[:n_low]
    Es = energies[:n_low]
    _draw_levels(ax, Ls, Es, hw=hw, lw=lw, color=color)
    ax.set_xlabel('L')
    ax.set_ylabel('E')
    ax.set_xticks(range(6))
    ax.grid(True, alpha=0.3)
    return ax
