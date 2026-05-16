"""Compare old vs new discretized trap per site.

Old: V_trap * (4 * d)**2  with V_trap = 0.005, d = |raw position|
New: round(r_half_sq * 9) * (0.005/9)  with r_half_sq = (4*pos[0])**2 + (4*pos[1])**2
"""

import numpy as np
from kagome import DiskLattice, ModelParams

lat = DiskLattice()
params = ModelParams()
V_trap = params.V_trap  # 0.005

old = np.zeros(lat.n_sites)
new = np.zeros(lat.n_sites)
r_half_sq_vals = np.zeros(lat.n_sites)

for i in range(lat.n_sites):
    pos = lat.positions[i]
    d = np.linalg.norm(pos)
    # half-lattice-unit coordinates: a/2 = 0.25 so multiply by 4
    pos_half = 4.0 * pos
    r_half_sq = pos_half[0] ** 2 + pos_half[1] ** 2
    r_half_sq_vals[i] = r_half_sq

    old[i] = V_trap * (4.0 * d) ** 2
    new[i] = round(r_half_sq * 9) * (V_trap / 9.0)

print(f"V_trap = {V_trap}")
print(f"Sites: {lat.n_sites}")
print(f"\nold trap range: [{old.min():.6f}, {old.max():.6f}]")
print(f"new trap range: [{new.min():.6f}, {new.max():.6f}]")
print(f"sum old: {old.sum():.6f}")
print(f"sum new: {new.sum():.6f}")
print(f"max |old-new|: {np.abs(old-new).max():.6e}")
print(f"mean |old-new|: {np.abs(old-new).mean():.6e}")
print(f"max |old-new|/old (excl. zero): "
      f"{np.max(np.abs(old-new) / np.where(old > 1e-12, old, np.inf)):.4f}")

print("\nPer-site comparison (sorted by |r|):")
order = np.argsort(np.linalg.norm(lat.positions, axis=1))
print(f"{'i':>3}  {'|r|':>8}  {'r_half_sq':>10}  {'round(9 r2)':>12}  "
      f"{'old V':>12}  {'new V':>12}  {'diff':>10}")
for k in order:
    d = np.linalg.norm(lat.positions[k])
    print(f"{k:3d}  {d:8.4f}  {r_half_sq_vals[k]:10.4f}  "
          f"{round(r_half_sq_vals[k]*9):12d}  {old[k]:12.6f}  "
          f"{new[k]:12.6f}  {new[k]-old[k]:+10.6f}")
