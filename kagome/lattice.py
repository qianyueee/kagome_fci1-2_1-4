"""72-site C6-symmetric Kagome disk lattice construction."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Kagome lattice vectors and orbital positions
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2.0])
ORB_POS = np.array([
    [0.0, 0.0],              # A
    [0.5, 0.0],              # B
    [0.25, np.sqrt(3) / 4],  # C
])
SUBLAT_NAMES = ['A', 'B', 'C']

# C6 symmetry center: hexagonal hole center
C0 = np.array([0.75, np.sqrt(3) / 4.0])

# Characteristic distances
NN_DIST = 0.5
NNN_DIST = np.sqrt(3) / 2.0


class DiskLattice:
    """72-site C6-symmetric Kagome disk.

    Attributes:
        positions : (72, 2) array of site coordinates centred on C0.
        sublats   : (72,) int array, 0/1/2 for A/B/C.
        n_sites   : 72.
        sigma     : (72,) int array — C6 permutation (CW 60 deg).
        D         : (72, 72) distance matrix.
        nn_pairs  : set of (i, j) with i < j for nearest-neighbour pairs.
        nnn_pairs : set of (i, j) with i < j for next-nearest-neighbour pairs.
    """

    N_SITES = 72

    def __init__(self):
        self.positions, self.sublats = self._build_disk()
        self.n_sites = len(self.positions)
        self.D = cdist(self.positions, self.positions)
        self.sigma = self._build_c6_permutation()
        self.nn_pairs, self.nnn_pairs = self._find_neighbors()

    def _build_disk(self):
        N_max = 10
        all_pos, all_sub = [], []
        for n1 in range(-N_max, N_max + 1):
            for n2 in range(-N_max, N_max + 1):
                for s in range(3):
                    all_pos.append(n1 * A1 + n2 * A2 + ORB_POS[s] - C0)
                    all_sub.append(s)
        all_pos = np.array(all_pos)
        all_sub = np.array(all_sub)

        dist = np.linalg.norm(all_pos, axis=1)
        order = np.argsort(dist)
        ds = dist[order]
        assert ds[self.N_SITES - 1] + 1e-8 < ds[self.N_SITES], (
            f"72-site boundary ambiguous: d[71]={ds[71]:.6f}, d[72]={ds[72]:.6f}")
        R_cut = 0.5 * (ds[self.N_SITES - 1] + ds[self.N_SITES])

        sel = dist < R_cut
        return all_pos[sel], all_sub[sel].astype(int)

    def _build_c6_permutation(self):
        # CW 60 deg so eigenvalue on |m> is exp(+i 2pi m/6)
        theta = -np.pi / 3
        Rmat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
        rotated = self.positions @ Rmat.T
        tree = cKDTree(self.positions)
        err, sigma = tree.query(rotated, k=1)
        assert err.max() < 1e-8, f"C6 broken: max err = {err.max():.2e}"
        # Verify order 6
        P = sigma.copy()
        for _ in range(5):
            P = sigma[P]
        assert np.all(P == np.arange(self.n_sites)), "C6^6 != identity"
        return sigma.astype(np.int32)

    def _find_neighbors(self):
        tol = 1e-6
        nn, nnn = set(), set()
        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                d = self.D[i, j]
                if abs(d - NN_DIST) < tol:
                    nn.add((i, j))
                elif abs(d - NNN_DIST) < tol:
                    nnn.add((i, j))
        return nn, nnn
