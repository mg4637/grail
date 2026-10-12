"""Height map of a static terrain mesh (e.g. stairs) for foot-to-terrain losses.

The map is built once, in the mesh's own (canonical) frame, by rasterising the mesh's triangles onto a
regular XY grid and keeping the highest surface at every cell. Cells not covered by any triangle get
the floor height (the lowest vertex of the mesh, i.e. the base the terrain stands on). Vertical faces
(side walls) have no XY area and are ignored: they only produce the height jump between two cells.

Pure numpy for building; torch only for the differentiable-friendly lookup (the lookup itself is not
differentiated: heights are treated as constants, the loss gradient acts on the foot height).
"""

import numpy as np


class TerrainHeightMap:
    def __init__(self, verts, faces, cell=0.01, margin=0.5):
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        self.cell = float(cell)
        self.floor = float(verts[:, 2].min())

        lo = verts[:, :2].min(axis=0) - margin
        hi = verts[:, :2].max(axis=0) + margin
        self.x0, self.y0 = float(lo[0]), float(lo[1])
        self.nx = int(np.ceil((hi[0] - lo[0]) / self.cell)) + 1
        self.ny = int(np.ceil((hi[1] - lo[1]) / self.cell)) + 1
        H = np.full((self.ny, self.nx), self.floor, dtype=np.float64)

        gx = self.x0 + np.arange(self.nx) * self.cell     # cell-centre coordinates
        gy = self.y0 + np.arange(self.ny) * self.cell
        for tri in faces:
            a, b, c = verts[tri]
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-12:                           # vertical / degenerate in XY
                continue
            xmin, xmax = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
            ymin, ymax = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
            ix = np.where((gx >= xmin - 1e-9) & (gx <= xmax + 1e-9))[0]
            iy = np.where((gy >= ymin - 1e-9) & (gy <= ymax + 1e-9))[0]
            if len(ix) == 0 or len(iy) == 0:
                continue
            X, Y = np.meshgrid(gx[ix], gy[iy])
            l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / det
            l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / det
            l3 = 1.0 - l1 - l2
            inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
            z = l1 * a[2] + l2 * b[2] + l3 * c[2]
            sub = H[np.ix_(iy, ix)]
            H[np.ix_(iy, ix)] = np.where(inside, np.maximum(sub, z), sub)
        self.H = H
        self._torch_cache = {}

    def height_np(self, xy):
        """Nearest-cell height for numpy points (n, 2)."""
        ix = np.clip(np.rint((xy[:, 0] - self.x0) / self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip(np.rint((xy[:, 1] - self.y0) / self.cell).astype(int), 0, self.ny - 1)
        return self.H[iy, ix]

    def height(self, xy):
        """Nearest-cell height for torch points (..., 2); returns a constant (no gradient)."""
        import torch

        key = (xy.device, xy.dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = torch.as_tensor(self.H, device=xy.device, dtype=xy.dtype)
        Ht = self._torch_cache[key]
        with torch.no_grad():
            ix = torch.clamp(torch.round((xy[..., 0] - self.x0) / self.cell).long(), 0, self.nx - 1)
            iy = torch.clamp(torch.round((xy[..., 1] - self.y0) / self.cell).long(), 0, self.ny - 1)
            return Ht[iy, ix]
