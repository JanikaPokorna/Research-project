from os import PathLike
from pathlib import Path

import meshio
import numpy as np
from skfem import MeshTri


def load_gmsh_tri(msh_path: str | PathLike[str]) -> MeshTri:
    """Load all 3-node triangle blocks from a Gmsh mesh."""
    mesh = meshio.read(Path(msh_path))
    triangle_blocks = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangle_blocks:
        raise ValueError("No triangle cells in .msh. This loader expects 2D triangles.")

    triangles = np.vstack(triangle_blocks)
    used_points = np.unique(triangles)
    points = mesh.points[used_points, :2]
    point_indices = {old: new for new, old in enumerate(used_points)}
    compact_triangles = np.vectorize(point_indices.__getitem__)(triangles)

    return MeshTri(points.T, compact_triangles.T)
