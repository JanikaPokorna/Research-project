from os import PathLike
from pathlib import Path

import meshio
import numpy as np
from skfem import MeshTri


def load_gmsh_tri_with_boundary_groups(
    msh_path: str | PathLike[str],
) -> tuple[MeshTri, dict[str, np.ndarray]]:
    """Load a triangular Gmsh mesh and map physical curve groups to facets."""
    mesh = meshio.read(Path(msh_path))
    triangle_blocks = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangle_blocks:
        raise ValueError("No triangle cells in .msh. This loader expects 2D triangles.")

    triangles = np.vstack(triangle_blocks)
    used_points = np.unique(triangles)
    points = mesh.points[used_points, :2]
    point_indices = {old: new for new, old in enumerate(used_points)}
    compact_triangles = np.vectorize(point_indices.__getitem__)(triangles)
    mesh_tri = MeshTri(points.T, compact_triangles.T)

    boundary_group_facets: dict[str, list[int]] = {}
    physical_data = mesh.cell_data.get("gmsh:physical", [])
    physical_names = {
        int(values[0]): name
        for name, values in mesh.field_data.items()
        if int(values[1]) == 1
    }
    facets = mesh_tri.facets
    if facets is None:
        raise ValueError("The mesh does not contain facets.")
    mesh_facets = np.sort(facets.T, axis=1)
    facet_indices = {tuple(edge): index for index, edge in enumerate(mesh_facets)}

    for cell_block, tags in zip(mesh.cells, physical_data):
        if cell_block.type != "line":
            continue
        for edge, tag in zip(cell_block.data, tags):
            compact_edge = tuple(sorted(point_indices[int(point)] for point in edge))
            facet_index = facet_indices.get(compact_edge)
            group_name = physical_names.get(int(tag))
            if facet_index is None or group_name is None:
                continue
            boundary_group_facets.setdefault(group_name, []).append(facet_index)

    boundary_groups = {
        name: np.asarray(sorted(set(facets)), dtype=int)
        for name, facets in boundary_group_facets.items()
    }
    return mesh_tri, boundary_groups


def load_gmsh_tri(msh_path: str | PathLike[str]) -> MeshTri:
    """Load all 3-node triangle blocks from a Gmsh mesh."""
    mesh, _ = load_gmsh_tri_with_boundary_groups(msh_path)
    return mesh
