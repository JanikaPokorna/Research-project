import argparse
from pathlib import Path

import meshio
from matplotlib.tri import Triangulation
import numpy as np
from skfem import MeshTri

from mesh_utils import load_gmsh_tri_with_boundary_groups


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MESH_DIR = PROJECT_DIR / "meshes_diffusion_refinement"
REFINEMENT_LEVELS = (
    ("square_nested_2.msh", 2),
    ("square_nested_4.msh", 4),
    ("square_nested_8.msh", 8),
    ("square_nested_16.msh", 16),
    ("square_nested_32.msh", 32),
)
BOUNDARY_GROUPS = (
    ("Gamma_bottom", 2, 1),
    ("Gamma_right", 3, 2),
    ("Gamma_top", 4, 3),
    ("Gamma_left", 5, 4),
)


def element_max_edge_lengths(mesh: MeshTri) -> np.ndarray:
    return np.maximum.reduce(
        [
            np.linalg.norm(mesh.p[:, mesh.t[first]] - mesh.p[:, mesh.t[second]], axis=0)
            for first, second in ((0, 1), (1, 2), (2, 0))
        ]
    )


def validate_nested(coarse: MeshTri, fine: MeshTri) -> None:
    coarse_vertices = {tuple(point) for point in np.round(coarse.p.T, 12)}
    fine_vertices = {tuple(point) for point in np.round(fine.p.T, 12)}
    if not coarse_vertices <= fine_vertices:
        raise ValueError("A refined mesh is missing vertices from its parent mesh.")

    coarse_triangulation = Triangulation(coarse.p[0], coarse.p[1], coarse.t.T)
    fine_centroids = fine.p[:, fine.t].mean(axis=1)
    parents = np.asarray(
        coarse_triangulation.get_trifinder()(fine_centroids[0], fine_centroids[1]),
        dtype=np.intp,
    )
    if np.any(parents < 0):
        raise ValueError("A refined triangle lies outside its parent mesh.")

    def areas(mesh: MeshTri) -> np.ndarray:
        vertices = mesh.p[:, mesh.t]
        first = vertices[:, 1] - vertices[:, 0]
        second = vertices[:, 2] - vertices[:, 0]
        return 0.5 * np.abs(first[0] * second[1] - first[1] * second[0])

    coarse_areas = areas(coarse)
    fine_areas_by_parent = np.bincount(
        parents, weights=areas(fine), minlength=coarse.t.shape[1]
    )
    if not np.allclose(fine_areas_by_parent, coarse_areas, rtol=1e-9, atol=1e-12):
        raise ValueError("Refined triangles do not exactly partition their parent triangles.")


def validate_boundary_groups(mesh: MeshTri, groups: dict[str, np.ndarray]) -> None:
    expected = {name for name, _, _ in BOUNDARY_GROUPS}
    if set(groups) != expected:
        raise ValueError(f"Expected boundary groups {expected}, received {set(groups)}.")

    boundary = set(map(int, mesh.boundary_facets()))
    grouped = set()
    for facets in groups.values():
        group_facets = set(map(int, facets))
        if not group_facets <= boundary:
            raise ValueError("A physical boundary group includes an interior facet.")
        grouped.update(group_facets)
    if grouped != boundary:
        raise ValueError("Physical boundary groups do not cover the entire exterior.")


def boundary_edges_by_side(mesh: MeshTri) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in BOUNDARY_GROUPS}
    boundary_facets = mesh.boundary_facets()
    mesh_facets = mesh.facets
    if mesh_facets is None:
        raise ValueError("Mesh facets are unavailable.")
    facets = mesh_facets[:, boundary_facets].T

    for edge in facets:
        endpoints = mesh.p[:, edge]
        if np.allclose(endpoints[1], 0.0, atol=1e-10):
            side = "Gamma_bottom"
        elif np.allclose(endpoints[0], 1.0, atol=1e-10):
            side = "Gamma_right"
        elif np.allclose(endpoints[1], 1.0, atol=1e-10):
            side = "Gamma_top"
        elif np.allclose(endpoints[0], 0.0, atol=1e-10):
            side = "Gamma_left"
        else:
            raise ValueError(f"Boundary edge is not on the unit-square boundary: {endpoints}")
        result[side].append(edge)

    return {
        name: np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        for name, edges in result.items()
    }


def write_gmsh_mesh(mesh: MeshTri, output_path: Path) -> None:
    side_edges = boundary_edges_by_side(mesh)
    cells = []
    physical_data = []
    geometrical_data = []
    field_data = {"Omega": np.array([1, 2], dtype=np.int32)}

    for name, physical_tag, geometrical_tag in BOUNDARY_GROUPS:
        edges = side_edges[name]
        cells.append(("line", edges))
        physical_data.append(np.full(len(edges), physical_tag, dtype=np.int32))
        geometrical_data.append(np.full(len(edges), geometrical_tag, dtype=np.int32))
        field_data[name] = np.array([physical_tag, 1], dtype=np.int32)

    triangles = mesh.t.T.astype(np.int64, copy=False)
    cells.append(("triangle", triangles))
    physical_data.append(np.ones(len(triangles), dtype=np.int32))
    geometrical_data.append(np.ones(len(triangles), dtype=np.int32))

    points = np.column_stack((mesh.p.T, np.zeros(mesh.p.shape[1])))
    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            "gmsh:physical": physical_data,
            "gmsh:geometrical": geometrical_data,
        },
        field_data=field_data,
    )
    meshio.write(output_path, output_mesh, file_format="gmsh22")


def generate(mesh_dir: Path) -> None:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    first_intervals = REFINEMENT_LEVELS[0][1]
    coordinates = np.linspace(0.0, 1.0, first_intervals + 1)
    mesh = MeshTri.init_tensor(coordinates, coordinates)
    previous_intervals = first_intervals

    for index, (filename, intervals) in enumerate(REFINEMENT_LEVELS):
        if index > 0:
            if intervals != 2 * previous_intervals:
                raise ValueError("Each nested level must double intervals per side.")
            previous_mesh = mesh
            mesh = mesh.refined(1)
            validate_nested(previous_mesh, mesh)

        output_path = mesh_dir / filename
        write_gmsh_mesh(mesh, output_path)
        loaded_mesh, boundary_groups = load_gmsh_tri_with_boundary_groups(output_path)
        validate_boundary_groups(loaded_mesh, boundary_groups)

        print(
            f"{filename}: {mesh.p.shape[1]} vertices, {mesh.t.shape[1]} triangles, "
            f"h_max={element_max_edge_lengths(mesh).max():.6g}"
        )
        previous_intervals = intervals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a nested adaptive refinement sequence for diffusion validation."
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="Directory containing the coarse mesh and receiving refined meshes.",
    )
    args = parser.parse_args()
    generate(args.mesh_dir)


if __name__ == "__main__":
    main()