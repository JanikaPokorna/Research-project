import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import meshio
import numpy as np


DEFAULT_MESH = "hexagon_irregular_complex_mesh.msh"
DEFAULT_OUTPUT = "figures/mesh_boundary_facets.png"


def classify_facets(mesh_path: str | Path) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    mesh = meshio.read(mesh_path)
    geometrical_data = mesh.cell_data.get("gmsh:geometrical")
    if geometrical_data is None:
        raise ValueError("The mesh needs Gmsh geometrical tags to identify district interfaces.")

    edge_surfaces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cell_block, surface_tags in zip(mesh.cells, geometrical_data):
        if cell_block.type != "triangle":
            continue
        for triangle, surface_tag in zip(cell_block.data, surface_tags):
            vertices = [int(vertex) for vertex in triangle]
            for first, second in (
                (vertices[0], vertices[1]),
                (vertices[1], vertices[2]),
                (vertices[2], vertices[0]),
            ):
                edge_surfaces[tuple(sorted((first, second)))].append(int(surface_tag))

    exterior = []
    district_interfaces = []
    for edge, surfaces in edge_surfaces.items():
        if len(surfaces) == 1:
            exterior.append(edge)
        elif len(surfaces) == 2 and surfaces[0] != surfaces[1]:
            district_interfaces.append(edge)
        elif len(surfaces) > 2:
            raise ValueError(f"Non-manifold edge {edge} belongs to {len(surfaces)} triangles.")

    if not exterior:
        raise ValueError("No exterior facets were found in the triangle mesh.")
    return mesh.points[:, :2], exterior, district_interfaces


def plot_mesh_boundaries(mesh_path: str | Path, output_path: str | Path) -> None:
    points, exterior, district_interfaces = classify_facets(mesh_path)
    mesh = meshio.read(mesh_path)
    triangle_blocks = [block.data for block in mesh.cells if block.type == "triangle"]
    triangles = np.vstack(triangle_blocks)
    edges = set(exterior) | set(district_interfaces)

    all_triangle_edges = set()
    for triangle in triangles:
        vertices = [int(vertex) for vertex in triangle]
        all_triangle_edges.update(
            tuple(sorted(edge))
            for edge in (
                (vertices[0], vertices[1]),
                (vertices[1], vertices[2]),
                (vertices[2], vertices[0]),
            )
        )
    ordinary_edges = sorted(all_triangle_edges - edges)

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    axis.add_collection(
        LineCollection(points[np.asarray(ordinary_edges)], colors="#c7d0d5", linewidths=0.55)
    )
    axis.add_collection(
        LineCollection(points[np.asarray(district_interfaces)], colors="#f04e37", linewidths=2.4)
    )
    axis.add_collection(
        LineCollection(points[np.asarray(exterior)], colors="#00a6c7", linewidths=3.2)
    )
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Exterior facets and district interfaces")
    axis.grid(True, color="#dce2e5", linewidth=0.5, alpha=0.7)
    axis.legend(
        handles=[
            Patch(facecolor="#00a6c7", label=f"Exterior facets ({len(exterior)})"),
            Patch(facecolor="#f04e37", label=f"District interfaces ({len(district_interfaces)})"),
            Patch(facecolor="#c7d0d5", label="Within-district triangle edges"),
        ],
        loc="upper right",
        frameon=True,
    )
    axis.text(
        0.02,
        0.02,
        "Each exterior facet is one boundary mesh edge; several facets make up each polygon side.",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 4},
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}: {len(exterior)} exterior facets, {len(district_interfaces)} district-interface facets.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exterior mesh facets and district interfaces.")
    parser.add_argument("--mesh", default=DEFAULT_MESH, help="Input Gmsh mesh path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    args = parser.parse_args()
    plot_mesh_boundaries(args.mesh, args.output)


if __name__ == "__main__":
    main()