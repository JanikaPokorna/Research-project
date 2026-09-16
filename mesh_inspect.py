import argparse
from collections import Counter
from pathlib import Path

import meshio
import numpy as np


def boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    edge_counts = Counter(map(tuple, edges))
    return [edge for edge, count in edge_counts.items() if count == 1]


def inspect_mesh(mesh_filename: str) -> None:
    mesh_path = Path(mesh_filename)
    mesh = meshio.read(mesh_path)
    triangle_blocks = [cell_block.data for cell_block in mesh.cells if cell_block.type == "triangle"]
    triangle_count = sum(len(block) for block in triangle_blocks)

    print(f"Mesh: {mesh_path}")
    print(f"Points: {len(mesh.points)}")
    print(f"Cell blocks: {len(mesh.cells)}")
    print("Cells:")
    for cell_block in mesh.cells:
        print(f"  {cell_block.type}: {len(cell_block.data)}")

    print(f"Triangle blocks: {len(triangle_blocks)}")
    print(f"Triangles: {triangle_count}")

    print("Physical groups:")
    if mesh.field_data:
        for name, values in sorted(mesh.field_data.items()):
            tag, dimension = values
            entity_type = {1: "curve", 2: "surface"}.get(int(dimension), f"dim {dimension}")
            print(f"  {name}: {entity_type}, tag {int(tag)}")
    else:
        print("  none")

    if triangle_blocks:
        triangles = np.vstack(triangle_blocks)
        edges = boundary_edges(triangles)
        print(f"Boundary edges: {len(edges)}")
    else:
        print("Boundary edges: unavailable (no triangles)")

    print("Physical cell data:")
    physical_data = mesh.cell_data.get("gmsh:physical")
    if physical_data is None:
        print("  none")
    else:
        for cell_block, tags in zip(mesh.cells, physical_data):
            tag_counts = ", ".join(
                f"{int(tag)}: {count}" for tag, count in sorted(Counter(tags).items())
            )
            print(f"  {cell_block.type}: {tag_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Gmsh mesh.")
    parser.add_argument("mesh", help="Path to a .msh file.")
    args = parser.parse_args()
    inspect_mesh(args.mesh)


if __name__ == "__main__":
    main()
