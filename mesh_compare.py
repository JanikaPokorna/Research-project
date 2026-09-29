import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mesh_utils import load_gmsh_tri


DEFAULT_MESHES = {
    "Square (regular)": "square_regular_mesh.msh",
    "Nested square (2 intervals)": "meshes_diffusion_refinement/square_nested_2.msh",
    "Nested square (4 intervals)": "meshes_diffusion_refinement/square_nested_4.msh",
    "Nested square (8 intervals)": "meshes_diffusion_refinement/square_nested_8.msh",
    "Nested square (16 intervals)": "meshes_diffusion_refinement/square_nested_16.msh",
    "Nested square (32 intervals)": "meshes_diffusion_refinement/square_nested_32.msh",
}


def plot_mesh_comparison(meshes: dict[str, str], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows = (len(meshes) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.5 * columns, 5 * rows), constrained_layout=True
    )

    for axis, (title, mesh_filename) in zip(axes.flat, meshes.items()):
        mesh = load_gmsh_tri(mesh_filename)
        axis.triplot(mesh.p[0], mesh.p[1], mesh.t.T, color="#263238", linewidth=0.45)
        axis.set_title(f"{title}\n{mesh.p.shape[1]} points, {mesh.t.shape[1]} triangles")
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, linewidth=0.3, alpha=0.4)

    for axis in axes.flat[len(meshes):]:
        axis.set_visible(False)

    figure.suptitle("Finite-element mesh comparison", fontsize=16)
    figure.savefig(output_path, dpi=200)
    print(f"Saved mesh comparison to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a comparison of project meshes.")
    parser.add_argument(
        "--output",
        default="mesh_comparison_refinement.png",
        help="Output image path (default: mesh_comparison_refinement.png).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent
    meshes = {
        title: str(project_root / filename)
        for title, filename in DEFAULT_MESHES.items()
    }
    plot_mesh_comparison(meshes, args.output)


if __name__ == "__main__":
    main()
