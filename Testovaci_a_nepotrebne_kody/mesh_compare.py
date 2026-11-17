import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from mesh_utils import load_gmsh_tri


DEFAULT_OUTPUT = PROJECT_DIR / "figures" / "mesh_comparison_refinement.png"
DEFAULT_SHAPE_OUTPUT = PROJECT_DIR / "figures" / "mesh_comparison.png"

SQUARE_MESHES = {
    "Nested square (16 intervals)": "meshes_diffusion_refinement/square_nested_16.msh",
    "Nested square (32 intervals)": "meshes_diffusion_refinement/square_nested_32.msh",
    "Nested square (64 intervals)": "meshes_diffusion_refinement/square_nested_64.msh",
    "Nested square (128 intervals)": "meshes_diffusion_refinement/square_nested_128.msh",
}

POLYGON_MESHES = {
    "Hexagon": "Testovaci_a_nepotrebne_kody/hexagon_mesh.msh",
    "Star-split hexagon": "Testovaci_a_nepotrebne_kody/hexagon_starsplit_mesh.msh",
    "Complex hexagon": "Testovaci_a_nepotrebne_kody/complex_hexagon_mesh.msh",
    "Irregular complex hexagon": "hexagon_irregular_complex_mesh.msh",
    "Very refined irregular hexagon": "hexagon_irregular_complex_very_refined.msh",
    "Heptagon": "Testovaci_a_nepotrebne_kody/heptagon_mesh.msh",
    "Irregular heptagon": "Testovaci_a_nepotrebne_kody/heptagon_irregular_mesh.msh",
}


def plot_mesh_comparison(meshes: dict[str, str], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 2
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
        default=str(DEFAULT_OUTPUT),
        help=f"Output image path for square refinements (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--shape-output",
        default=str(DEFAULT_SHAPE_OUTPUT),
        help=f"Output image path for polygon meshes (default: {DEFAULT_SHAPE_OUTPUT}).",
    )
    args = parser.parse_args()

    square_meshes = {
        title: str(PROJECT_DIR / filename)
        for title, filename in SQUARE_MESHES.items()
    }
    polygon_meshes = {
        title: str(PROJECT_DIR / filename)
        for title, filename in POLYGON_MESHES.items()
    }
    plot_mesh_comparison(square_meshes, args.output)
    plot_mesh_comparison(polygon_meshes, args.shape_output)


if __name__ == "__main__":
    main()
