import argparse
import csv
from pathlib import Path

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from mesh_utils import load_gmsh_tri_with_boundary_groups
from run_reaction_diffusion_SIR_snapshots_adaptive import (
    DEFAULT_DT,
    DEFAULT_MIN_DT,
    DEFAULT_T,
    run_simulation,
)


SOURCE_GEO = "complex_mesh.geo"
DEFAULT_MESH_DIR = "meshes_mesh_size_comparison"
DEFAULT_OUTPUT_DIR = "figures/mesh_size_comparison"
# Adjust these global Gmsh size limits to create other resolutions.
MESH_RESOLUTIONS = (
    ("Coarse", 0.10, 0.35, "complex_coarse.msh"),
    ("Medium", 0.07, 0.25, "complex_medium.msh"),
    ("Fine", 0.035, 0.14, "complex_fine.msh"),
)


FIELDS = (
    ("S", "Susceptible density", 1, "viridis"),
    ("I", "Infected density", 2, "magma"),
    ("R", "Recovered density", 3, "cividis"),
)


def generate_comparison_meshes(mesh_dir: str | Path):
    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    geo_path = Path(__file__).with_name(SOURCE_GEO)
    if not geo_path.exists():
        raise FileNotFoundError(f"Mesh geometry not found: {geo_path}")

    generated = []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        for label, size_min, size_max, filename in MESH_RESOLUTIONS:
            gmsh.clear()
            gmsh.open(str(geo_path))
            gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            gmsh.model.mesh.generate(2)

            mesh_path = mesh_dir / filename
            gmsh.write(str(mesh_path))
            mesh, _ = load_gmsh_tri_with_boundary_groups(mesh_path)
            generated.append((label, size_min, size_max, mesh_path, mesh.p.shape[1], mesh.t.shape[1]))
    finally:
        gmsh.finalize()

    return generated


def run_comparison(
    T: float,
    dt: float,
    min_dt: float,
    mesh_dir: str | Path,
    max_rejections: int,
):
    generated = generate_comparison_meshes(mesh_dir)
    results = []

    for label, size_min, size_max, mesh_path, point_count, triangle_count in generated:
        print(
            f"Running {label.lower()} mesh: h_min={size_min:g}, h_max={size_max:g}, "
            f"nodes={point_count}, triangles={triangle_count}"
        )
        mesh, snapshots = run_simulation(
            mesh_filename=str(mesh_path),
            T=T,
            dt=dt,
            n_snapshots=3,
            min_dt=min_dt,
            max_rejections=max_rejections,
        )
        if len(snapshots) != 3:
            raise RuntimeError(
                f"Expected 3 snapshots for {label} mesh, received {len(snapshots)}."
            )
        results.append(
            {
                "label": label,
                "h_min": size_min,
                "h_max": size_max,
                "path": mesh_path,
                "mesh": mesh,
                "snapshots": snapshots,
                "points": point_count,
                "triangles": triangle_count,
            }
        )

    return results


def plot_comparison(results, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(results) != 3:
        raise ValueError(f"Expected exactly three mesh results, received {len(results)}.")

    # Layout: one row per mesh; columns are S at three times, then I, then R.
    figure, axes = plt.subplots(3, 9, figsize=(25, 10), squeeze=False)
    field_limits = {}
    for field_name, field_label, field_index, cmap in FIELDS:
        values = [
            snapshot[field_index]
            for result in results
            for snapshot in result["snapshots"]
        ]
        vmin = min(float(array.min()) for array in values)
        vmax = max(float(array.max()) for array in values)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        field_limits[field_name] = (vmin, vmax)

    for row, result in enumerate(results):
        mesh = result["mesh"]
        triangulation = Triangulation(mesh.p[0], mesh.p[1], triangles=mesh.t.T)
        snapshots = result["snapshots"]

        for field_idx, (field_name, field_label, state_index, cmap) in enumerate(FIELDS):
            vmin, vmax = field_limits[field_name]
            for time_idx, (time, _, _, _) in enumerate(snapshots):
                column = field_idx * 3 + time_idx
                ax = axes[row, column]
                state = snapshots[time_idx][state_index]
                ax.tricontourf(
                    triangulation,
                    state,
                    levels=30,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax.set_aspect("equal")
                ax.set_title(f"{field_name}, t={time:.3f}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_xlabel(field_label, fontsize=9)
                    ax.xaxis.set_label_position("top")

    for row, result in enumerate(results):
        figure.text(
            0.015,
            0.82 - row * 0.30,
            f"{result['label']}\nh_min={result['h_min']:g}\nh_max={result['h_max']:g}\n"
            f"{result['points']} nodes\n{result['triangles']} triangles",
            ha="left",
            va="center",
            fontsize=9,
        )

    figure.suptitle(
        "Reaction-diffusion SIR: mesh-resolution comparison\n"
        "Rows: coarse to fine meshes | Column groups: S, I, R; three times per group",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.08, wspace=0.08, hspace=0.18)

    for field_idx, (field_name, field_label, _, cmap) in enumerate(FIELDS):
        vmin, vmax = field_limits[field_name]
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            ax=axes[:, field_idx * 3:(field_idx + 1) * 3].ravel().tolist(),
            shrink=0.82,
            pad=0.015,
        )
        colorbar.set_label(field_name)

    output_path = output_dir / "sir_mesh_resolution_comparison_3x9.png"
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    summary_path = output_dir / "mesh_resolution_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(("label", "h_min", "h_max", "mesh_file", "nodes", "triangles"))
        for result in results:
            writer.writerow((
                result["label"],
                result["h_min"],
                result["h_max"],
                str(result["path"]),
                result["points"],
                result["triangles"],
            ))

    print(f"Saved comparison figure: {output_path}")
    print(f"Saved mesh summary: {summary_path}")


def main(
    T: float = DEFAULT_T,
    dt: float = DEFAULT_DT,
    min_dt: float = DEFAULT_MIN_DT,
    mesh_dir: str | Path = DEFAULT_MESH_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_rejections: int = 20,
):
    if T <= 0.0:
        raise ValueError("T must be positive for a three-time comparison.")

    results = run_comparison(T, dt, min_dt, mesh_dir, max_rejections)
    plot_comparison(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate three mesh resolutions and compare adaptive SIR snapshots."
    )
    parser.add_argument("--t-end", type=float, default=DEFAULT_T)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Initial and maximum time step.")
    parser.add_argument("--min-dt", type=float, default=DEFAULT_MIN_DT)
    parser.add_argument("--mesh-dir", default=DEFAULT_MESH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rejections", type=int, default=20)
    args = parser.parse_args()

    main(
        T=args.t_end,
        dt=args.dt,
        min_dt=args.min_dt,
        mesh_dir=args.mesh_dir,
        output_dir=args.output_dir,
        max_rejections=args.max_rejections,
    )
