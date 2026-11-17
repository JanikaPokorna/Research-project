import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from run_diffusion_mesh_verification import (
    DEFAULT_DT,
    REFINED_MESHES,
    exact_solution,
    time_step_for_mesh,
)
from run_diffusion_transient import DEFAULT_T, run_simulation


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "figures"
    / "diffusion_transient_verification"
    / "analytical_fem_mesh_comparison.png"
)
REFERENCE_RESOLUTION = 250


def collect_mesh_solutions(T: float, dt: float):
    results = []
    for mesh_name in REFINED_MESHES:
        mesh_path = PROJECT_DIR / "meshes_diffusion_refinement" / mesh_name
        mesh_dt = time_step_for_mesh(mesh_name, dt)
        mesh, snapshots = run_simulation(
            mesh_filename=str(mesh_path),
            T=T,
            dt=mesh_dt,
            n_snapshots=2,
        )
        final_time, values = snapshots[-1]
        results.append((mesh_name, mesh, final_time, values))
    return results


def plot_comparison(results, T: float, output_path: Path) -> None:
    coordinates = np.linspace(0.0, 1.0, REFERENCE_RESOLUTION)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    analytical = exact_solution(x_grid, T)

    mesh_errors = []
    for mesh_name, mesh, final_time, values in results:
        if not np.isclose(final_time, T):
            raise RuntimeError(
                f"{mesh_name} ended at t={final_time:g}, expected final time T={T:g}."
            )
        exact_at_vertices = exact_solution(mesh.p[0], T)
        mesh_errors.append(np.abs(values - exact_at_vertices))

    max_errors = np.asarray([float(error.max()) for error in mesh_errors])
    error_levels = np.linspace(0.0, 1.0, 21)
    solution_min = float(analytical.min())
    solution_max = float(analytical.max())
    solution_levels = np.linspace(solution_min, solution_max, 31)

    columns = 4
    rows = (len(results) + 2 + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(19, 4.5 * rows), constrained_layout=True
    )
    axes = axes.ravel()
    analytical_plot = axes[0].contourf(
        x_grid,
        y_grid,
        analytical,
        levels=solution_levels,
        vmin=solution_min,
        vmax=solution_max,
        cmap="viridis",
    )
    axes[0].set_title(f"Analytical solution, t={T:g}")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    error_plot = None
    for axis, (mesh_name, mesh, _, _), error, max_error in zip(
        axes[1:1 + len(results)], results, mesh_errors, max_errors
    ):
        triangulation = Triangulation(mesh.p[0], mesh.p[1], triangles=mesh.t.T)
        normalized_error = error / max_error if max_error > 0.0 else error
        error_plot = axis.tricontourf(
            triangulation,
            normalized_error,
            levels=error_levels,
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        axis.set_title(
            f"{mesh_name.removesuffix('.msh')}\n"
            f"{mesh.t.shape[1]} triangles, max error={max_error:.2e}",
            fontsize=10,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    convergence_axis = axes[1 + len(results)]
    triangle_counts = [mesh.t.shape[1] for _, mesh, _, _ in results]
    convergence_axis.loglog(triangle_counts, max_errors, "o-", color="#B33A3A")
    convergence_axis.set_xticks(triangle_counts, [str(count) for count in triangle_counts])
    convergence_axis.set_title("Maximum error vs. mesh size")
    convergence_axis.set_xlabel("Triangles")
    convergence_axis.set_ylabel("Maximum nodal error")
    convergence_axis.grid(True, which="both", alpha=0.3)
    for axis in axes[2 + len(results):]:
        axis.set_visible(False)

    figure.colorbar(
        analytical_plot,
        ax=axes[0],
        label=f"u(x, y, {T:g})",
        shrink=0.8,
    )
    if error_plot is not None:
        figure.colorbar(
            error_plot,
            ax=axes[1:1 + len(results)].tolist(),
            label=r"$|u_h-u|$ / maximum error for this mesh",
            shrink=0.85,
        )
    figure.suptitle("Analytical solution and FEM errors across mesh refinements", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    print(f"Saved analytical/FEM comparison to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the analytical diffusion solution beside FEM results for every nested mesh."
    )
    parser.add_argument("--t-end", type=float, default=DEFAULT_T)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.t_end < 0.0:
        parser.error("--t-end must be nonnegative.")
    if args.dt <= 0.0:
        parser.error("--dt must be positive.")

    results = collect_mesh_solutions(args.t_end, args.dt)
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    plot_comparison(results, args.t_end, output_path)


if __name__ == "__main__":
    main()