import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np
from scipy.sparse import diags

from skfem import Basis, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import mass

from run_diffusion_transient import DEFAULT_T, initial_condition, run_simulation


PROJECT_DIR = Path(__file__).resolve().parent
REFINED_MESHES = (
    "square_nested_2.msh",
    "square_nested_4.msh",
    "square_nested_8.msh",
    "square_nested_16.msh",
    "square_nested_32.msh",
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "figures" / "diffusion_transient_verification"
DEFAULT_DT = 1e-4


def exact_solution(x, time: float):
    return 0.5 + 0.5 * np.exp(-(np.pi**2) * time) * np.cos(np.pi * x)


def maximum_edge_length(mesh) -> float:
    edges = ((0, 1), (1, 2), (2, 0))
    return max(
        float(
            np.linalg.norm(
                mesh.p[:, mesh.t[first]] - mesh.p[:, mesh.t[second]], axis=0
            ).max()
        )
        for first, second in edges
    )


def measure_solution(mesh, final_values: np.ndarray, time: float):
    basis = Basis(mesh, ElementTriP1(), intorder=5)
    numerical = basis.interpolate(final_values)
    coordinates = basis.global_coordinates()
    exact = exact_solution(coordinates[0], time)
    l2_error = float(np.sqrt(np.sum((numerical - exact) ** 2 * basis.dx)))

    lumped_mass = diags(np.asarray(asm(mass, basis).sum(axis=1)).ravel())
    initial_values = initial_condition(basis)
    initial_mass = float(np.sum(lumped_mass @ initial_values))
    final_mass = float(np.sum(lumped_mass @ final_values))
    relative_mass_drift = abs(final_mass - initial_mass) / initial_mass

    return l2_error, relative_mass_drift


def run_verification(T: float, dt: float, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for mesh_name in REFINED_MESHES:
        mesh_path = PROJECT_DIR / "meshes_diffusion_refinement" / mesh_name
        mesh, snapshots = run_simulation(
            mesh_filename=str(mesh_path),
            T=T,
            dt=dt,
            n_snapshots=2,
        )
        final_time, final_values = snapshots[-1]
        l2_error, mass_drift = measure_solution(mesh, final_values, final_time)
        results.append(
            {
                "mesh": mesh_name,
                "vertices": int(mesh.p.shape[1]),
                "triangles": int(mesh.t.shape[1]),
                "h_max": maximum_edge_length(mesh),
                "time": final_time,
                "l2_error": l2_error,
                "relative_mass_drift": mass_drift,
                "mesh_object": mesh,
                "solution": final_values,
            }
        )

    results.sort(key=lambda result: result["h_max"], reverse=True)
    for previous, current in zip(results, results[1:]):
        current["observed_order"] = np.log(previous["l2_error"] / current["l2_error"]) / np.log(
            previous["h_max"] / current["h_max"]
        )
    if results:
        results[0]["observed_order"] = float("nan")

    csv_path = output_dir / "diffusion_mesh_verification.csv"
    csv_columns = (
        "mesh",
        "vertices",
        "triangles",
        "h_max",
        "time",
        "l2_error",
        "observed_order",
        "relative_mass_drift",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows({key: result[key] for key in csv_columns} for result in results)

    fig, (profile_axis, error_axis) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x_line = np.linspace(0.0, 1.0, 301)
    profile_axis.plot(
        x_line,
        exact_solution(x_line, T),
        color="black",
        linewidth=2.5,
        label="Analytical solution",
    )
    for result in results:
        mesh = result["mesh_object"]
        triangulation = Triangulation(mesh.p[0], mesh.p[1], triangles=mesh.t.T)
        profile = LinearTriInterpolator(triangulation, result["solution"])
        profile_values = profile(x_line, np.full_like(x_line, 0.5))
        profile_axis.plot(
            x_line,
            profile_values,
            linewidth=1.2,
            label=f"{result['vertices']} vertices",
        )
    profile_axis.set_title(f"Solution along y = 0.5 at t = {T:g}")
    profile_axis.set_xlabel("x")
    profile_axis.set_ylabel("u(x, 0.5, t)")
    profile_axis.grid(True, alpha=0.25)
    profile_axis.legend(fontsize=8)

    mesh_sizes = np.array([result["h_max"] for result in results])
    errors = np.array([result["l2_error"] for result in results])
    error_axis.loglog(mesh_sizes, errors, "o-", label="Measured $L^2$ error")
    reference = errors[0] * (mesh_sizes / mesh_sizes[0]) ** 2
    error_axis.loglog(mesh_sizes, reference, "--", label="$O(h^2)$ reference")
    for result in results[1:]:
        error_axis.annotate(
            f"p={result['observed_order']:.2f}",
            (result["h_max"], result["l2_error"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    error_axis.set_title("Spatial mesh convergence")
    error_axis.set_xlabel("Maximum element edge length h")
    error_axis.set_ylabel("$L^2$ error at final time")
    error_axis.grid(True, which="both", alpha=0.25)
    error_axis.legend()

    fig.suptitle(f"Transient diffusion verification (dt={dt:g}, T={T:g})")
    figure_path = output_dir / "diffusion_mesh_verification.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    for result in results:
        print(
            f"{result['mesh']}: vertices={result['vertices']}, "
            f"h_max={result['h_max']:.4g}, L2 error={result['l2_error']:.4e}, "
            f"order={result['observed_order']:.3f}, "
            f"relative mass drift={result['relative_mass_drift']:.3e}"
        )
    print(f"Saved verification figure to {figure_path}")
    print(f"Saved verification data to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify transient diffusion on a sequence of refined square meshes."
    )
    parser.add_argument("--t-end", type=float, default=DEFAULT_T)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run_verification(T=args.t_end, dt=args.dt, output_dir=args.output_dir)