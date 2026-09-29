import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from skfem import Basis, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, mass

from mesh_utils import load_gmsh_tri_with_boundary_groups


DEFAULT_T = 0.1
DEFAULT_DT = 1e-3
DEFAULT_MIN_DT = 1e-8
DEFAULT_MESH = "hexagon_irregular_complex_mesh.msh"


def initial_condition(basis: Basis):
    """Broad smooth initial variation across the domain."""
    x = basis.doflocs[0]
    x_min, x_max = float(x.min()), float(x.max())
    return 0.5 * (1.0 + np.cos(np.pi * (x - x_min) / (x_max - x_min)))


def sample_snapshot_times(T: float, n_snapshots: int):
    if n_snapshots <= 1:
        return [T]
    return [T * idx / (n_snapshots - 1) for idx in range(n_snapshots)]


def run_simulation(
    mesh_filename: str | None = None,
    T: float = DEFAULT_T,
    dt: float = DEFAULT_DT,
    n_snapshots: int = 6,
    min_dt: float = DEFAULT_MIN_DT,
    max_rejections: int = 20,
):
    if T < 0.0:
        raise ValueError("T must be nonnegative.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if min_dt <= 0.0 or min_dt > dt:
        raise ValueError("min_dt must be positive and no greater than dt.")
    if n_snapshots < 1:
        raise ValueError("n_snapshots must be at least 1.")
    if max_rejections < 0:
        raise ValueError("max_rejections must be nonnegative.")

    msh_path = mesh_filename or str(Path(__file__).with_name(DEFAULT_MESH))
    mesh, _ = load_gmsh_tri_with_boundary_groups(msh_path)
    basis = Basis(mesh, ElementTriP1())

    stiffness = asm(laplace, basis)
    mass_matrix = asm(mass, basis)
    lumped_mass = diags(np.asarray(mass_matrix.sum(axis=1)).ravel())

    u = initial_condition(basis)
    initial_mass = float(np.sum(lumped_mass @ u))
    target_times = sample_snapshot_times(T, n_snapshots)
    snapshots = []
    target_index = 0
    if target_times[0] == 0.0:
        snapshots.append((0.0, u.copy()))
        target_index = 1

    current_time = 0.0
    dt_current = dt
    step_number = 0
    while current_time < T:
        next_target = target_times[target_index] if target_index < len(target_times) else T
        proposed_dt = min(dt_current, T - current_time, next_target - current_time)
        if proposed_dt <= 0.0:
            if target_index < len(target_times) and abs(current_time - next_target) <= 1e-12:
                snapshots.append((float(current_time), u.copy()))
                target_index += 1
                continue
            break

        trial_dt = proposed_dt
        rejections = 0
        while True:
            system = (lumped_mass / trial_dt) + stiffness
            rhs = (lumped_mass @ u) / trial_dt
            candidate = np.asarray(spsolve(system, rhs))

            state_scale = max(1.0, float(np.max(np.abs(u))))
            allowed_negative = 1e-12 * state_scale
            failure_reason = None
            if not np.isfinite(candidate).all():
                failure_reason = "non-finite solution"
            elif float(candidate.min()) < -allowed_negative:
                failure_reason = f"negative solution value {candidate.min():.3e}"

            if failure_reason is None:
                u = np.maximum(candidate, 0.0)
                current_time = min(current_time + trial_dt, T)
                step_number += 1
                if rejections:
                    print(
                        f"Accepted t={current_time:.6f} with dt={trial_dt:.3e} "
                        f"after {rejections} rejected attempt(s)."
                    )
                dt_current = min(dt, trial_dt * 1.25)
                break

            rejections += 1
            if rejections > max_rejections:
                raise FloatingPointError(
                    f"Could not accept a step near t={current_time:.6g} after "
                    f"{max_rejections} rejections; last failure: {failure_reason}."
                )

            trial_dt *= 0.5
            if trial_dt < min_dt:
                raise FloatingPointError(
                    f"Required dt fell below min_dt={min_dt:.3e} near "
                    f"t={current_time:.6g}; last failure: {failure_reason}."
                )
            print(
                f"Rejected step near t={current_time:.6f}; retrying with "
                f"dt={trial_dt:.3e} ({failure_reason})."
            )

        while (
            target_index < len(target_times)
            and current_time >= target_times[target_index] - 1e-12
        ):
            snapshots.append((float(current_time), u.copy()))
            target_index += 1

        if step_number % 1000 == 0:
            print(f"Accepted step {step_number}, t={current_time:.6f}, dt-next={dt_current:.3e}")

    final_mass = float(np.sum(lumped_mass @ u))
    print(
        f"Completed {step_number} steps; mass change={final_mass - initial_mass:.3e}; "
        f"u[min,max]=({u.min():.3e},{u.max():.3e})."
    )
    return mesh, snapshots


def plot_snapshots(mesh, snapshots, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not snapshots:
        raise ValueError("No snapshots were generated.")

    triangulation = Triangulation(mesh.p[0], mesh.p[1], triangles=mesh.t.T)
    values = [field for _, field in snapshots]
    vmin = min(float(field.min()) for field in values)
    vmax = max(float(field.max()) for field in values)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()

    for idx, (time, field) in enumerate(snapshots[:6]):
        ax = axes[idx]
        ax.set_axis_on()
        contour = ax.tricontourf(triangulation, field, levels=30, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(f"t = {time:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(contour, ax=ax, label="u")

    for ax in axes[min(len(snapshots), 6):]:
        ax.set_visible(False)

    combined_path = output_dir / "diffusion_snapshots.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)

    for idx, (time, field) in enumerate(snapshots):
        fig_single, ax_single = plt.subplots(figsize=(5, 4))
        contour = ax_single.tricontourf(
            triangulation, field, levels=30, vmin=vmin, vmax=vmax, cmap="viridis"
        )
        ax_single.set_aspect("equal")
        ax_single.set_title(f"u at t = {time:.3f}")
        fig_single.colorbar(contour, ax=ax_single, label="u")
        fig_single.tight_layout()
        fig_single.savefig(output_dir / f"diffusion_snapshot_{idx + 1:02d}.png", dpi=200)
        plt.close(fig_single)

    print(f"Saved diffusion snapshot panels to {combined_path}")


def main(
    mesh_filename: str | None = None,
    output_dir: str | Path = "figures/diffusion_snapshots",
    n_snapshots: int = 6,
    T: float = DEFAULT_T,
    dt: float = DEFAULT_DT,
    min_dt: float = DEFAULT_MIN_DT,
    max_rejections: int = 20,
):
    mesh, snapshots = run_simulation(
        mesh_filename=mesh_filename,
        T=T,
        dt=dt,
        n_snapshots=n_snapshots,
        min_dt=min_dt,
        max_rejections=max_rejections,
    )
    plot_snapshots(mesh, snapshots, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the time-dependent diffusion equation with implicit Euler."
    )
    parser.add_argument("mesh", nargs="?", help="Optional path to a .msh file.")
    parser.add_argument("--output-dir", default="figures/diffusion_snapshots")
    parser.add_argument("--n-snapshots", type=int, default=6)
    parser.add_argument("--t-end", type=float, default=DEFAULT_T)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Initial and maximum time step.")
    parser.add_argument("--min-dt", type=float, default=DEFAULT_MIN_DT)
    parser.add_argument("--max-rejections", type=int, default=20)
    args = parser.parse_args()

    main(
        mesh_filename=args.mesh,
        output_dir=args.output_dir,
        n_snapshots=args.n_snapshots,
        T=args.t_end,
        dt=args.dt,
        min_dt=args.min_dt,
        max_rejections=args.max_rejections,
    )