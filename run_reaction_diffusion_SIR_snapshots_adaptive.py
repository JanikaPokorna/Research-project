import argparse
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from skfem import Basis, FacetBasis, LinearForm, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, mass


from mesh_utils import load_gmsh_tri_with_boundary_groups

DEFAULT_T = 7
DEFAULT_DT = 1e-3
DEFAULT_MIN_DT = 1e-8
DEFAULT_MESH = "hexagon_irregular_complex_mesh.msh"


def initial_conditions(basis: Basis):
    """Example initial condition with two infected Gaussian peaks."""
    x = basis.doflocs[0]
    y = basis.doflocs[1]

    I_left = 0.1 * np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
    I_right = 0.1 * np.exp(-((x - 0.75) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
    I = I_left + I_right
    S = 1.0 - I
    R = np.zeros_like(x)
    return S, I, R


def reaction_terms(S, I, R, nu, beta, mu, gamma, eps=1e-12):
    """Density-based nodewise reactions for the SIR model."""
    N = S + I + R + eps
    incidence = beta * (S * I / N)

    fS = nu * (1.0 - S) - incidence - mu * S
    fI = incidence - (gamma + mu) * I
    fR = gamma * I - mu * R
    return fS, fI, fR


def assemble_boundary_fluxes(mesh, boundary_groups, fluxes, basis):
    boundary_load = np.zeros(basis.N)
    for group_name, facets in boundary_groups.items():
        flux = fluxes[group_name]

        @LinearForm
        def boundary_flux(v, w):
            return flux * v

        facet_basis = FacetBasis(mesh, ElementTriP1(), facets=facets)
        boundary_load += asm(boundary_flux, facet_basis)
    return boundary_load


def sample_snapshot_times(T, n_snapshots):
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
    mesh, boundary_groups = load_gmsh_tri_with_boundary_groups(msh_path)
    basis = Basis(mesh, ElementTriP1())

    # Zero flux through the outer boundary: a closed population.
    boundary_fluxes = {name: 0.0 for name in boundary_groups}
    boundary_load = assemble_boundary_fluxes(mesh, boundary_groups, boundary_fluxes, basis)

    K = asm(laplace, basis)
    M = asm(mass, basis)
    # Row-sum mass lumping reduces small negative undershoots in P1 states.
    M = diags(np.asarray(M.sum(axis=1)).ravel())

    nu = 0.0
    beta = 3.0
    mu = 0.0
    gamma = 0.5

    DS = 1e-3
    DI = 1e-3
    DR = 1e-3

    S, I, R = initial_conditions(basis)
    picard_maxit = 20
    picard_tol = 1e-8
    positivity_tol = 1e-12

    target_times = sample_snapshot_times(T, n_snapshots)
    snapshots = []
    target_index = 0
    if target_times[0] == 0.0:
        snapshots.append((0.0, S.copy(), I.copy(), R.copy()))
        target_index = 1

    current_time = 0.0
    dt_current = dt
    step_number = 0
    while current_time < T:
        next_target = target_times[target_index] if target_index < len(target_times) else T
        proposed_dt = min(dt_current, T - current_time, next_target - current_time)
        if proposed_dt <= 0.0:
            if target_index < len(target_times) and abs(current_time - next_target) <= 1e-12:
                snapshots.append((float(current_time), S.copy(), I.copy(), R.copy()))
                target_index += 1
                continue
            break

        trial_dt = proposed_dt
        rejections = 0
        accepted = False
        failure_reason = "unknown failure"

        while not accepted:
            AS = (M / trial_dt) + DS * K
            AI = (M / trial_dt) + DI * K
            AR = (M / trial_dt) + DR * K

            rhsS_base = (M @ S) / trial_dt
            rhsI_base = (M @ I) / trial_dt
            rhsR_base = (M @ R) / trial_dt
            Sk, Ik, Rk = S.copy(), I.copy(), R.copy()
            converged = False
            state_scale = max(
                1.0,
                float(np.max(np.abs(S))),
                float(np.max(np.abs(I))),
                float(np.max(np.abs(R))),
            )
            allowed_negative = positivity_tol * state_scale

            for iteration in range(picard_maxit):
                fS, fI, fR = reaction_terms(Sk, Ik, Rk, nu, beta, mu, gamma)
                bS = rhsS_base + (M @ fS) + boundary_load
                bI = rhsI_base + (M @ fI) + boundary_load
                bR = rhsR_base + (M @ fR) + boundary_load

                Snew = cast(np.ndarray, spsolve(AS, bS))
                Inew = cast(np.ndarray, spsolve(AI, bI))
                Rnew = cast(np.ndarray, spsolve(AR, bR))
                new_state = (Snew, Inew, Rnew)

                if not all(np.isfinite(field).all() for field in new_state):
                    failure_reason = f"non-finite iterate at Picard iteration {iteration + 1}"
                    break

                minimum_value = min(float(field.min()) for field in new_state)
                if minimum_value < -allowed_negative:
                    failure_reason = (
                        f"negative compartment value {minimum_value:.3e} at "
                        f"Picard iteration {iteration + 1}"
                    )
                    break

                # Remove only tiny negative round-off values within the tolerance.
                Snew, Inew, Rnew = (np.maximum(field, 0.0) for field in new_state)
                err = max(
                    np.linalg.norm(Snew - Sk),
                    np.linalg.norm(Inew - Ik),
                    np.linalg.norm(Rnew - Rk),
                )
                normU = max(
                    np.linalg.norm(Snew),
                    np.linalg.norm(Inew),
                    np.linalg.norm(Rnew),
                    1.0,
                )
                relative_change = err / normU
                Sk, Ik, Rk = Snew, Inew, Rnew

                if relative_change < picard_tol:
                    converged = True
                    break

            if converged:
                S, I, R = Sk, Ik, Rk
                accepted = True
                step_number += 1
                current_time = min(current_time + trial_dt, T)
                if rejections:
                    print(
                        f"Accepted t={current_time:.6f} with dt={trial_dt:.3e} "
                        f"after {rejections} rejected attempt(s); "
                        f"Picard iterations={iteration + 1}."
                    )
                dt_current = min(dt, trial_dt * 1.25)
                continue

            if failure_reason == "unknown failure":
                failure_reason = (
                    f"Picard did not converge in {picard_maxit} iterations "
                    f"(relative change={relative_change:.3e})"
                )

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
            snapshots.append((float(current_time), S.copy(), I.copy(), R.copy()))
            target_index += 1

        if step_number % 1000 == 0:
            print(f"Accepted step {step_number}, t={current_time:.6f}, dt-next={dt_current:.3e}")

    return mesh, snapshots


def plot_snapshots(mesh, snapshots, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not snapshots:
        raise ValueError("No snapshots were generated.")

    triangulation = Triangulation(mesh.p[0], mesh.p[1], triangles=mesh.t.T)
    fields = (
        ("S", "Susceptible density", 1, "viridis"),
        ("I", "Infected density", 2, "magma"),
        ("R", "Recovered density", 3, "cividis"),
    )

    for field_name, field_label, field_index, cmap in fields:
        field_values = [snapshot[field_index] for snapshot in snapshots]
        vmin = min(float(values.min()) for values in field_values)
        vmax = max(float(values.max()) for values in field_values)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        axes = axes.ravel()
        for ax in axes:
            ax.set_axis_off()

        for idx, (time, _, _, _) in enumerate(snapshots[:6]):
            ax = axes[idx]
            ax.set_axis_on()
            contour = ax.tricontourf(
                triangulation,
                field_values[idx],
                levels=30,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_aspect("equal")
            ax.set_title(f"t = {time:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(contour, ax=ax, label=field_name)

        for ax in axes[min(len(snapshots), 6):]:
            ax.set_visible(False)

        if field_name == "I":
            combined_name = "reaction_diffusion_SIR_snapshots_adaptive.png"
        else:
            combined_name = f"reaction_diffusion_SIR_{field_name}_snapshots_adaptive.png"
        combined_path = output_dir / combined_name
        fig.savefig(combined_path, dpi=200)
        plt.close(fig)

        for idx, (time, _, _, _) in enumerate(snapshots):
            fig_single, ax_single = plt.subplots(figsize=(5, 4))
            contour = ax_single.tricontourf(
                triangulation,
                field_values[idx],
                levels=30,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax_single.set_aspect("equal")
            ax_single.set_title(f"{field_label} at t = {time:.3f}")
            fig_single.colorbar(contour, ax=ax_single, label=field_name)
            fig_single.tight_layout()
            if field_name == "I":
                image_name = f"adaptive_snapshot_{idx + 1:02d}.png"
            else:
                image_name = f"adaptive_{field_name}_snapshot_{idx + 1:02d}.png"
            fig_single.savefig(output_dir / image_name, dpi=200)
            plt.close(fig_single)

        print(f"Saved {field_name} snapshot panels to {combined_path}")

    print(f"Saved individual snapshots in {output_dir}")


def main(
    mesh_filename: str | None = None,
    output_dir: str | Path = "figures/reaction_diffusion_SIR_snapshots_adaptive",
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
        description="Run reaction-diffusion SIR with rejected/retried adaptive time steps."
    )
    parser.add_argument("mesh", nargs="?", help="Optional path to a .msh file.")
    parser.add_argument(
        "--output-dir",
        default="figures/reaction_diffusion_SIR_snapshots_adaptive",
        help="Folder for saved snapshots and the combined figure.",
    )
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
