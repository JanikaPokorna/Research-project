import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
from typing import cast

from skfem import Basis, FacetBasis, LinearForm, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, mass

from mesh_utils import load_gmsh_tri_with_boundary_groups

DEFAULT_T = 3.5
DEFAULT_MESH = "hexagon_irregular_complex_mesh.msh"


def initial_conditions(basis: Basis):
    """Example ICs: mostly S, small infected Gaussian bump, R=0."""
    x = basis.doflocs[0]
    y = basis.doflocs[1]

    # I0 = 0.1 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2)) 
    # S0 = 1.0 - I0
    # R0 = np.zeros_like(x)

    # two infected peaks on opposite sides, offset from the boundary
    I_left = 0.1 * np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
    I_right = 0.1 * np.exp(-((x - 0.75) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
    I0 = I_left + I_right
    S0 = 1.0 - I0
    R0 = np.zeros_like(x)

    return S0, I0, R0


def reaction_terms(S, I, R, nu, beta, mu, gamma, eps=1e-12):
    """Density-based nodewise reactions for a normalized SIR model."""
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


# def normalize_initial_population(S, I, R, M):
#     """Scale all three densities so their integral over the domain is 1."""
#     ones = np.ones_like(S)
#     total = float(ones @ (M @ (S + I + R)))

#     if total <= 0.0:
#         raise ValueError("The initial total population must be positive.")

#     return S / total, I / total, R / total

# def total_population(M, S, I, R):
#     """FEM integral of S + I + R over the whole domain."""
#     ones = np.ones_like(S)
#     return float(ones @ (M @ (S + I + R)))


def sample_snapshot_times(T, n_snapshots):
    if n_snapshots <= 1:
        return [T]
    return [T * idx / (n_snapshots - 1) for idx in range(n_snapshots)]


def run_simulation(
    mesh_filename: str | None = None,
    T: float = DEFAULT_T, 
    dt: float = 1e-3, 
    n_snapshots: int = 6
    ):

    if T < 0.0:
        raise ValueError("T must be nonnegative.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if n_snapshots < 1:
        raise ValueError("n_snapshots must be at least 1.")

    msh_path = mesh_filename or str(Path(__file__).with_name(DEFAULT_MESH))
    mesh, boundary_groups = load_gmsh_tri_with_boundary_groups(msh_path)
    basis = Basis(mesh, ElementTriP1())

    # Zero flux through the outer boundary: a closed population.
    boundary_fluxes = {name: 0.0 for name in boundary_groups}
    boundary_load = assemble_boundary_fluxes(mesh, boundary_groups, boundary_fluxes, basis)

    # Optional inter-region fluxes (disabled):
    # This solver uses one continuous FEM field over the whole conforming mesh,
    # so fluxes across shared internal facets are already coupled and conserved.
    # Do not add those facets to boundary_load as if they were outer boundaries.
    # The current mesh only tags its exterior curves (Outer_1 through Outer_6);
    # internal region interfaces would first need to be tagged in the .geo file.
    # For separate region-by-region solves with independent unknowns, apply each
    # interface transfer as equal-and-opposite outward flux loads, for example:
    #
    # outer_fluxes = {name: 0.0 for name in outer_boundary_groups}
    # q_A_to_B = transfer_rate * (u_A - u_B)
    # fluxes_region_A = {"Interface_A_B": -q_A_to_B}
    # fluxes_region_B = {"Interface_A_B": +q_A_to_B}
    # load_A = assemble_boundary_fluxes(mesh_A, {"Interface_A_B": facets_A},
    #                                   fluxes_region_A, basis_A)
    # load_B = assemble_boundary_fluxes(mesh_B, {"Interface_A_B": facets_B},
    #                                   fluxes_region_B, basis_B)
    # Set every exterior boundary flux to zero in each region's flux dictionary.

    K = asm(laplace, basis)
    M = asm(mass, basis)

    nu = 0.0
    beta = 3.0
    mu = 0.0
    gamma = 0.5

    DS = 1e-3
    DI = 1e-3
    DR = 1e-3

    AS = (M / dt) + DS * K
    AI = (M / dt) + DI * K
    AR = (M / dt) + DR * K

    S, I, R = initial_conditions(basis)

    picard_maxit = 20
    picard_tol = 1e-8
    target_times = sample_snapshot_times(T, n_snapshots)
    targets = iter(target_times)
    next_time = next(targets, None)
    snapshots = []

    # Store the actual initial state if t=0 is a requested snapshot.
    if target_times[0] == 0.0:
        snapshots.append((0.0, S.copy(), I.copy(), R.copy()))
        target_index = 1


    nsteps = int(np.ceil(T / dt))
    print("Mesh area:", float(np.ones(basis.N) @ (M @ np.ones(basis.N))))
    print("Initial ranges:", (S.min(), S.max()), (I.min(), I.max()))
    for step in range(nsteps):
        t = (step + 1) * dt
        Sk, Ik, Rk = S.copy(), I.copy(), R.copy()
        rhsS_base = (M @ S) / dt
        rhsI_base = (M @ I) / dt
        rhsR_base = (M @ R) / dt

        for it in range(picard_maxit):
            
            fS, fI, fR = reaction_terms(Sk, Ik, Rk, nu, beta, mu, gamma)
            bS = rhsS_base + (M @ fS) + boundary_load
            bI = rhsI_base + (M @ fI) + boundary_load
            bR = rhsR_base + (M @ fR) + boundary_load

            Snew = cast(np.ndarray, spsolve(AS, bS))
            Inew = cast(np.ndarray, spsolve(AI, bI))
            Rnew = cast(np.ndarray, spsolve(AR, bR))

            Nnew = Snew + Inew + Rnew
            # Convergence check (relative-ish)
            err = max(
                np.linalg.norm(Snew - Sk),
                np.linalg.norm(Inew - Ik),
                np.linalg.norm(Rnew - Rk),
            )
            normU = max(np.linalg.norm(Snew), np.linalg.norm(Inew), np.linalg.norm(Rnew), 1.0)

            relative_change = err / normU
            Sk, Ik, Rk = Snew, Inew, Rnew
            if relative_change < picard_tol:
                break
        else:
            raise FloatingPointError(
                f"Picard failed to converge at t={t:.6g}; "
                f"last relative change={relative_change:.3e}"
            )

        S, I, R = Sk, Ik, Rk

        while (
            target_index < len(target_times)
            and t >= target_times[target_index] - 1e-12
        ):
            snapshots.append((float(t), S.copy(), I.copy(), R.copy()))
            target_index += 1
    return mesh, snapshots


def plot_snapshots(mesh, snapshots, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not snapshots:
        raise ValueError("No snapshots were generated.")

    vmin = min(np.min(I) for _, _, I, _ in snapshots)
    vmax = max(np.max(I) for _, _, I, _ in snapshots)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flat

    for ax in axes:
        ax.set_axis_off()

    for idx, (time, S, I, R) in enumerate(snapshots[:6]):
        ax = axes[idx]
        ax.set_axis_on()
        levels = 30
        contour = ax.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, I, levels=levels, vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_aspect("equal")
        ax.set_title(f"t = {time:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(contour, ax=ax, label="I")

    for ax in axes[min(len(snapshots), 6):]:
        ax.set_visible(False)

    combined_path = output_dir / "reaction_diffusion_SIR_snapshots.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)

    for idx, (time, S, I, R) in enumerate(snapshots):
        fig_i, ax_i = plt.subplots(figsize=(5, 4))
        contour = ax_i.tricontourf(
        mesh.p[0], 
        mesh.p[1], 
        I,
        levels=30, 
        vmin=vmin, 
        vmax=vmax, 
        cmap="magma")
        ax_i.set_aspect("equal")
        ax_i.set_title(f"Infected density at t = {time:.3f}")
        fig_i.colorbar(contour, ax=ax_i, label="I")
        fig_i.tight_layout()
        fig_i.savefig(output_dir / f"snapshot_{idx + 1:02d}.png", dpi=200)
        plt.close(fig_i)

    print(f"Saved 6 snapshot panels to {combined_path}")
    print(f"Saved individual snapshots in {output_dir}")


def main(mesh_filename: str | None = None,
    output_dir: str | Path = "figures/reaction_diffusion_SIR_snapshots",
    n_snapshots: int = 6,
    T: float = DEFAULT_T, 
    dt: float = 1e-3
    ):
    mesh, snapshots = run_simulation(
    mesh_filename=mesh_filename, 
    T=T, 
    dt=dt, 
    n_snapshots=n_snapshots
    )
    plot_snapshots(mesh, snapshots, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a population-conserving reaction-diffusion SIR model.")
    parser.add_argument("mesh", nargs="?", help="Optional path to a .msh file.")
    parser.add_argument("--output-dir", default="figures/reaction_diffusion_SIR_snapshots", help="Folder for saved snapshots and the combined figure.")
    parser.add_argument("--n-snapshots", type=int, default=6, help="Number of time snapshots to save.")
    parser.add_argument("--t-end", type=float, default=DEFAULT_T, help="Final time of the simulation.")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step for the implicit Euler solver.")
    args = parser.parse_args()

    main(
        mesh_filename=args.mesh,
        output_dir=args.output_dir,
        n_snapshots=args.n_snapshots,
        T=args.t_end,
        dt=args.dt,
    )
