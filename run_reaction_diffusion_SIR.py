import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from typing import cast

from skfem import Basis, FacetBasis, LinearForm, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, mass
from mesh_utils import load_gmsh_tri_with_boundary_groups

def initial_conditions(basis: Basis):
    """Example ICs: mostly S, small infected Gaussian bump, R=0."""
    x = basis.doflocs[0]
    y = basis.doflocs[1]

    S0 = 10.0 * np.ones_like(x)
    I0 = 0.1 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
    #I0 = np.zeros_like(x)
    R0 = np.zeros_like(x)

    return S0, I0, R0

def reaction_terms(S, I, R, nu, beta, mu, gamma, eps=1e-12):
    """Nodewise reactions f(S,I,R) (no diffusion)."""
    N = S + I + R + eps
    incidence = beta * (S * I / N)

    fS = nu - incidence - mu * S
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

def main(mesh_filename: str | None = None):
    #msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\mesh.msh"
    msh_path = mesh_filename or str(Path(__file__).with_name("test_mesh.msh"))
    #msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\hexagon_irregular_complex_mesh.msh"
    
    mesh, boundary_groups = load_gmsh_tri_with_boundary_groups(msh_path)
    basis = Basis(mesh, ElementTriP1())

    # Boundary fluxes are applied to S, I, and R on each named physical group.
    boundary_fluxes = {name: 0.0 for name in boundary_groups}
    boundary_load = assemble_boundary_fluxes(mesh, boundary_groups, boundary_fluxes, basis)

    K = asm(laplace, basis)
    M = asm(mass, basis)

    nu = 1.0
    beta = 3
    mu = 0.2
    gamma = 0.5

    DS = 1e-3
    DI = 1e-3
    DR = 1e-3

    dt = 1e-3
    T = 3.5
    nsteps = int(np.ceil(T / dt))

    #implicit Euler
    AS = (M / dt) + DS * K 
    AI = (M / dt) + DI * K
    AR = (M / dt) + DR * K

    S, I, R = initial_conditions(basis)

    picard_maxit = 20
    picard_tol = 1e-8

    for step in range(nsteps):
        t = (step + 1) * dt

        # Picard iteration: start from previous time step as initial guess
        Sk, Ik, Rk = S.copy(), I.copy(), R.copy()

        #right-hand side contribution from prev time step
        rhsS_base = (M @ S) / dt
        rhsI_base = (M @ I) / dt
        rhsR_base = (M @ R) / dt

        for it in range(picard_maxit):
            fS, fI, fR = reaction_terms(Sk, Ik, Rk, nu, beta, mu, gamma)

            # Because reactions are nodewise, we put them into a FE load via M @ f
            bS = rhsS_base + (M @ fS) + boundary_load
            bI = rhsI_base + (M @ fI) + boundary_load
            bR = rhsR_base + (M @ fR) + boundary_load

            Snew = cast(np.ndarray, spsolve(AS, bS))
            Inew = cast(np.ndarray, spsolve(AI, bI))
            Rnew = cast(np.ndarray, spsolve(AR, bR))

            # Convergence check (relative-ish)
            err = max(
                np.linalg.norm(Snew - Sk),
                np.linalg.norm(Inew - Ik),
                np.linalg.norm(Rnew - Rk),
            )
            normU = max(np.linalg.norm(Snew), np.linalg.norm(Inew), np.linalg.norm(Rnew), 1.0)

            Sk, Ik, Rk = Snew, Inew, Rnew

            if err / normU < picard_tol:
                break

        S, I, R = Sk, Ik, Rk

        # Positivity clamp
        S = np.maximum(S, 0.0)
        I = np.maximum(I, 0.0)
        R = np.maximum(R, 0.0)

        if step % 20 == 0 or step == nsteps - 1:
            print(f"t={t:.3f}, Picard iters={it+1}, "
                  f"S[min,max]=({S.min():.3e},{S.max():.3e}), "
                  f"I[min,max]=({I.min():.3e},{I.max():.3e}), "
                  f"R[min,max]=({R.min():.3e},{R.max():.3e})")

    # Plot final I
    print("mesh.p shape:", mesh.p.shape)   # expected (2, n_vertices)
    print("mesh.t shape:", mesh.t.shape)   # expected (3, n_triangles)
    print("I shape:", I.shape)             # expected (n_vertices,)
    mesh.draw()
    plt.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, I, levels=30)
    plt.colorbar(label="I")
    plt.gca().set_aspect("equal")
    plt.title("Infected I at final time")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the robust reaction-diffusion SIR model on a Gmsh mesh.")
    parser.add_argument("mesh", nargs="?", help="Path to a .msh file.")
    args = parser.parse_args()
    main(args.mesh)