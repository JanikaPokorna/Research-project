import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from skfem import MeshTri, Basis
from skfem.element import ElementTriP1
from mesh_utils import load_gmsh_tri

def initial_conditions(basis: Basis):
    """Nodal initial conditions: mostly S, with a Gaussian bump in I, R=0."""
    x = basis.doflocs[0]
    y = basis.doflocs[1]

    S0 = 10.0 * np.ones_like(x)
    I0 = 0.1 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))
    R0 = np.zeros_like(x)
    return S0, I0, R0

def sir_rhs_field(t, y, beta, gamma, eps=1e-12, use_local_N=True, N_global=None):
    ndof = y.size // 3
    Y = y.reshape(3, ndof)
    S, I, R = Y[0], Y[1], Y[2]

    if use_local_N:
        N = S + I + R + eps          # local population per node
    else:
        if N_global is None:
            raise ValueError("N_global must be provided when use_local_N=False.")
        N = N_global                 # scalar, broadcasted

    incidence = beta * S * I / N
    
    dS = -incidence
    dI = incidence - gamma * I
    dR = gamma * I

    return np.concatenate([dS, dI, dR])

def main(mesh_filename: str | None = None):
    msh_path = mesh_filename or str(Path(__file__).with_name("hexagon_mesh.msh"))
    mesh = load_gmsh_tri(msh_path)
    basis = Basis(mesh, ElementTriP1())

    beta = 3
    gamma = 0.5
    S0, I0, R0 = initial_conditions(basis)
    # Choose how to define N:
    # (1) Local N at each node: N_i = S_i+I_i+R_i (what your PDE code used)
    use_local_N = True

    # (2) Or global constant N: N = average or total
    N_global = None
    if not use_local_N:
        # Example: use a constant N equal to 10 everywhere
        N_global = 10.0

    y0 = np.concatenate([S0, I0, R0])

    # Time
    T = 10.0
    t_eval = np.linspace(0.0, T, 301)

    sol = solve_ivp(
        fun=lambda t, y: sir_rhs_field(t, y, beta, gamma, use_local_N=use_local_N, N_global=N_global),
        t_span=(0.0, T),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    ndof = basis.N
    S = sol.y[0:ndof, -1]
    I = sol.y[ndof:2*ndof, -1]
    R = sol.y[2*ndof:3*ndof, -1]


    # Plot final I
    mesh.draw()
    plt.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, I, levels=30)
    plt.colorbar(label="I")
    plt.gca().set_aspect("equal")
    plt.title("SIR on mesh (no diffusion): I at final time")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SIR model on a Gmsh mesh.")
    parser.add_argument("mesh", nargs="?", help="Path to a .msh file.")
    args = parser.parse_args()
    main(args.mesh)