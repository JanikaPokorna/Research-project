import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from skfem import MeshTri, Basis, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, mass

"""def load_gmsh_tri(msh_path: str) -> MeshTri:
    msh = meshio.read(msh_path)
    tris = None
    for block in msh.cells:
        if block.type == "triangle":
            tris = block.data
            break
    if tris is None:
        raise ValueError("No triangle cells in .msh. This loader expects 2D triangles.")
    used = np.unique(tris) # find which point indices are actually used by triangles
    points_used = msh.points[used, :2]
    old_to_new = {old: new for new, old in enumerate(used)}
    tris_new = np.vectorize(old_to_new.get)(tris)
    p = points_used.T
    t = tris_new.T
    return MeshTri(p, t)"""

def load_gmsh_tri(msh_path: str) -> MeshTri:
    msh = meshio.read(msh_path)

    tri_blocks = [block.data for block in msh.cells if block.type == "triangle"]
    if not tri_blocks:
        raise ValueError("No triangle cells in .msh. This loader expects 2D triangles.")

    # stack all triangle blocks from all fragments/surfaces
    tris = np.vstack(tri_blocks)

    # keep only points actually used by these triangles
    used = np.unique(tris)
    points_used = msh.points[used, :2]

    # old point index -> new compact index
    old_to_new = {old: new for new, old in enumerate(used)}
    tris_new = np.vectorize(old_to_new.get)(tris)

    p = points_used.T
    t = tris_new.T

    return MeshTri(p, t)

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

def main():
    #msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\mesh.msh"
    msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\test_mesh.msh"
    #msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\hexagon_irregular_complex_mesh.msh"
    
    mesh = load_gmsh_tri(msh_path)
    basis = Basis(mesh, ElementTriP1())

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
            bS = rhsS_base + (M @ fS)
            bI = rhsI_base + (M @ fI)
            bR = rhsR_base + (M @ fR)

            Snew = spsolve(AS, bS)
            Inew = spsolve(AI, bI)
            Rnew = spsolve(AR, bR)

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
    main()