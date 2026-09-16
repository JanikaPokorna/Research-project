import argparse
import numpy as np
from pathlib import Path
from scipy.sparse.linalg import spsolve

from skfem import MeshTri, Basis, asm, FacetBasis, LinearForm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, unit_load, mass
from skfem.helpers import dot
from mesh_utils import load_gmsh_tri_with_boundary_groups

@LinearForm
def rhs(v, w):
    x, y = w.x[0], w.x[1]
    f = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    #f = np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)
    #f = ((x-0.5)**2 + (y-0.5)**2 < 0.15**2).astype(float)
    return f * v
    #f1 = np.exp(-((x-0.3)**2 + (y-0.3)**2) / 0.01)
    #f2 = np.exp(-((x-0.7)**2 + (y-0.6)**2) / 0.02)
    #return (f1 + f2) * v


def main(mesh_filename: str | None = None):
    msh_path = mesh_filename or str(Path(__file__).with_name("mesh.msh"))
    mesh, boundary_groups = load_gmsh_tri_with_boundary_groups(msh_path)
    
    basis = Basis(mesh, ElementTriP1())
    k = 0.02
    alpha = 0.5

    D = k * asm(laplace, basis)
    M = alpha * asm(mass, basis)
    A = D + M
    boundary_fluxes = {name: 0.0 for name in boundary_groups}
    bN = np.zeros(basis.N)
    for group_name, facets in boundary_groups.items():
        flux = boundary_fluxes[group_name]

        @LinearForm
        def neumann_load(v, w):
            return flux * v

        fbasis = FacetBasis(mesh, ElementTriP1(), facets=facets)
        bN += asm(neumann_load, fbasis)
    b = asm(rhs, basis) + bN
    print("A shape:", A.shape, "nnz:", A.nnz)
    print("b shape:", b.shape, "b min/max:", float(b.min()), float(b.max()))

    u = spsolve(A, b)
    import matplotlib.pyplot as plt

    mesh.draw()
    plt.tricontourf(mesh.p[0], mesh.p[1], mesh.t.T, u, levels=30)
    plt.colorbar(label="u")
    plt.gca().set_aspect("equal")
    plt.title("Solution u")
    plt.show()

    print("Solved. u min/max:", float(u.min()), float(u.max()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the steady diffusion problem on a Gmsh mesh.")
    parser.add_argument("mesh", nargs="?", help="Path to a .msh file.")
    args = parser.parse_args()
    main(args.mesh)