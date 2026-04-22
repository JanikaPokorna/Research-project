import numpy as np
import meshio
from scipy.sparse.linalg import spsolve

from skfem import MeshTri, Basis, asm, FacetBasis, LinearForm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, unit_load, mass
from skfem.helpers import dot

def neumann_load(v, w):
    x, y = w.x[0], w.x[1]
    g = 0.0
    return g * v

def rhs(v, w):
    x, y = w.x[0], w.x[1]
    f = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    #f = np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)
    #f = ((x-0.5)**2 + (y-0.5)**2 < 0.15**2).astype(float)
    return f * v
    #f1 = np.exp(-((x-0.3)**2 + (y-0.3)**2) / 0.01)
    #f2 = np.exp(-((x-0.7)**2 + (y-0.6)**2) / 0.02)
    #return (f1 + f2) * v


def load_gmsh_tri(msh_path: str) -> MeshTri:
    msh = meshio.read(msh_path)
    tris = None
    for block in msh.cells:
        if block.type == "triangle":
            tris = block.data
            break
    if tris is None:
        raise ValueError("No triangle cells in .msh. This loader expects 2D triangles.")
    p = msh.points[:, :2].T
    t = tris.T
    return MeshTri(p, t)

def main():
    msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\mesh.msh"
    mesh = load_gmsh_tri(msh_path)
    
    basis = Basis(mesh, ElementTriP1())
    fbasis = FacetBasis(mesh, ElementTriP1(), facets=mesh.boundary_facets())

    k = 0.02
    alpha = 0.5

    D = k * asm(laplace, basis)
    M = alpha * asm(mass, basis)
    A = D + M
    bN = asm(neumann_load, fbasis)
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
    main()