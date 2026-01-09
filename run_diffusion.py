import numpy as np
import meshio
from scipy.sparse.linalg import spsolve

from skfem import MeshTri, Basis, asm, FacetBasis, LinearForm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, unit_load, mass
from skfem.helpers import dot

def neumann_load(v, w):
    g = 0.0
    return g * v

def rhs(v, w):
    x, y = w.x[0], w.x[1]
    f = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    return f * v

def load_gmsh_tri(msh_path: str) -> MeshTri:
    msh = meshio.read(msh_path)
    print("Cell blocks:", [(c.type, len(c.data)) for c in msh.cells])
    print("Points shape:", msh.points.shape)
    print("cell_data keys:", list(msh.cell_data.keys()))
    tris = None
    for block in msh.cells:
        if block.type == "triangle":
            tris = block.data
            break
    if tris is None:
        raise ValueError("No triangle cells in .msh. This loader expects 2D triangles.")
    
    print("Triangles array shape:", tris.shape)  
    print("First triangle node indices:", tris[0])

    p = msh.points[:, :2].T
    t = tris.T

    print("p shape (should be 2 x N):", p.shape)
    print("t shape (should be 3 x ntri):", t.shape)

    return MeshTri(p,t)

def main():
    msh_path = r"C:\Users\janik\OneDrive\Dokumenty\škola\vejska\magisterske studium\diplomová práce\programky\hexagon_mesh.msh"
    mesh = load_gmsh_tri(msh_path)
    
    basis = Basis(mesh, ElementTriP1())
    fbasis = FacetBasis(mesh, ElementTriP1())

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