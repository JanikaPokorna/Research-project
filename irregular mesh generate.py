# generates a mesh with adjustable size of elements, 
# or multiple areas with meshes, well conected
# and saves as .msh file

import gmsh
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

gmsh.initialize()
geo_file = "hexagon_starsplit_mesh.geo"
gmsh.open(geo_file)

lc_min = 0.05
lc_max = 0.2

gmsh.option.setNumber("Mesh.CharacteristicLengthMin",lc_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",lc_max)

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [1])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.02)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_max)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.option.setNumber("Mesh.Algorithm", 1)


gmsh.model.mesh.generate(2)
tri_conn = None

types, elem_tags, elem_conns = gmsh.model.mesh.getElements(dim=2)
print("2D element types:", types)

for etype, con in zip(types, elem_conns):
    name, dim, order, nNodes, *_ = gmsh.model.mesh.getElementProperties(etype)
    print(f"type {etype}: {name}, nodes/elem={nNodes}, num_elems={len(con)//nNodes}")
    if dim == 2 and "Triangle" in name and nNodes == 3:
        tri_conn = con  # this is a flat list of node tags [n1,n2,n3, n1,n2,n3, ...]
        break

if tri_conn is None:
    gmsh.finalize()
    raise RuntimeError("No 3-node triangles found. Maybe you generated quads only or higher-order elements.")

# Save mesh (useful for meshio -> skfem later)
gmsh.write("test_mesh.msh")

# --- Nodes ---
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
gmsh.finalize()

# Map gmsh node tag -> 0..N-1 index for plotting
node_tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

# Convert triangle connectivity (node tags) -> indices
tri_conn = np.array(tri_conn, dtype=np.int64)
triangles = np.array([node_tag_to_index[tag] for tag in tri_conn], dtype=int).reshape(-1, 3)

# Extract x,y (coords are [x0,y0,z0,x1,y1,z1,...])
X = node_coords[0::3]
Y = node_coords[1::3]

plt.figure(figsize=(8, 8))
plt.triplot(X, Y, triangles, lw=0.5)
plt.title("Gmsh 2D Mesh Visualization (P1 triangles)")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.axis("equal")
plt.show()