import gmsh
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

gmsh.initialize()
gmsh.open("complex_mesh.geo")

lc_min = 0.08
lc_max = 0.2

gmsh.option.setNumber("Mesh.CharacteristicLengthMin",lc_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",lc_max)

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [1])
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_min)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_max)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.2)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.8)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.option.setNumber("Mesh.Algorithm", 1)


gmsh.model.mesh.generate(2)
types, tags, conns = gmsh.model.mesh.getElements(2)
print("2D element types:", types)

for t, c in zip(types, conns):
    props = gmsh.model.mesh.getElementProperties(t)
    name = props[0]
    nNodes = props[3]
    print(f"type {t}: {name}, nodes/elem={nNodes}, num_elems={len(c)//nNodes}")
    if "Triangle" in name and nNodes == 3:
        tri_conn = c
gmsh.write("hexagon_irregular_complex_mesh.msh")
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
element_types, element_tags, element_connectivities = gmsh.model.mesh.getElements(dim=2)
connectivity = element_connectivities[0]
gmsh.finalize()

if tri_conn is None:
    raise RuntimeError("No 3-node triangles found. Maybe you generated quads only, or no 2D mesh.")
node_tag_to_index = {tag: i for i, tag in enumerate(node_tags)}
num_elements = len(connectivity) // 3
triangles = np.array([node_tag_to_index[tag] for tag in connectivity], dtype=int).reshape(num_elements, 3)
X = node_coords[::3]
Y = node_coords[1::3]

# Create the plot
plt.figure(figsize=(8, 8))
plt.triplot(X, Y, triangles, 'g-', lw=0.5)
plt.title('Gmsh 2D Mesh Visualization')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.axis('equal')
plt.show()