import gmsh
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) #sets the directory as CWD


gmsh.initialize()
gmsh.open("complex_mesh.geo")

gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
gmsh.write("complex_hexagon_mesh.msh")
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
element_types, element_tags, element_connectivities = gmsh.model.mesh.getElements(dim=2)


connectivity = element_connectivities[0]
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