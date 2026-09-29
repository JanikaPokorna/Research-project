from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
GEOMETRY_FILE = PROJECT_DIR / "square_mesh"
OUTPUT_FILE = PROJECT_DIR / "square_regular_mesh.msh"

gmsh.initialize()
try:
    gmsh.open(str(GEOMETRY_FILE))
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(OUTPUT_FILE))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, _, element_connectivities = gmsh.model.mesh.getElements(dim=2)

    triangle_connectivity = None
    for element_type, connectivity in zip(element_types, element_connectivities):
        name, dimension, _, nodes_per_element, *_ = gmsh.model.mesh.getElementProperties(element_type)
        if dimension == 2 and name == "Triangle 3":
            triangle_connectivity = connectivity
            break

    if triangle_connectivity is None:
        raise RuntimeError("Gmsh did not generate first-order triangles.")

    node_tag_to_index = {tag: index for index, tag in enumerate(node_tags)}
    triangles = np.array(
        [node_tag_to_index[tag] for tag in triangle_connectivity], dtype=int
    ).reshape(-1, 3)
    x_coords = node_coords[0::3]
    y_coords = node_coords[1::3]
finally:
    gmsh.finalize()

print(f"Wrote {OUTPUT_FILE.name}: {len(node_tags)} points, {len(triangles)} triangles")

plt.figure(figsize=(8, 8))
plt.triplot(x_coords, y_coords, triangles, "g-", linewidth=0.5)
plt.title("Regular Gmsh square mesh (P1 triangles)")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.axis("equal")
plt.show()