import argparse
from collections import Counter
from pathlib import Path

import gmsh


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GEOMETRY = PROJECT_DIR / "Testovaci_a_nepotrebne_kody" / "complex_mesh.geo"
DEFAULT_OUTPUT = PROJECT_DIR / "hexagon_irregular_complex_refined.msh"
DEFAULT_SIZE = 0.015


def add_internal_interface_group() -> int:
    curve_uses = Counter()
    for _, surface_tag in gmsh.model.getEntities(2):
        for dimension, curve_tag in gmsh.model.getBoundary(
            [(2, surface_tag)], combined=False, oriented=False
        ):
            if dimension == 1:
                curve_uses[curve_tag] += 1

    internal_curves = [tag for tag, uses in curve_uses.items() if uses > 1]
    if not internal_curves:
        raise RuntimeError("The complex geometry has no shared internal curves.")

    existing_tags = [
        tag for dimension, tag in gmsh.model.getPhysicalGroups(1) if dimension == 1
    ]
    physical_tag = max(existing_tags, default=0) + 1
    group_tag = gmsh.model.addPhysicalGroup(1, internal_curves, physical_tag)
    gmsh.model.setPhysicalName(1, group_tag, "InternalInterface")
    return len(internal_curves)


def generate_mesh(geometry: Path, output: Path, size: float) -> None:
    if size <= 0.0:
        raise ValueError("Mesh size must be positive.")

    geometry = geometry.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.open(str(geometry))
        internal_curve_count = add_internal_interface_group()
        gmsh.model.mesh.generate(2)

        triangle_count = 0
        for element_type in gmsh.model.mesh.getElements(dim=2)[0]:
            _, dimension, _, nodes_per_element, *_ = gmsh.model.mesh.getElementProperties(
                element_type
            )
            if dimension == 2 and nodes_per_element == 3:
                element_tags = gmsh.model.mesh.getElementsByType(element_type)[0]
                triangle_count += len(element_tags)

        if triangle_count == 0:
            raise RuntimeError("Gmsh did not generate any linear triangles.")

        gmsh.write(str(output))
        print(f"Generated {triangle_count} triangles at target size {size:g}.")
        print(f"Tagged {internal_curve_count} shared curves as InternalInterface.")
        print(f"Saved refined complex hexagon mesh to {output}")
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a refined mesh from the complex irregular hexagon geometry."
    )
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE)
    args = parser.parse_args()
    generate_mesh(args.geometry, args.output, args.size)


if __name__ == "__main__":
    main()