# Research Project Status

## Project Overview

This project investigates SIR epidemic models and spatial reaction-diffusion models on two-dimensional domains. The spatial solvers use Gmsh meshes, `meshio`, `scikit-fem`, NumPy, SciPy, and Matplotlib.

The project currently contains four main model types:

- A basic non-spatial SIR ordinary differential equation model.
- A nodewise SIR model evaluated on a mesh without diffusion coupling.
- A steady diffusion/reaction finite-element problem.
- A time-dependent SIR reaction-diffusion finite-element model.

## Python Codes

| File | Purpose | Main method |
|---|---|---|
| `model_test.py` | Basic one-dimensional SIR model | `scipy.integrate.solve_ivp` |
| `run_SIR.py` | SIR model at mesh nodes without spatial diffusion | `solve_ivp` and `MeshTri` |
| `run_diffusion.py` | Steady diffusion/reaction problem | P1 FEM, stiffness matrix, mass matrix, Neumann load |
| `run_reaction_diffusion.py` | Spatial SIR reaction-diffusion model | Implicit Euler and Picard iteration |
| `run_reaction_diffusion_SIR.py` | More robust spatial SIR reaction-diffusion model | Implicit Euler, Picard iteration, multiple triangle blocks |
| `mesh_utils.py` | Shared Gmsh triangle mesh loader | Combines all triangle blocks and compacts used points |
| `mesh generate.py` | Generates the star-split mesh | Gmsh Python API |
| `irregular mesh generate.py` | Generates a locally refined irregular mesh | Gmsh size fields and Python API |

## Current Solver Details

### `model_test.py`

A simple non-spatial SIR model with scalar values for susceptible, infected, and recovered populations. It is useful for testing the epidemic equations independently of meshes and FEM.

### `run_SIR.py`

Applies the SIR equations independently at mesh degrees of freedom. It supports either a local population value at each node or a global constant population value. There is no diffusion coupling between neighboring nodes.

### `run_diffusion.py`

Solves a steady problem of the form

`-k Laplacian(u) + alpha u = f`

It uses:

- `ElementTriP1`
- A stiffness matrix from `laplace`
- A mass matrix from `mass`
- A volume load decorated with `@LinearForm`
- A boundary load decorated with `@LinearForm`
- A repository-relative `mesh.msh` path

The script has been tested successfully. Its current mesh contains 15 points and 19 triangles.

### `run_reaction_diffusion.py`

Solves a spatial SIR model with separate diffusion constants for `S`, `I`, and `R`. It uses:

- Implicit Euler time stepping
- Picard iteration for nonlinear reaction terms
- Positivity clamping
- A repository-relative `hexagon_mesh.msh` path

The script has been tested through `t = 1.0` successfully.

### `run_reaction_diffusion_SIR.py`

This is the more robust reaction-diffusion implementation. It uses the shared loader in `mesh_utils.py`, which combines all triangle blocks from a Gmsh file and compacts the used point indices. It currently uses the repository-relative `test_mesh.msh` path.

It has been tested through `t = 3.5` successfully and is suitable for meshes containing multiple triangle blocks, such as the complex and star-split meshes.

## Geometry Sources and Mesh Generators

### `square_mesh`

This is a Gmsh geometry file without a `.geo` extension. It defines a unit square with four named boundaries:

- `Gamma_bottom`
- `Gamma_right`
- `Gamma_top`
- `Gamma_left`

Its generated mesh is `square_mesh.msh`.

### `mesh.geo`

Defines an irregular seven-sided polygon. It has one plane surface and seven named boundary curves `Gamma_1` through `Gamma_7`. Despite some filenames using the word hexagon, this geometry has seven outer vertices.

### `simple_hexagon_mesh.geo`

Defines another irregular seven-sided polygon with seven outer boundary edges. It is also technically a heptagon rather than a hexagon.

### `hexagon_starsplit_mesh.geo`

Defines a six-sided outer polygon with one central point and six lines from the center to the outer vertices. Boolean fragmentation creates six connected sectors.

`mesh generate.py` currently opens this file and writes `hexagon_starsplit_mesh.msh`.

### `complex_mesh.geo`

Defines a hexagonal outer boundary with several interior points and connecting curves. Boolean fragmentation divides the domain into multiple conforming regions with shared internal interfaces.

`complex_mesh.msh` and `complex_hexagon_mesh.msh` are generated examples of this geometry.

### `mesh generate.py`

Current behavior:

- Opens `hexagon_starsplit_mesh.geo`
- Generates a two-dimensional mesh
- Uses Gmsh mesh optimization
- Writes `hexagon_starsplit_mesh.msh`
- Plots the resulting triangular mesh

The commented lines can be changed to generate `mesh.msh` from another geometry, but the output filename and input geometry should be kept consistent.

### `irregular mesh generate.py`

Current behavior:

- Opens `hexagon_starsplit_mesh.geo`
- Sets minimum and maximum element sizes
- Uses a distance field centered around point `1`
- Uses a threshold field for local refinement
- Generates 3-node triangles
- Writes `test_mesh.msh`
- Plots the mesh

This is currently the main generator for a locally refined star-split mesh.

### `mesh_utils.py`

This shared utility provides `load_gmsh_tri(...)` for all mesh-based solvers. It accepts either a string path or a `Path`, loads every triangle block in the Gmsh file, removes unused points, remaps point indices, and returns a `scikit-fem` `MeshTri`.

## Command-Line Mesh Selection

All mesh-based solvers accept an optional positional mesh filename. If no filename is provided, each solver keeps its current default:

```text
python run_SIR.py [mesh.msh]
python run_diffusion.py [mesh.msh]
python run_reaction_diffusion.py [mesh.msh]
python run_reaction_diffusion_SIR.py [mesh.msh]
```

For example, to run the diffusion solver on the refined multi-block mesh:

```text
python run_diffusion.py test_mesh.msh
```

The default mesh is `hexagon_mesh.msh` for `run_SIR.py` and `run_reaction_diffusion.py`, `test_mesh.msh` for `run_reaction_diffusion_SIR.py`, and `mesh.msh` for `run_diffusion.py`.

## Mesh Inventory

| Mesh | Geometry / structure | Points | Triangles | Notes |
|---|---|---:|---:|---|
| `square_mesh.msh` | Unit square | 144 | 246 | Regular benchmark mesh |
| `mesh.msh` | Irregular seven-sided polygon | 15 | 19 | Very coarse test mesh |
| `heptagon_mesh.msh` | Irregular seven-sided polygon | 15 | 19 | Very coarse test mesh |
| `hexagon_mesh.msh` | Irregular six-sided polygon | 66 | 105 | Standard reaction-diffusion mesh |
| `hexagon_starsplit_mesh.msh` | Six-sector star-split hexagon | 73 | 126 | Six triangle blocks |
| `test_mesh.msh` | Refined star-split hexagon | 99 | 170 | Current robust SIR test mesh |
| `complex_mesh.msh` | Internally fragmented hexagon | 43 | 66 | Multiple conforming regions |
| `complex_hexagon_mesh.msh` | Internally fragmented hexagon | 43 | 66 | Similar to `complex_mesh.msh` |
| `hexagon_irregular_complex_mesh.msh` | Finer irregular complex hexagon | 122 | 216 | More detailed complex mesh |
| `heptagon_irregular_mesh.msh` | Boundary-only irregular geometry | 79 | 0 | Not usable by current FEM loaders |

The triangle counts above include all triangle blocks in each mesh.

## What Can Currently Be Generated

The project can generate and simulate on:

- A regular square domain
- A coarse irregular seven-sided polygon
- A standard irregular six-sided polygon
- A six-sector star-split hexagon
- A locally refined star-split hexagon
- A conformingly fragmented complex hexagonal domain
- A finer irregular complex hexagonal domain

The solvers can currently represent:

- Non-spatial SIR dynamics
- Nodewise SIR dynamics on a mesh
- Steady diffusion and reaction
- Time-dependent reaction-diffusion SIR dynamics
- Different diffusion rates for susceptible, infected, and recovered populations
- Nonlinear infection terms
- Local or global population normalization in the basic SIR solver
- Positive-population clamping after each time step

## Important Limitations

### Triangle loading

All current mesh-based solvers now use `mesh_utils.py`, which loads every triangle block:

```python
for block in msh.cells:
    if block.type == "triangle":
    triangle_blocks.append(block.data)
```

This supports single-block meshes as well as meshes such as `test_mesh.msh` and the complex meshes.

### `heptagon_irregular_mesh.msh`

This file contains boundary line elements but no triangles. It cannot currently be used by the FEM solvers and will produce:

```text
No triangle cells in .msh
```

It needs to be regenerated as a two-dimensional mesh before use.

### Mesh physical groups

The `.msh` files contain physical names for domains and boundaries, but the current loaders mostly detect the outer boundary geometrically through `mesh.boundary_facets()`. They do not yet use the Gmsh physical group names to apply different boundary conditions to individual edges.

### Plotting

Running with a non-interactive Matplotlib backend produces a warning at `plt.show()`. This is expected in a headless terminal. Use a graphical Python environment or replace `plt.show()` with `plt.savefig(...)` when an image file is preferred.

### Configuration

Mesh selection is currently hard-coded inside each solver. A useful future improvement would be to accept the mesh filename and model parameters through command-line arguments or a configuration file.

## Recommended Next Steps

1. Regenerate `heptagon_irregular_mesh.msh` with triangle elements.
2. Add a mesh inspection script that reports points, triangles, physical groups, and boundary edges.
3. Add plots comparing coarse, refined, star-split, and complex meshes.
4. Use physical boundary groups for separate boundary conditions on different edges.
