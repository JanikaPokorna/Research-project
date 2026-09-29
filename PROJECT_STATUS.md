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
| `run_reaction_diffusion_SIR.py` | More robust spatial SIR reaction-diffusion model | Implicit Euler, Picard iteration, multiple triangle blocks, named boundary fluxes |
| `run_reaction_diffusion_SIR_snapshots_adaptive.py` | Snapshot runner with rejected/retried time steps | Implicit Euler, Picard iteration, row-sum mass lumping, nonnegativity checks |
| `run_reaction_diffusion_SIR_mesh_comparison.py` | Generates three mesh resolutions and compares SIR snapshots | Gmsh mesh-size limits and shared adaptive solver |
| `mesh_utils.py` | Shared Gmsh mesh loader | Loads triangle blocks and maps physical boundary groups to facets |
| `mesh_inspect.py` | Mesh inspection utility | Reports points, cells, triangles, physical groups, and boundary edges |
| `mesh_compare.py` | Mesh comparison plotter | Creates a four-panel comparison of representative meshes |
| `mesh generate.py` | Generates the star-split mesh | Gmsh Python API |
| `irregular mesh generate.py` | Generates a locally refined irregular mesh | Gmsh size fields and Python API |
| `square mesh generate.py` | Generates a structured regular square mesh | Gmsh transfinite curves and surface |

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

It has been tested through `t = 3.5` successfully and is suitable for meshes containing multiple triangle blocks, such as the complex and star-split meshes. It also assembles separate Neumann fluxes for each physical boundary group; the default flux for every group is zero.

### `run_reaction_diffusion_SIR_snapshots_adaptive.py`

A separate snapshot-oriented runner that retries a time step when its Picard iteration fails or produces non-finite or materially negative compartment values. On rejection it halves the trial time step and restarts from the last accepted state. The run stops with an error if the step would need to go below `min_dt` or the rejection limit is exceeded. After an accepted step, the next proposed step may grow by a factor of 1.25, up to the configured `dt` maximum.

The solver uses row-sum mass lumping for the P1 finite-element mass matrix. This reduced the small negative infected-value undershoot seen on the selected complex mesh with a consistent mass matrix. Negative values no larger than the configured positivity tolerance are clipped to zero; that small projection can introduce a correspondingly small mass change. Larger negative values cause rejection rather than being silently clipped.

The current default configuration uses `hexagon_irregular_complex_mesh.msh`, two interior Gaussian infected peaks, zero outer-boundary flux, `T = 7`, initial/maximum `dt = 1e-3`, and `min_dt = 1e-8`. It saves six time snapshots for each of S, I, and R, including combined figures and individual images under `figures/reaction_diffusion_SIR_snapshots_adaptive`. This is rejection-based step-size adaptation, not an error-estimator method: it reduces the step based on Picard convergence and state-validity checks, not an estimate of time-discretization error. Row-sum mass lumping is used to reduce small negative P1 undershoots; negative values within a small tolerance are clipped to zero, while materially negative/non-finite trials or Picard non-convergence trigger step rejection and retry. Short simulation and forced-rejection tests passed; the full default-duration run has not been verified.

### `run_reaction_diffusion_SIR_mesh_comparison.py`

Generates three meshes from `complex_mesh.geo` by setting Gmsh's global minimum and maximum mesh-size limits, then runs the same adaptive SIR solver on each mesh. The default rows are coarse (`h_min = 0.10`, `h_max = 0.35`), medium (`0.07`, `0.25`), and fine (`0.035`, `0.14`). It produces one 3-by-9 figure: one row per resolution, and groups of three time snapshots for S, I, and R. It also writes mesh node/triangle counts to `mesh_resolution_summary.csv`. Generated meshes are stored in `meshes_mesh_size_comparison`; the figure and summary default to `figures/mesh_size_comparison`. The size settings are editable in the `MESH_RESOLUTIONS` tuple near the top of the script. A short end-to-end comparison passed; the default final-time comparison has not been verified.

## Geometry Sources and Mesh Generators

### `square_mesh`

This is a Gmsh geometry file without a `.geo` extension. It defines a unit square with four named boundaries:

- `Gamma_bottom`
- `Gamma_right`
- `Gamma_top`
- `Gamma_left`

It uses transfinite meshing with 11 points per side, producing 10 equal intervals along each edge. Run `python "square mesh generate.py"` to create `square_regular_mesh.msh` with first-order triangles and the same named boundary groups.

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

This shared utility provides `load_gmsh_tri(...)` and `load_gmsh_tri_with_boundary_groups(...)`. It accepts either a string path or a `Path`, loads every triangle block in the Gmsh file, removes unused points, remaps point indices, and can map named physical curve groups such as `Gamma_1` to scikit-fem facet indices.

### `mesh_inspect.py`

This utility reports the structure of a Gmsh mesh without running a simulation. It displays:

- Point count
- Cell blocks and element counts
- Triangle block count and total triangle count
- Gmsh physical groups and their dimensions/tags
- Boundary-edge count computed from triangle connectivity
- Physical tags attached to each cell block

Run it with:

```text
python mesh_inspect.py heptagon_irregular_mesh.msh
```

### `mesh_compare.py`

This utility creates `mesh_comparison.png`, a comparison of:

- The coarse irregular heptagon
- The refined star-split mesh
- The star-split hexagon
- The complex irregular hexagon
- The existing square mesh
- The regular square mesh

Each panel shows the mesh geometry and its point/triangle counts. Run it with:

```text
python mesh_compare.py
```

Use `--output` to choose another image path:

```text
python mesh_compare.py --output figures/mesh_comparison.png
```

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
| `square_mesh.msh` | Unit square | 144 | 246 | Existing unstructured mesh; kept unchanged |
| `square_regular_mesh.msh` | Structured unit square | 121 | 200 | 10 equal intervals per side; generated by `square mesh generate.py` |
| `mesh.msh` | Irregular seven-sided polygon | 15 | 19 | Very coarse test mesh |
| `heptagon_mesh.msh` | Irregular seven-sided polygon | 15 | 19 | Very coarse test mesh |
| `hexagon_mesh.msh` | Irregular six-sided polygon | 66 | 105 | Standard reaction-diffusion mesh |
| `hexagon_starsplit_mesh.msh` | Six-sector star-split hexagon | 73 | 126 | Six triangle blocks |
| `test_mesh.msh` | Refined star-split hexagon | 99 | 170 | Current robust SIR test mesh |
| `complex_mesh.msh` | Internally fragmented hexagon | 43 | 66 | Multiple conforming regions |
| `complex_hexagon_mesh.msh` | Internally fragmented hexagon | 43 | 66 | Similar to `complex_mesh.msh` |
| `hexagon_irregular_complex_mesh.msh` | Finer irregular complex hexagon | 122 | 216 | More detailed complex mesh |
| `heptagon_irregular_mesh.msh` | Irregular seven-sided polygon | 15 | 19 | Regenerated from `simple_heptagon_mesh.geo` |

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

### Mesh physical groups

The shared loader now maps physical curve groups to scikit-fem facets. `run_diffusion.py` and `run_reaction_diffusion_SIR.py` assemble Neumann contributions separately for every named group using a `boundary_fluxes` dictionary. The default value for every group is `0.0`, preserving the previous zero-flux behavior; individual values can be changed per edge.

### Plotting

Running with a non-interactive Matplotlib backend produces a warning at `plt.show()`. This is expected in a headless terminal. Use a graphical Python environment or replace `plt.show()` with `plt.savefig(...)` when an image file is preferred.

### Configuration

Mesh filenames can now be supplied to every mesh-based solver as an optional command-line argument. Model parameters remain configured in the Python files.

## Completed Recommendations

The recommendations from the initial project review are complete:

- Shared multi-block mesh loading was moved to `mesh_utils.py`.
- All mesh-based solvers accept an optional mesh filename from the command line.
- `heptagon_irregular_mesh.msh` was regenerated with triangle elements.
- `mesh_inspect.py` reports mesh structure, physical groups, and boundary edges.
- `mesh_compare.py` generates plots comparing representative meshes.
- Physical boundary groups are used for separate Neumann flux assembly in `run_diffusion.py` and `run_reaction_diffusion_SIR.py`.

There are no outstanding recommendations from the original project review.

## Next steps

- Perform time-step and mesh-refinement convergence studies; the adaptive rejection checks protect solver validity but do not estimate discretization error.
- Create a validation script for the basic steady diffusion equation solver; define benchmark problem and expected convergence behavior.
- Verify and document Neumann-flux sign convention and boundary-group assignment. Current reaction-diffusion solvers use zero exterior flux by default.
- Decide whether the commented paired interface-flux example should become an implemented model. The current single continuous conforming FEM solve already couples internal interfaces; explicit equal-and-opposite interface loads are needed for separate regional solves.
- Complete a full-duration adaptive SIR run and mesh-resolution comparison, and review the saved figures and numerical diagnostics.
- Assemble the project methods and results into the thesis LaTeX document.