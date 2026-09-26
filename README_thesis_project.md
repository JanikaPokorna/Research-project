# SIR Reaction--Diffusion Research Project

## 1. Project aim

This project studies mathematical models of infectious-disease spread, with emphasis on the spatial SIR reaction--diffusion system and its numerical solution by the finite element method (FEM).

The current implementation supports:

- the classical non-spatial SIR model;
- independent SIR dynamics at mesh nodes;
- a steady diffusion--reaction FEM problem;
- a time-dependent SIR reaction--diffusion FEM model;
- Gmsh meshes with multiple triangle blocks and named boundary groups;
- implicit time integration, nonlinear Picard iteration, and rejected/retried time steps.

The immediate goal is to obtain a reliable baseline solver. Later work will focus on physically meaningful population movement and paired fluxes across internal interfaces.

## 2. Mathematical model

The spatial model has the general form

\[
\begin{aligned}
\partial_t S-D_S\Delta S &= f_S(S,I,R),\\
\partial_t I-D_I\Delta I &= f_I(S,I,R),\\
\partial_t R-D_R\Delta R &= f_R(S,I,R).
\end{aligned}
\]

The current solver treats \(S,I,R\) as local population densities and uses the local total density

\[
N=S+I+R
\]

in the frequency-dependent incidence term

\[
\mathcal I(S,I,R)=\beta\frac{SI}{N}.
\]

The implemented reaction functions are

\[
\begin{aligned}
f_S &= \nu(1-S)-\beta\frac{SI}{N}-\mu S,\\
f_I &= \beta\frac{SI}{N}-(\gamma+\mu)I,\\
f_R &= \gamma I-\mu R.
\end{aligned}
\]

A small numerical constant is added to \(N\) in the code to avoid division by zero. The current parameter choice is

\[
\nu=0,\qquad \mu=0,\qquad \beta=3,\qquad \gamma=0.5.
\]

The initial values satisfy \(S+I+R=1\) pointwise. The current diffusion coefficients are equal,

\[
D_S=D_I=D_R=10^{-3},
\]

which is consistent with preserving a spatially constant total density under zero-flux boundary conditions. The external boundary condition is

\[
D_U\nabla U\cdot n=0,
\qquad U\in\{S,I,R\}.
\]

## 3. Numerical method

The domain is discretized with triangular \(P_1\) finite elements. Let \(M\) denote the mass matrix and \(K\) the stiffness matrix. Backward Euler gives, for each compartment \(U\),

\[
\left(\frac{M}{\Delta t}+D_UK\right)U^{n+1}
=\frac{M}{\Delta t}U^n+F_U(S^{n+1},I^{n+1},R^{n+1}).
\]

The nonlinear reaction terms are treated by Picard iteration. The previous time step is used as the initial iterate:

\[
(S^{n+1,(0)},I^{n+1,(0)},R^{n+1,(0)})=(S^n,I^n,R^n).
\]

At Picard iteration \(k\), the reaction terms are evaluated at the current iterate and three linear systems are solved:

\[
\left(\frac{M}{\Delta t}+D_UK\right)U^{n+1,(k+1)}
=\frac{M}{\Delta t}U^n
+F_U(S^{n+1,(k)},I^{n+1,(k)},R^{n+1,(k)}).
\]

The adaptive snapshot solver additionally:

- rejects steps when Picard iteration fails;
- rejects non-finite or materially negative solutions;
- retries rejected steps with half the time step;
- clips only negative values within a specified numerical tolerance;
- uses row-sum mass lumping to reduce small negative undershoots.

This is rejection-based step-size control, not an error-estimator-based adaptive method.

## 4. Main Python files

| File | Purpose |
|---|---|
| `model_test.py` | Classical non-spatial SIR model solved with `solve_ivp`. |
| `run_SIR.py` | Independent SIR evolution at mesh nodes; no diffusion coupling. |
| `run_diffusion.py` | Steady diffusion--reaction problem using triangular `P1` FEM. |
| `run_reaction_diffusion.py` | Initial reaction--diffusion SIR implementation with backward Euler and Picard iteration. |
| `run_reaction_diffusion_SIR.py` | Main robust solver with multi-block meshes and named Neumann boundary fluxes. |
| `run_reaction_diffusion_SIR_snapshots_adaptive.py` | Current density-based solver with mass lumping, adaptive step rejection, and snapshots of all compartments. |
| `mesh_utils.py` | Shared Gmsh loader; combines triangle blocks and maps physical boundary groups to facets. |
| `mesh_inspect.py` | Reports mesh size, cell blocks, physical groups, and boundary edges. |
| `mesh_compare.py` | Produces a four-panel comparison of representative meshes. |
| `mesh generate.py` | Generates the star-split mesh. |
| `irregular mesh generate.py` | Generates a locally refined star-split mesh. |

## 5. Mesh handling

All mesh-based solvers use `mesh_utils.py`. The loader:

- reads every triangle block in a Gmsh file;
- removes unused points and remaps indices;
- supports conformingly fragmented domains;
- maps named physical curves, such as `Gamma_1`, to scikit-fem facets.

The main geometries are:

| Geometry or mesh | Description |
|---|---|
| `square_mesh.msh` | Regular unit-square benchmark. |
| `mesh.geo` / `mesh.msh` | Coarse irregular seven-sided domain. |
| `hexagon_starsplit_mesh.geo` | Hexagon divided into six connected sectors. |
| `complex_mesh.geo` | Hexagon fragmented into conforming subregions with shared interfaces. |
| `test_mesh.msh` | Locally refined star-split mesh used by the robust solver. |
| `hexagon_irregular_complex_mesh.msh` | Finer complex mesh used by the adaptive snapshot solver. |

The complete mesh inventory can be obtained with `mesh_inspect.py`; therefore, detailed point and triangle counts are not duplicated here.

## 6. Running the code

Mesh-based solvers accept an optional mesh path:

```text
python run_SIR.py [mesh.msh]
python run_diffusion.py [mesh.msh]
python run_reaction_diffusion.py [mesh.msh]
python run_reaction_diffusion_SIR.py [mesh.msh]
python run_reaction_diffusion_SIR_snapshots_adaptive.py [mesh.msh]
```

Examples:

```text
python run_diffusion.py test_mesh.msh
python mesh_inspect.py hexagon_irregular_complex_mesh.msh
python mesh_compare.py --output figures/mesh_comparison.png
python run_reaction_diffusion_SIR_snapshots_adaptive.py --t-end 7 --n-snapshots 6
```

Model parameters are currently configured inside the Python scripts.

## 7. Current status

- The basic SIR, nodewise SIR, and steady diffusion solvers run successfully.
- `run_reaction_diffusion.py` has been tested to \(t=1\).
- `run_reaction_diffusion_SIR.py` has been tested to \(t=3.5\) on multi-block meshes.
- Separate Neumann contributions can be assembled for each named external boundary; all default values are zero.
- The adaptive runner saves six equally spaced snapshots, including \(t=0\) and \(t=T\), for each of \(S,I,R\).
- Its default configuration uses `hexagon_irregular_complex_mesh.msh`, two Gaussian infection peaks, \(T=7\), and an initial/maximum step \(\Delta t=10^{-3}\).
- Short runs and forced large-step retries have been tested successfully.
- The full default adaptive run to \(T=7\) still requires documented verification.
- In a headless terminal, `plt.show()` may produce a harmless warning; saved figures are preferred.

## 8. Important modelling and numerical limitations

- The code currently calls \(S,I,R\) densities, while choosing initial values normalized so that \(S+I+R=1\). This convention and its units should be stated explicitly in the written report.
- Clipping small negative values changes total mass slightly; larger negative values are therefore rejected rather than silently corrected.
- The adaptive strategy reacts to solver failure and invalid states, but does not control the time-discretization error.
- Boundary-flux signs and units still require systematic validation.
- Internal-interface transport is not yet implemented. Ultimately, flux leaving one subdomain must enter the adjacent subdomain.
- Numerical validation against a problem with a known solution is still required.
- The nonlinear load is approximated by evaluating the reaction terms at the nodes and multiplying by the lumped mass matrix.

## 9. Next steps

1. State the density normalization and parameter units rigorously in the written model.
2. Validate the diffusion operator, boundary-flux convention, and conservation of total population.
3. Verify the full default adaptive simulation and document Picard convergence and rejected steps.
4. Test several initial infection distributions, time steps, and mesh resolutions.
5. Compare the spatially homogeneous PDE solution with the classical SIR ODE solution.
6. Implement paired internal-interface fluxes so that outflow from one subdomain equals inflow into its neighbour.
7. Transfer the mathematical formulation, discretization, and verified experiments into the written research report.
