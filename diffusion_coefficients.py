import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from mesh_utils import load_gmsh_tri_with_boundary_groups

DEFAULT_MESH = "hexagon_irregular_complex_very_refined.msh"
DEFAULT_OUTPUT = "figures/nonuniform_diffusion_preview.png"
DEFAULT_PROFILE_OUTPUT = "figures/diffusion_profile_along_region_line.png"
DEFAULT_D_HIGH = 1.0e-3
DEFAULT_D_EPSILON = 1.0e-7
DEFAULT_TRANSITION_WIDTH = 0.12
INTERFACE_GROUP = "InternalInterface"


def get_interface_segments(mesh, boundary_groups):
    facets = boundary_groups.get(INTERFACE_GROUP)
    if facets is None or len(facets) == 0:
        raise ValueError(
            f"Mesh is missing the {INTERFACE_GROUP!r} physical group; regenerate it "
            "with generate_refined_complex_mesh.py."
        )
    if mesh.facets is None:
        raise ValueError("Mesh facets are unavailable.")
    return mesh.p[:, mesh.facets[:, facets]].transpose(2, 1, 0)


def distance_to_interfaces(x, y, interface_segments):
    """Compute each point's shortest distance to the internal line segments."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    points = np.column_stack((x.ravel(), y.ravel()))
    segments = np.asarray(interface_segments, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or len(segments) == 0:
        raise ValueError("interface_segments must have shape (n, 2, 2) and be nonempty.")

    starts = segments[:, 0, :]
    directions = segments[:, 1, :] - starts
    lengths_squared = np.einsum("si,si->s", directions, directions)
    if np.any(lengths_squared <= 0.0):
        raise ValueError("Interface segments must have nonzero length.")

    distances = np.empty(len(points), dtype=float)
    for offset in range(0, len(points), 2048):
        block = points[offset:offset + 2048]
        relative = block[:, None, :] - starts[None, :, :]
        parameter = np.einsum("pmi,mi->pm", relative, directions)
        parameter /= lengths_squared[None, :]
        parameter = np.clip(parameter, 0.0, 1.0)
        nearest = starts[None, :, :] + parameter[:, :, None] * directions[None, :, :]
        squared = np.sum((block[:, None, :] - nearest) ** 2, axis=2)
        distances[offset:offset + len(block)] = np.sqrt(squared.min(axis=1))

    return distances.reshape(x.shape)


def diffusion_coefficient(
    x,
    y,
    interface_segments,
    D_high=DEFAULT_D_HIGH,
    D_epsilon=DEFAULT_D_EPSILON,
    transition_width=DEFAULT_TRANSITION_WIDTH,
):
    """Set D to epsilon on interfaces and smoothly raise it into each region."""
    if not np.isfinite(D_epsilon) or not np.isfinite(D_high) or D_epsilon <= 0.0 or D_high <= D_epsilon:
        raise ValueError("Require finite values with 0 < D_epsilon < D_high.")
    if not np.isfinite(transition_width) or transition_width <= 0.0:
        raise ValueError("transition_width must be finite and positive.")

    distance = distance_to_interfaces(x, y, interface_segments)
    scaled_distance = np.clip(distance / transition_width, 0.0, 1.0)
    smooth_ramp = scaled_distance**2 * (3.0 - 2.0 * scaled_distance)
    return D_epsilon + (D_high - D_epsilon) * smooth_ramp


def plot_diffusion_field(
    mesh,
    boundary_groups,
    output_path: str | Path,
    D_high=DEFAULT_D_HIGH,
    D_epsilon=DEFAULT_D_EPSILON,
    transition_width=DEFAULT_TRANSITION_WIDTH,
):
    interface_segments = get_interface_segments(mesh, boundary_groups)
    points = mesh.p.T
    values = diffusion_coefficient(
        points[:, 0],
        points[:, 1],
        interface_segments,
        D_high=D_high,
        D_epsilon=D_epsilon,
        transition_width=transition_width,
    )

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    triangulation = Triangulation(mesh.p[0], mesh.p[1], triangles=mesh.t.T)
    sc = ax.tripcolor(
        triangulation,
        values,
        cmap="viridis",
        norm=LogNorm(vmin=D_epsilon, vmax=D_high),
        shading="gouraud",
    )
    ax.add_collection(
        LineCollection(interface_segments, colors="#440154", linewidths=0.35, zorder=3)
    )
    exterior_facets = boundary_groups.get("Exterior")
    if exterior_facets is not None and mesh.facets is not None:
        exterior_segments = mesh.p[:, mesh.facets[:, exterior_facets]].transpose(2, 1, 0)
        ax.add_collection(
            LineCollection(exterior_segments, colors="black", linewidths=0.65, zorder=4)
        )
    ax.set_aspect("equal")
    ax.set_title("Distance-based diffusion coefficient")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, label="D(x, y)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved distance-based diffusion preview to {output}")


def plot_interface_profile(
    interface_segments,
    output_path=DEFAULT_PROFILE_OUTPUT,
    D_high=DEFAULT_D_HIGH,
    D_epsilon=DEFAULT_D_EPSILON,
    transition_width=DEFAULT_TRANSITION_WIDTH,
):
    lengths = np.linalg.norm(interface_segments[:, 1, :] - interface_segments[:, 0, :], axis=1)
    segment = interface_segments[int(np.argmax(lengths))]
    tangent = segment[1] - segment[0]
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    midpoint = segment.mean(axis=0)
    signed_distance = np.linspace(-transition_width, transition_width, 301)
    points = midpoint + signed_distance[:, None] * normal
    values = diffusion_coefficient(
        points[:, 0],
        points[:, 1],
        interface_segments,
        D_high=D_high,
        D_epsilon=D_epsilon,
        transition_width=transition_width,
    )

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(signed_distance, values, color="tab:blue", linewidth=2)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Signed distance from internal interface")
    ax.set_ylabel("D")
    ax.set_title("Diffusion coefficient across an internal interface")
    ax.set_ylim(D_epsilon * 0.8, D_high * 1.05)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved interface diffusion profile to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a distance-based coefficient field on the refined complex mesh."
    )
    parser.add_argument("mesh", nargs="?", default=DEFAULT_MESH, help="Path to a .msh file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-output", default=DEFAULT_PROFILE_OUTPUT)
    parser.add_argument("--d-high", type=float, default=DEFAULT_D_HIGH)
    parser.add_argument("--d-epsilon", type=float, default=DEFAULT_D_EPSILON)
    parser.add_argument("--transition-width", type=float, default=DEFAULT_TRANSITION_WIDTH)
    args = parser.parse_args()

    mesh, boundary_groups = load_gmsh_tri_with_boundary_groups(args.mesh)
    interface_segments = get_interface_segments(mesh, boundary_groups)
    plot_diffusion_field(
        mesh,
        boundary_groups,
        args.output,
        D_high=args.d_high,
        D_epsilon=args.d_epsilon,
        transition_width=args.transition_width,
    )
    plot_interface_profile(
        interface_segments,
        output_path=args.profile_output,
        D_high=args.d_high,
        D_epsilon=args.d_epsilon,
        transition_width=args.transition_width,
    )
    print(
        f"Parameters: D_high={args.d_high:.3e}, D_epsilon={args.d_epsilon:.3e}, "
        f"transition_width={args.transition_width:.3f}"
    )


if __name__ == "__main__":
    main()
