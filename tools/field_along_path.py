"""field_along_path.py
Interpolate simulation-field values at each waypoint of a :class:`~trajectory.Trajectory`
in a *single snapshot* and produce publication-ready figures.

Public API
----------
``PathFieldConfig``
    All settings for one job: which run, which snapshot, which fields,
    figure options.  Built from a TOML file via ``PathFieldConfig.from_toml``.

``PathFieldPlotter``
    Consumes a ``PathFieldConfig``.  Call :meth:`~PathFieldPlotter.run` to
    produce (and optionally save) the figures.

TOML layout
-----------
See ``path_field_example.toml`` for a worked example.  Required sections:

    [header]           – run paths, snapshot step, output folder
    [[variable]]       – one block per field component to plot
    [path]             – how to construct the ``Trajectory``
                         (orbit | line | csv)

Usage (CLI)
-----------
    python field_along_path.py path_field_config.toml [path_field_config2.toml ...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Python < 3.11 requires tomli: pip install tomli")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analysator as alr

from rhybrid_configparser import RhybridConfigParser
from simrun import RhybridRun
from trajectory import Trajectory


# ---------------------------------------------------------------------------
# Variable descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldVariable:
    """One quantity to sample and plot along the trajectory.

    Attributes
    ----------
    name:
        Variable name as it appears in the VLSV file.
    components:
        Which components to plot.  ``"all"`` → scalar or all 3 + magnitude;
        ``"magnitude"`` → only |v|; or a list such as ``["x", "y", "z"]``.
    label:
        Y-axis / legend label (LaTeX accepted).
    unit:
        Divisor applied before plotting (e.g. ``1e-9`` for nT).
    unit_str:
        Unit string appended to the axis label (e.g. ``"nT"``).
    """

    name:       str
    components: str | list[str] = "all"
    label:      str = ""
    unit:       float = 1.0
    unit_str:   str = ""

    @classmethod
    def from_dict(cls, d: dict, index: int) -> "FieldVariable":
        required = {"name"}
        missing = required - d.keys()
        if missing:
            raise ValueError(
                f"[[variable]] block {index} is missing keys: {sorted(missing)}"
            )
        return cls(
            name=d["name"],
            components=d.get("components", "all"),
            label=d.get("label", d["name"]),
            unit=float(d.get("unit", 1.0)),
            unit_str=d.get("unit_str", ""),
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PathFieldConfig:
    """All settings for one field-along-path job.

    Constructed directly or via :meth:`from_toml`.
    """

    # --- I/O ---
    output_dir:  str = "./figures/"
    fig_prefix:  str = "field_along_path"
    save_csv:    bool = True    # also write interpolated values as CSV

    # --- snapshot ---
    step: int = 0

    # --- figure ---
    fig_dpi:        int = 100
    fig_size:       tuple[int, int] = (10, 8)
    show_3d_plot:   bool = True     # render the 3-D domain + trajectory panel

    # --- aggregated objects ---
    run:       RhybridRun   = field(default=None)
    variables: list[FieldVariable] = field(default_factory=list)
    trajectory: Trajectory  = field(default=None)

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "PathFieldConfig":
        """Build a ``PathFieldConfig`` from a TOML file."""
        toml_path = Path(toml_path).resolve()
        with open(toml_path, "rb") as fh:
            raw = tomllib.load(fh)

        hdr = raw.get("header", {})
        _validate_path_field_header(hdr, toml_path)

        run_cfg = RhybridConfigParser()
        with open(Path(hdr["runConfig"]).resolve()) as f:
            run_cfg.read_file(f)

        run = RhybridRun(run_cfg, hdr["runFolder"], hdr.get("runDescr", ""))

        variables = [
            FieldVariable.from_dict(d, i)
            for i, d in enumerate(raw.get("variable", []))
        ]
        if not variables:
            raise ValueError(f"No [[variable]] blocks found in {toml_path}")

        trajectory = _build_trajectory(raw.get("path", {}), float(run.config_params["r_object"]))

        return cls(
            output_dir=hdr.get("outputFolder", "./figures/"),
            fig_prefix=hdr.get("figPrefix", "field_along_path"),
            save_csv=bool(hdr.get("saveCsv", True)),
            step=int(hdr.get("step", 0)),
            fig_dpi=int(hdr.get("figDpi", 100)),
            fig_size=tuple(hdr.get("figSize", [10, 8])),
            show_3d_plot=bool(hdr.get("show3dPlot", True)),
            run=run,
            variables=variables,
            trajectory=trajectory,
        )


def _validate_path_field_header(hdr: dict, path: Path) -> None:
    for key in ("runConfig", "runFolder"):
        if key not in hdr:
            raise ValueError(f"[header] is missing required key '{key}' in {path}")
    if not Path(hdr["runConfig"]).exists():
        raise FileNotFoundError(f"runConfig not found: {hdr['runConfig']}")
    if not Path(hdr["runFolder"]).is_dir():
        raise FileNotFoundError(f"runFolder not found: {hdr['runFolder']}")


def _build_trajectory(path_cfg: dict, r_object: float) -> Trajectory:
    """Build a :class:`Trajectory` from the ``[path]`` TOML block."""
    kind = path_cfg.get("kind", "orbit")

    if kind == "orbit":
        radius_rp = float(path_cfg.get("radius_rp", 1.5))
        n_points  = int(path_cfg.get("n_points", 100))
        plane     = path_cfg.get("plane", "xy")
        return Trajectory.from_orbit(
            radius=radius_rp * r_object,
            n_points=n_points,
            plane=plane,
            label=path_cfg.get("label", f"orbit_{radius_rp}Rp_{plane}"),
        )

    if kind == "line":
        start = [v * r_object for v in path_cfg["start_rp"]]
        end   = [v * r_object for v in path_cfg["end_rp"]]
        return Trajectory.from_line(
            start=start, end=end,
            n_points=int(path_cfg.get("n_points", 100)),
            label=path_cfg.get("label", "line"),
        )

    if kind == "csv":
        csv_path = Path(path_cfg["file"]).resolve()
        return Trajectory.from_csv(csv_path, label=path_cfg.get("label", None))

    raise ValueError(f"[path] kind must be 'orbit', 'line', or 'csv', got {kind!r}")


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def interpolate_fields_along_trajectory(
    vr,
    trajectory: Trajectory,
    var_names:  list[str],
    linear:     bool = True,
) -> np.ndarray:
    """Return interpolated field values along *trajectory*.

    Parameters
    ----------
    vr:
        Open :class:`analysator.vlsvfile.VlsvReader`.
    trajectory:
        The path to sample along.
    var_names:
        List of VLSV variable names.
    linear:
        ``True`` for bi/tri-linear interpolation; ``False`` for nearest cell.

    Returns
    -------
    np.ndarray
        Shape ``(N, total_components)`` where ``total_components`` is the sum
        of the vector sizes of all requested variables.
    """
    order = 1 if linear else 0
    _, _, values, _ = alr.calculations.vlsv_intpol_points(
        vr, trajectory.coords, var_names, interpolation_order=order
    )
    return values


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class PathFieldPlotter:
    """Interpolate and plot field values along a trajectory in one snapshot.

    Parameters
    ----------
    cfg:
        Fully populated :class:`PathFieldConfig`.
    """

    def __init__(self, cfg: PathFieldConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Interpolate fields and save all output (figures + optional CSV)."""
        cfg  = self._cfg
        vr   = cfg.run.run_out_files[cfg.step]
        traj = cfg.trajectory

        var_names = [v.name for v in cfg.variables]
        _check_variables(vr, var_names)

        values = interpolate_fields_along_trajectory(vr, traj, var_names)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        if cfg.save_csv:
            self._save_csv(values, var_names, vr)

        self._plot_time_series(values, var_names, vr)

        if cfg.show_3d_plot:
            self._plot_3d(vr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _col_slices(self, vr) -> list[tuple[FieldVariable, slice, list[str]]]:
        """Return ``(FieldVariable, column_slice, component_labels)`` triples."""
        result = []
        col = 0
        for fv in self._cfg.variables:
            dim = vr.read_variable_vectorsize(fv.name)
            col_slice = slice(col, col + dim)
            col += dim
            if dim == 1:
                comp_labels = [fv.label or fv.name]
            else:
                comp_labels = [f"{fv.label or fv.name} {c}" for c in ("x", "y", "z")]
                comp_labels.append(f"|{fv.label or fv.name}|")
            result.append((fv, col_slice, comp_labels))
        return result

    def _save_csv(self, values: np.ndarray, var_names: list[str], vr) -> None:
        cfg  = self._cfg
        traj = cfg.traj if hasattr(cfg, "traj") else cfg.trajectory

        header_parts = ["param", "x_m", "y_m", "z_m"]
        col_data = [traj.param, traj.coords[:, 0], traj.coords[:, 1], traj.coords[:, 2]]

        col = 0
        for fv in cfg.variables:
            dim = vr.read_variable_vectorsize(fv.name)
            if dim == 1:
                header_parts.append(fv.name)
                col_data.append(values[:, col])
            else:
                for i, c in enumerate(("x", "y", "z")):
                    header_parts.append(f"{fv.name}_{c}")
                    col_data.append(values[:, col + i])
                mag = np.sqrt((values[:, col:col + 3] ** 2).sum(axis=1))
                header_parts.append(f"|{fv.name}|")
                col_data.append(mag)
            col += dim

        out = np.column_stack(col_data)
        csv_path = Path(cfg.output_dir) / f"{cfg.fig_prefix}_{cfg.trajectory.label}.csv"
        np.savetxt(csv_path, out, delimiter=",",
                   header=",".join(header_parts), comments="")
        print(f"Saved interpolated data → {csv_path}")

    def _plot_time_series(self, values: np.ndarray, var_names: list[str], vr) -> None:
        cfg  = self._cfg
        traj = cfg.trajectory
        col_info = self._col_slices(vr)
        n_panels = len(col_info)

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=cfg.fig_size,
            sharex=True,
            squeeze=False,
        )

        for ax_row, (fv, sl, comp_labels) in zip(axes, col_info):
            ax = ax_row[0]
            block = values[:, sl] / fv.unit
            dim = sl.stop - sl.start

            if dim == 1:
                ax.plot(traj.param, block[:, 0], label=comp_labels[0])
            else:
                colours = ("tab:red", "tab:green", "tab:blue", "tab:gray")
                for i, (lbl, clr) in enumerate(zip(comp_labels[:3], colours[:3])):
                    ax.plot(traj.param, block[:, i], color=clr, label=lbl)
                mag = np.sqrt((block ** 2).sum(axis=1))
                ax.plot(traj.param, mag, color=colours[3], lw=1.5,
                        label=comp_labels[3])
                ax.plot(traj.param, -mag, color=colours[3], lw=1.5, ls="--")

            unit_label = f" [{fv.unit_str}]" if fv.unit_str else ""
            ax.set_ylabel(f"{fv.label or fv.name}{unit_label}")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, lw=0.4, alpha=0.5)

        axes[-1][0].set_xlabel("Curve parameter")
        axes[0][0].set_title(f"Fields along trajectory '{traj.label}'  (step {cfg.step})")
        fig.tight_layout()

        out_path = Path(cfg.output_dir) / f"{cfg.fig_prefix}_{traj.label}_step{cfg.step:08d}.png"
        fig.savefig(out_path, dpi=cfg.fig_dpi)
        plt.close(fig)
        print(f"Saved figure → {out_path}")

    def _plot_3d(self, vr) -> None:
        """Render the simulation domain box + trajectory in 3D."""
        cfg  = self._cfg
        traj = cfg.trajectory
        rp   = float(cfg.run.config_params["r_object"])
        dom  = cfg.run.config_params["domain"]

        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(projection="3d")

        # Simulation box
        _draw_box_3d(ax, dom, rp)

        # Planet sphere (unit radius = 1 Rp)
        _draw_planet_3d(ax)

        # Trajectory
        ax.plot(
            traj.coords[:, 0] / rp,
            traj.coords[:, 1] / rp,
            traj.coords[:, 2] / rp,
            ".b", ms=2, label=traj.label,
        )

        ax.set_xlabel("x [$R_p$]")
        ax.set_ylabel("y [$R_p$]")
        ax.set_zlabel("z [$R_p$]")
        ax.set_title(f"Trajectory '{traj.label}'")
        ax.set_box_aspect([1, 1, 1])

        out_path = Path(cfg.output_dir) / f"{cfg.fig_prefix}_{traj.label}_3d.png"
        fig.savefig(out_path, dpi=cfg.fig_dpi)
        plt.close(fig)
        print(f"Saved 3-D figure → {out_path}")


# ---------------------------------------------------------------------------
# Shared 3-D plot helpers (also imported by particle_tools)
# ---------------------------------------------------------------------------

def _draw_planet_3d(ax) -> None:
    """Overlay a grey/white sphere of radius 1 (in Rp) on *ax*."""
    from matplotlib import cm, colors as mcolors
    u, v = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 40))
    xs = np.sin(u) * np.cos(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(u)
    cs = np.where(xs > 0, 0.0, 1.0)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    ax.plot_surface(xs, ys, zs, cmap=cm.Greys, facecolors=cm.Greys(norm(cs)),
                    shade=True, alpha=0.6)


def _draw_box_3d(ax, domain: dict, r_object: float) -> None:
    """Draw the simulation domain bounding box on *ax* (coordinates in Rp)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    xlo, xhi = domain["x_min"] / r_object, domain["x_max"] / r_object
    ylo, yhi = domain["y_min"] / r_object, domain["y_max"] / r_object
    zlo, zhi = domain["z_min"] / r_object, domain["z_max"] / r_object

    corners = np.array([[xlo, ylo, zlo], [xhi, ylo, zlo],
                         [xhi, yhi, zlo], [xlo, yhi, zlo],
                         [xlo, ylo, zhi], [xhi, ylo, zhi],
                         [xhi, yhi, zhi], [xlo, yhi, zhi]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    segs = [[corners[i], corners[j]] for i, j in edges]
    ax.add_collection(Line3DCollection(segs, colors="gray", lw=0.5, alpha=0.4))

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _check_variables(vr, var_names: list[str]) -> None:
    missing = [v for v in var_names if not vr.check_variable(v)]
    if missing:
        available = vr.get_all_variables()
        raise ValueError(
            f"The following variables were not found in {vr.file_name}:\n"
            f"  missing : {missing}\n"
            f"  available: {available}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(toml_paths: list[str]) -> None:
    for p in toml_paths:
        cfg     = PathFieldConfig.from_toml(p)
        plotter = PathFieldPlotter(cfg)
        plotter.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interpolate and plot field values along a trajectory."
    )
    parser.add_argument("configs", nargs="+", metavar="TOML",
                        help="One or more TOML configuration files.")
    args = parser.parse_args()
    main(args.configs)
