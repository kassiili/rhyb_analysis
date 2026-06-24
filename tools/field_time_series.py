"""field_time_series.py
Build time-series of interpolated field values either at one or several
*fixed points* in the domain, or at waypoints of a
:class:`~trajectory.Trajectory` (a spatial curve).

In both cases the result is a plain CSV file and, optionally, a multi-panel
figure with one subplot per variable.

Public API
----------
``PointSpec``
    A single fixed (x, y, z) location and a human label.

``TimeSeriesConfig``
    All settings for one job, constructed from a TOML via
    ``TimeSeriesConfig.from_toml``.

``FieldTimeSeriesWriter``
    Does the actual work: validates inputs, loops over snapshots, writes the
    CSV, and optionally renders the figure.

TOML layout
-----------
See ``time_series_example.toml`` for a worked example.  Supported sampling
modes (set ``[sampling]  kind = ...``):

    kind = "points"     – sample at one or more fixed (x, y, z) points
    kind = "trajectory" – sample at the waypoints of a Trajectory loaded
                          from a CSV (useful for spacecraft trajectories)

Usage (CLI)
-----------
    python field_time_series.py config.toml [config2.toml ...]
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
from field_along_path import FieldVariable, _check_variables


# ---------------------------------------------------------------------------
# PointSpec – a named fixed location
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointSpec:
    """A single fixed spatial location.

    Attributes
    ----------
    x, y, z:
        Coordinates in metres.
    label:
        Short tag used in CSV column headers and plot legends.
    """

    x: float
    y: float
    z: float
    label: str = "point"

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_dict(cls, d: dict, r_object: float, index: int) -> "PointSpec":
        """Build from a TOML dict.

        Coordinates may be given in metres (``x``, ``y``, ``z``) or in units
        of the object radius (``x_rp``, ``y_rp``, ``z_rp``).
        """
        if "x_rp" in d:
            x = float(d["x_rp"]) * r_object
            y = float(d["y_rp"]) * r_object
            z = float(d["z_rp"]) * r_object
        elif "x" in d:
            x, y, z = float(d["x"]), float(d["y"]), float(d["z"])
        else:
            raise ValueError(
                f"[[point]] block {index}: provide either (x, y, z) in metres "
                "or (x_rp, y_rp, z_rp) in units of R_object."
            )
        return cls(x=x, y=y, z=z, label=d.get("label", f"p{index}"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesConfig:
    """All settings for one field-time-series job."""

    # --- I/O ---
    output_dir:  str  = "./figures/"
    fig_prefix:  str  = "time_series"
    save_figure: bool = True

    # --- time range ---
    t_start: int = 0
    t_end:   int = 1_000_000

    # --- figure ---
    fig_dpi:  int              = 100
    fig_size: tuple[int, int]  = (10, 8)

    # --- aggregated objects ---
    run:        RhybridRun          = field(default=None)
    variables:  list[FieldVariable] = field(default_factory=list)

    # Exactly one of the two below will be populated:
    points:     list[PointSpec]     = field(default_factory=list)
    trajectory: Optional[Trajectory] = None

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "TimeSeriesConfig":
        toml_path = Path(toml_path).resolve()
        with open(toml_path, "rb") as fh:
            raw = tomllib.load(fh)

        hdr = raw.get("header", {})
        _validate_header(hdr, toml_path)

        run_cfg = RhybridConfigParser()
        with open(Path(hdr["runConfig"]).resolve()) as f:
            run_cfg.read_file(f)

        run = RhybridRun(run_cfg, hdr["runFolder"], hdr.get("runDescr", ""))
        r_object = float(run.config_params["r_object"])

        variables = [
            FieldVariable.from_dict(d, i)
            for i, d in enumerate(raw.get("variable", []))
        ]
        if not variables:
            raise ValueError(f"No [[variable]] blocks found in {toml_path}")

        sampling = raw.get("sampling", {})
        kind = sampling.get("kind", "points")

        points: list[PointSpec] = []
        trajectory: Optional[Trajectory] = None

        if kind == "points":
            pts_raw = raw.get("point", [])
            if not pts_raw:
                raise ValueError(
                    f"sampling.kind = 'points' but no [[point]] blocks found in {toml_path}"
                )
            points = [PointSpec.from_dict(d, r_object, i) for i, d in enumerate(pts_raw)]

        elif kind == "trajectory":
            csv_file = sampling.get("file")
            if not csv_file:
                raise ValueError(
                    "sampling.kind = 'trajectory' requires a 'file' key pointing to a CSV."
                )
            trajectory = Trajectory.from_csv(csv_file)

        else:
            raise ValueError(f"sampling.kind must be 'points' or 'trajectory', got {kind!r}")

        return cls(
            output_dir=hdr.get("outputFolder", "./figures/"),
            fig_prefix=hdr.get("figPrefix", "time_series"),
            save_figure=bool(hdr.get("saveFigure", True)),
            t_start=int(hdr.get("tStartThisProcess", 0)),
            t_end=int(hdr.get("tEndThisProcess", 1_000_000)),
            fig_dpi=int(hdr.get("figDpi", 100)),
            fig_size=tuple(hdr.get("figSize", [10, 8])),
            run=run,
            variables=variables,
            points=points,
            trajectory=trajectory,
        )


def _validate_header(hdr: dict, path: Path) -> None:
    for key in ("runConfig", "runFolder"):
        if key not in hdr:
            raise ValueError(f"[header] missing key '{key}' in {path}")
    if not Path(hdr["runConfig"]).exists():
        raise FileNotFoundError(f"runConfig not found: {hdr['runConfig']}")
    if not Path(hdr["runFolder"]).is_dir():
        raise FileNotFoundError(f"runFolder not found: {hdr['runFolder']}")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class FieldTimeSeriesWriter:
    """Loop over snapshots, interpolate, write CSV, optionally plot.

    Parameters
    ----------
    cfg:
        Fully populated :class:`TimeSeriesConfig`.
    linear:
        Use linear (``True``) or nearest-cell (``False``) interpolation.
    """

    def __init__(self, cfg: TimeSeriesConfig, linear: bool = True) -> None:
        self._cfg    = cfg
        self._linear = linear
        self._order  = 1 if linear else 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Process all time steps and write outputs."""
        cfg  = self._cfg
        run  = cfg.run
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        steps = run.get_steps_in_range(cfg.t_start, cfg.t_end)
        if not steps:
            raise ValueError(
                f"No snapshots found in t ∈ [{cfg.t_start}, {cfg.t_end}]"
            )

        var_names = [v.name for v in cfg.variables]
        _check_variables(run.run_out_files[steps[0]], var_names)

        # Determine sampling points once (they do not change with time)
        coords, point_labels = self._resolve_coords()

        # Collect data: shape (n_steps, n_points, n_value_cols)
        times  = np.empty(len(steps))
        # We first collect raw interp matrices, then stack.
        all_values: list[np.ndarray] = []

        for i, step in enumerate(steps):
            vr = run.run_out_files[step]
            t  = vr.read_parameter("time")
            if t is None:
                t = vr.read_parameter("t")
            times[i] = float(t)

            _, _, values, _ = alr.calculations.vlsv_intpol_points(
                vr, coords, var_names, interpolation_order=self._order
            )
            all_values.append(values)   # (n_points, n_cols)
            print(f"  step {step:8d}  t = {times[i]:.2f} s")

        # Stack to (n_steps, n_points, n_cols)
        data_3d = np.stack(all_values, axis=0)

        csv_path = self._write_csv(times, steps, data_3d, point_labels, var_names,
                                   run.run_out_files[steps[0]])

        if cfg.save_figure:
            self._plot(times, data_3d, point_labels, run.run_out_files[steps[0]])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_coords(self) -> tuple[np.ndarray, list[str]]:
        """Return ``(coords, labels)`` depending on sampling mode."""
        cfg = self._cfg
        if cfg.trajectory is not None:
            return cfg.trajectory.coords, [f"wp{i}" for i in range(len(cfg.trajectory))]
        return (
            np.array([p.coord for p in cfg.points]),
            [p.label for p in cfg.points],
        )

    def _col_info(self, vr) -> list[tuple[FieldVariable, int, int]]:
        """List of (FieldVariable, col_start, col_end) for output columns."""
        info = []
        col = 0
        for fv in self._cfg.variables:
            dim = vr.read_variable_vectorsize(fv.name)
            info.append((fv, col, col + dim))
            col += dim
        return info

    def _write_csv(
        self,
        times:        np.ndarray,
        steps:        list[int],
        data_3d:      np.ndarray,
        point_labels: list[str],
        var_names:    list[str],
        vr,
    ) -> Path:
        """Write a flat CSV: one row per (step, point) combination.

        Columns: ``step, time_s, point, x_m, y_m, z_m, <var_cols...>``
        """
        cfg    = self._cfg
        coords, _ = self._resolve_coords()
        col_info  = self._col_info(vr)

        header_parts = ["step", "time_s", "point", "x_m", "y_m", "z_m"]
        for fv, cs, ce in col_info:
            dim = ce - cs
            if dim == 1:
                header_parts.append(fv.name)
            else:
                header_parts += [f"{fv.name}_{c}" for c in ("x", "y", "z")]
                header_parts.append(f"|{fv.name}|")

        rows = []
        for i, (step, t) in enumerate(zip(steps, times)):
            for j, (plabel, coord) in enumerate(zip(point_labels, coords)):
                row = [step, t, plabel, coord[0], coord[1], coord[2]]
                for fv, cs, ce in col_info:
                    dim = ce - cs
                    vals = data_3d[i, j, cs:ce] / fv.unit
                    row += list(vals)
                    if dim > 1:
                        row.append(float(np.sqrt((vals ** 2).sum())))
                rows.append(row)

        arr = np.array(rows, dtype=object)
        suffix = cfg.trajectory.label if cfg.trajectory is not None else "fixed_points"
        csv_path = Path(cfg.output_dir) / f"{cfg.fig_prefix}_{suffix}.csv"
        with open(csv_path, "w") as fh:
            fh.write(",".join(header_parts) + "\n")
            for row in rows:
                fh.write(",".join(str(v) for v in row) + "\n")

        print(f"Saved time-series data → {csv_path}")
        return csv_path

    def _plot(
        self,
        times:        np.ndarray,
        data_3d:      np.ndarray,
        point_labels: list[str],
        vr,
    ) -> None:
        """One figure per sampling point, one subplot per variable."""
        cfg      = self._cfg
        col_info = self._col_info(vr)
        n_panels = len(col_info)

        for j, plabel in enumerate(point_labels):
            fig, axes = plt.subplots(
                n_panels, 1,
                figsize=cfg.fig_size,
                sharex=True,
                squeeze=False,
            )
            for ax_row, (fv, cs, ce) in zip(axes, col_info):
                ax  = ax_row[0]
                dim = ce - cs
                block = data_3d[:, j, cs:ce] / fv.unit

                if dim == 1:
                    ax.plot(times, block[:, 0])
                else:
                    colours = ("tab:red", "tab:green", "tab:blue", "tab:gray")
                    comp_labels = ("x", "y", "z")
                    for k, (lbl, clr) in enumerate(zip(comp_labels, colours[:3])):
                        ax.plot(times, block[:, k], color=clr, label=lbl)
                    mag = np.sqrt((block ** 2).sum(axis=1))
                    ax.plot(times, mag, color=colours[3], lw=1.5, label="mag")
                    ax.plot(times, -mag, color=colours[3], lw=1.5, ls="--")
                    ax.legend(loc="upper right", fontsize=7)

                unit_str = f" [{fv.unit_str}]" if fv.unit_str else ""
                ax.set_ylabel(f"{fv.label or fv.name}{unit_str}")
                ax.grid(True, lw=0.4, alpha=0.5)

            axes[-1][0].set_xlabel("Time [s]")
            axes[0][0].set_title(f"Time series at '{plabel}'")
            fig.tight_layout()

            out_path = (
                Path(cfg.output_dir)
                / f"{cfg.fig_prefix}_{plabel}.png"
            )
            fig.savefig(out_path, dpi=cfg.fig_dpi)
            plt.close(fig)
            print(f"Saved time-series figure → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(toml_paths: list[str]) -> None:
    for p in toml_paths:
        cfg    = TimeSeriesConfig.from_toml(p)
        writer = FieldTimeSeriesWriter(cfg)
        writer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create field time-series at fixed points or along a trajectory."
    )
    parser.add_argument("configs", nargs="+", metavar="TOML",
                        help="One or more TOML configuration files.")
    args = parser.parse_args()
    main(args.configs)
