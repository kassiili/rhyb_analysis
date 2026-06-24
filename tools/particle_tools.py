"""particle_tools.py
Propagate test particles through the electromagnetic fields of a single
RHybrid simulation snapshot using the Boris–Buneman algorithm, and write
the resulting trajectories to CSV.

The saved CSV is readable by :func:`trajectory.particle_trajectory_reader`,
which yields one :class:`~trajectory.Trajectory` per particle.  Those objects
can be fed directly into :class:`~field_along_path.PathFieldPlotter` or
:class:`~field_time_series.FieldTimeSeriesWriter` to sample any field variable
along the reconstructed paths.

Public API
----------
``ParticleSpec``
    Physical description of one particle species (mass, charge, obstacle
    radius).

``ParticleEnsemble``
    A collection of initial positions and velocities.  Built from a TOML
    ``[ensemble]`` block or programmatically.

``ParticlePropagatorConfig``
    All settings for one propagation job, loaded from a TOML via
    ``ParticlePropagatorConfig.from_toml``.

``ParticlePropagator``
    Runs the Boris–Buneman loop, enforces boundary conditions, and writes
    output.

``ParticlePlotter``
    Renders time-series and 3-D trajectory figures from a saved CSV.

TOML layout
-----------
See ``particle_example.toml`` for a complete example.

Usage (CLI)
-----------
    # propagate
    python particle_tools.py propagate particle_config.toml

    # plot from a previously saved CSV
    python particle_tools.py plot particle_config.toml trajectories.csv
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
from trajectory import Trajectory, particle_trajectory_reader
from field_along_path import _draw_planet_3d, _draw_box_3d, _check_variables


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_MASS_PROTON      = 1.672_621_716e-27   # kg
_CHARGE_ELEMENTARY = 1.602_176_53e-19   # C


# ---------------------------------------------------------------------------
# ParticleSpec – species description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParticleSpec:
    """Physical description of one test-particle species.

    Attributes
    ----------
    mass:
        Particle mass [kg].  Convenience aliases ``"proton"`` and ``"oxygen"``
        are accepted in the TOML (resolved in :meth:`from_dict`).
    charge:
        Particle charge [C].
    label:
        Species label used in file names and plot legends.
    """

    mass:   float
    charge: float
    label:  str = "particle"

    @classmethod
    def from_dict(cls, d: dict) -> "ParticleSpec":
        _MASS_ALIASES = {
            "proton":  _MASS_PROTON,
            "oxygen":  15.883_821_896_403 * _MASS_PROTON,
            "alpha":   3.973_7 * _MASS_PROTON,
            "electron": 9.109_383_7015e-31,
        }
        raw_mass = d.get("mass", "proton")
        mass = _MASS_ALIASES[raw_mass] if isinstance(raw_mass, str) else float(raw_mass)
        charge_multiplier = float(d.get("charge", 1.0))
        return cls(
            mass=mass,
            charge=charge_multiplier * _CHARGE_ELEMENTARY,
            label=d.get("label", "particle"),
        )


# ---------------------------------------------------------------------------
# ParticleEnsemble – initial conditions
# ---------------------------------------------------------------------------

@dataclass
class ParticleEnsemble:
    """Initial positions and velocities of a set of test particles.

    Attributes
    ----------
    positions:
        Shape ``(N, 3)`` array of initial positions [m].
    velocities:
        Shape ``(N, 3)`` array of initial velocities [m/s].
    """

    positions:  np.ndarray   # (N, 3)  [m]
    velocities: np.ndarray   # (N, 3)  [m/s]

    def __post_init__(self) -> None:
        self.positions  = np.asarray(self.positions,  dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must be shape (N, 3)")

    @property
    def n_particles(self) -> int:
        return len(self.positions)

    @classmethod
    def from_dict(cls, d: dict, domain: dict, r_object: float,
                  rng: np.random.Generator) -> "ParticleEnsemble":
        """Build from a TOML ``[ensemble]`` block.

        Supports two layouts:

        * ``kind = "line_upstream"`` – particles distributed along a line on
          the upstream face (x = x_max), spanning y uniformly.
        * ``kind = "csv"``           – read initial conditions from a CSV file
          with columns ``x, y, z, vx, vy, vz`` (all in SI).

        The bulk velocity is taken from ``bulk_velocity_ms`` (list of 3) and a
        Gaussian thermal scatter of std ``thermal_velocity_ms`` is added.
        """
        kind = d.get("kind", "line_upstream")
        n    = int(d.get("n_particles", 10))
        seed = d.get("seed", None)

        if seed is not None:
            rng = np.random.default_rng(int(seed))

        bulk = np.array(d.get("bulk_velocity_ms", [-430e3, 0.0, 0.0]), dtype=float)
        vth  = float(d.get("thermal_velocity_ms", 30e3))

        if kind == "line_upstream":
            dx   = (domain["x_max"] - domain["x_min"]) / domain["x_size"]
            x0   = np.full(n, domain["x_max"] - 1.5 * dx)
            y0   = np.linspace(domain["y_min"] + 1.5 * dx,
                               domain["y_max"] - 1.5 * dx, n)
            z0   = np.zeros(n)
            pos  = np.column_stack([x0, y0, z0])

        elif kind == "csv":
            arr  = np.loadtxt(d["file"], delimiter=",")
            pos  = arr[:, :3]
            vel  = arr[:, 3:6]
            return cls(positions=pos, velocities=vel)

        else:
            raise ValueError(
                f"ensemble.kind must be 'line_upstream' or 'csv', got {kind!r}"
            )

        vel = bulk[None, :] + vth * rng.standard_normal((n, 3))
        return cls(positions=pos, velocities=vel)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParticlePropagatorConfig:
    """All settings for one test-particle propagation job."""

    # --- I/O ---
    output_dir:     str  = "./figures/"
    fig_prefix:     str  = "particles"
    save_figure:    bool = True

    # --- snapshot ---
    step: int = 0

    # --- field variable names in the VLSV file ---
    var_B:  str = "cellB"
    var_Ue: str = "cellUe"
    var_Ep: str = "cellEp"

    # --- integrator ---
    dt_s:           float = 0.5
    max_timesteps:  int   = 500
    linear_interp:  bool  = True

    # --- figure ---
    fig_dpi:  int             = 100
    fig_size: tuple[int, int] = (10, 12)

    # --- aggregated objects ---
    run:      RhybridRun      = field(default=None)
    species:  ParticleSpec    = field(default_factory=lambda: ParticleSpec(
                                    mass=_MASS_PROTON,
                                    charge=_CHARGE_ELEMENTARY))
    ensemble: ParticleEnsemble = field(default=None)

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "ParticlePropagatorConfig":
        toml_path = Path(toml_path).resolve()
        with open(toml_path, "rb") as fh:
            raw = tomllib.load(fh)

        hdr = raw.get("header", {})
        _validate_header(hdr, toml_path)

        run_cfg = RhybridConfigParser()
        with open(Path(hdr["runConfig"]).resolve()) as f:
            run_cfg.read_file(f)

        run = RhybridRun(run_cfg, hdr["runFolder"], hdr.get("runDescr", ""))

        species  = ParticleSpec.from_dict(raw.get("species", {}))
        rng      = np.random.default_rng(raw.get("ensemble", {}).get("seed", None))
        ensemble = ParticleEnsemble.from_dict(
            raw.get("ensemble", {}),
            run.config_params["domain"],
            float(run.config_params["r_object"]),
            rng,
        )

        return cls(
            output_dir=hdr.get("outputFolder", "./figures/"),
            fig_prefix=hdr.get("figPrefix", "particles"),
            save_figure=bool(hdr.get("saveFigure", True)),
            step=int(hdr.get("step", 0)),
            var_B=hdr.get("varB", "cellB"),
            var_Ue=hdr.get("varUe", "cellUe"),
            var_Ep=hdr.get("varEp", "cellEp"),
            dt_s=float(hdr.get("dt_s", 0.5)),
            max_timesteps=int(hdr.get("maxTimesteps", 500)),
            linear_interp=bool(hdr.get("linearInterp", True)),
            fig_dpi=int(hdr.get("figDpi", 100)),
            fig_size=tuple(hdr.get("figSize", [10, 12])),
            run=run,
            species=species,
            ensemble=ensemble,
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
# Boris–Buneman integrator  (pure function, no class state)
# ---------------------------------------------------------------------------

def boris_buneman_step(
    r:  np.ndarray,   # (M, 3)  positions
    v:  np.ndarray,   # (M, 3)  velocities
    E:  np.ndarray,   # (M, 3)  electric field at r
    B:  np.ndarray,   # (M, 3)  magnetic field at r
    m:  float,
    q:  float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Boris–Buneman push.  Returns updated ``(r, v)``."""
    # Half-step position push
    r = r + dt * v

    # Electric half-kick
    qm_half = 0.5 * q * dt / m
    dv  = qm_half * E
    t   = qm_half * B
    t2  = np.sum(t * t, axis=1, keepdims=True)
    b2  = 2.0 / (1.0 + t2)
    s   = b2 * t
    vm  = v + dv
    v0  = vm + np.cross(vm, t)
    vp  = vm + np.cross(v0, s)
    v   = vp + dv

    return r, v


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class ParticlePropagator:
    """Run the Boris–Buneman loop and persist results.

    Parameters
    ----------
    cfg:
        Fully populated :class:`ParticlePropagatorConfig`.
    """

    def __init__(self, cfg: ParticlePropagatorConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Propagate particles and write trajectories to CSV.

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        cfg  = self._cfg
        run  = cfg.run
        vr   = run.run_out_files[cfg.step]

        _check_variables(vr, [cfg.var_B, cfg.var_Ue, cfg.var_Ep])

        dom   = run.config_params["domain"]
        rp    = float(run.config_params["r_object"])
        obs_r = rp + 200e3      # default obstacle radius: R_object + 200 km
        # Allow override from TOML via config_params if present
        if "R_particleObstacle" in run._run_config.get("Hybrid", {}):
            obs_r = float(run._run_config["Hybrid"]["R_particleObstacle"])

        order = 1 if cfg.linear_interp else 0
        sp    = cfg.species

        # Working copies of all particle states
        n    = cfg.ensemble.n_particles
        rnow = cfg.ensemble.positions.copy()
        vnow = cfg.ensemble.velocities.copy()
        tnow = np.zeros(n)

        # Storage: list-of-arrays, one per particle
        r_hist = [rnow[i:i+1].copy() for i in range(n)]
        v_hist = [vnow[i:i+1].copy() for i in range(n)]
        t_hist = [np.array([0.0])     for i in range(n)]

        # Active particle index array
        active = np.arange(n)

        box_eps  = 0.1 * (dom["x_max"] - dom["x_min"]) / dom["x_size"]
        obs_r2   = obs_r ** 2

        bb_steps_total = 0
        for ii in range(cfg.max_timesteps):
            if len(active) == 0:
                break

            # --- boundary checks ---
            outer = (
                (rnow[:, 0] < dom["x_min"] + box_eps) |
                (rnow[:, 0] > dom["x_max"] - box_eps) |
                (rnow[:, 1] < dom["y_min"] + box_eps) |
                (rnow[:, 1] > dom["y_max"] - box_eps) |
                (rnow[:, 2] < dom["z_min"] + box_eps) |
                (rnow[:, 2] > dom["z_max"] - box_eps)
            )
            inner = np.sum(rnow ** 2, axis=1) <= obs_r2
            absorbed = outer | inner

            if absorbed.any():
                keep  = ~absorbed
                rnow  = rnow[keep]
                vnow  = vnow[keep]
                tnow  = tnow[keep]
                active = active[keep]
                if len(active) == 0:
                    break

            # --- field interpolation ---
            var_list = [cfg.var_B, cfg.var_Ue, cfg.var_Ep]
            _, _, fields, _ = alr.calculations.vlsv_intpol_points(
                vr, rnow, var_list, interpolation_order=order
            )
            B_now  = fields[:, 0:3]
            Ue_now = fields[:, 3:6]
            Ep_now = fields[:, 6:9]
            E_now  = -np.cross(Ue_now, B_now) + Ep_now

            # --- Boris–Buneman push ---
            rnow, vnow = boris_buneman_step(rnow, vnow, E_now, B_now,
                                            sp.mass, sp.charge, cfg.dt_s)
            tnow += cfg.dt_s
            bb_steps_total += len(active)

            # --- store ---
            for local_i, global_i in enumerate(active):
                r_hist[global_i] = np.vstack([r_hist[global_i], rnow[local_i]])
                v_hist[global_i] = np.vstack([v_hist[global_i], vnow[local_i]])
                t_hist[global_i] = np.append(t_hist[global_i], tnow[local_i])

        print(f"Propagation finished: {ii + 1} timesteps, "
              f"{bb_steps_total} total particle–step evaluations.")

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = self._save_trajectories(r_hist, v_hist, t_hist)

        if cfg.save_figure:
            plotter = ParticlePlotter(cfg)
            plotter.plot_time_series(r_hist, v_hist, t_hist)
            plotter.plot_3d(r_hist)

        return csv_path

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save_trajectories(
        self,
        r_hist: list[np.ndarray],
        v_hist: list[np.ndarray],
        t_hist: list[np.ndarray],
    ) -> Path:
        """Write all particle trajectories to a single CSV.

        Columns: ``particle_id, param, x, y, z, vx, vy, vz``

        This format is understood by :func:`trajectory.particle_trajectory_reader`.
        """
        cfg = self._cfg
        csv_path = Path(cfg.output_dir) / f"{cfg.fig_prefix}_trajectories.csv"

        with open(csv_path, "w") as fh:
            fh.write("particle_id,param,x,y,z,vx,vy,vz\n")
            for pid, (r, v, t) in enumerate(zip(r_hist, v_hist, t_hist)):
                for k in range(len(t)):
                    row = [pid, t[k],
                           r[k, 0], r[k, 1], r[k, 2],
                           v[k, 0], v[k, 1], v[k, 2]]
                    fh.write(",".join(f"{x:.6e}" if isinstance(x, float) else str(x)
                                      for x in row) + "\n")

        print(f"Saved trajectories → {csv_path}")
        return csv_path


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class ParticlePlotter:
    """Render figures for test-particle trajectories.

    Can be used standalone (reading a CSV) or called by
    :class:`ParticlePropagator` immediately after propagation.
    """

    def __init__(self, cfg: ParticlePropagatorConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # From CSV  (standalone use)
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, cfg: ParticlePropagatorConfig, csv_path: str | Path
                 ) -> tuple["ParticlePlotter", list, list, list]:
        """Read trajectories from *csv_path* and return plotter + history lists."""
        r_hist, v_hist, t_hist = [], [], []
        for traj in particle_trajectory_reader(csv_path):
            t_hist.append(traj.param)
            r_hist.append(traj.coords)
            # velocities are stored per-particle; read directly from CSV
            v_hist.append(np.zeros_like(traj.coords))   # placeholder if not needed

        # Re-read velocities properly
        import csv as _csv
        rows: dict[int, list] = {}
        with open(csv_path, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                pid = int(row["particle_id"])
                rows.setdefault(pid, []).append(
                    [float(row["vx"]), float(row["vy"]), float(row["vz"])]
                )
        v_hist = [np.array(rows[i]) for i in sorted(rows)]

        plotter = cls(cfg)
        return plotter, r_hist, v_hist, t_hist

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot_time_series(
        self,
        r_hist: list[np.ndarray],
        v_hist: list[np.ndarray],
        t_hist: list[np.ndarray],
    ) -> None:
        """Six-panel time-series figure: x, y, z, vx, vy, vz vs t."""
        cfg = self._cfg
        rp  = float(cfg.run.config_params["r_object"])
        n_panels = 6
        labels_r = ("$x$ [$R_p$]", "$y$ [$R_p$]", "$z$ [$R_p$]")
        labels_v = ("$v_x$ [km/s]", "$v_y$ [km/s]", "$v_z$ [km/s]")

        fig, axes = plt.subplots(n_panels, 1, figsize=cfg.fig_size,
                                 sharex=True, squeeze=True)

        for pid, (r, v, t) in enumerate(zip(r_hist, v_hist, t_hist)):
            for i in range(3):
                axes[i].plot(t, r[:, i] / rp)
                axes[3 + i].plot(t, v[:, i] / 1e3)

        for i, lbl in enumerate(labels_r):
            axes[i].set_ylabel(lbl)
            axes[i].grid(True, lw=0.4, alpha=0.5)
        for i, lbl in enumerate(labels_v):
            axes[3 + i].set_ylabel(lbl)
            axes[3 + i].grid(True, lw=0.4, alpha=0.5)

        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title(
            f"Test-particle trajectories  –  {cfg.species.label}  (step {cfg.step})"
        )
        fig.tight_layout()

        out = Path(cfg.output_dir) / f"{cfg.fig_prefix}_time_series.png"
        fig.savefig(out, dpi=cfg.fig_dpi)
        plt.close(fig)
        print(f"Saved time-series figure → {out}")

    def plot_3d(self, r_hist: list[np.ndarray]) -> None:
        """3-D trajectory figure with planet sphere and domain box."""
        cfg = self._cfg
        rp  = float(cfg.run.config_params["r_object"])
        dom = cfg.run.config_params["domain"]

        fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(projection="3d")

        _draw_box_3d(ax, dom, rp)
        _draw_planet_3d(ax)

        for r in r_hist:
            ax.plot(r[:, 0] / rp, r[:, 1] / rp, r[:, 2] / rp, lw=0.8)

        ax.set_xlabel("$x$ [$R_p$]")
        ax.set_ylabel("$y$ [$R_p$]")
        ax.set_zlabel("$z$ [$R_p$]")
        ax.set_title(
            f"Test-particle trajectories  –  {cfg.species.label}  (step {cfg.step})"
        )
        ax.set_box_aspect([1, 1, 1])

        out = Path(cfg.output_dir) / f"{cfg.fig_prefix}_trajectories_3d.png"
        fig.savefig(out, dpi=cfg.fig_dpi)
        plt.close(fig)
        print(f"Saved 3-D trajectory figure → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(mode: str, toml_path: str, csv_path: Optional[str] = None) -> None:
    cfg = ParticlePropagatorConfig.from_toml(toml_path)

    if mode == "propagate":
        propagator = ParticlePropagator(cfg)
        propagator.run()

    elif mode == "plot":
        if csv_path is None:
            raise ValueError("'plot' mode requires a CSV path argument.")
        plotter, r_hist, v_hist, t_hist = ParticlePlotter.from_csv(cfg, csv_path)
        plotter.plot_time_series(r_hist, v_hist, t_hist)
        plotter.plot_3d(r_hist)

    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'propagate' or 'plot'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Propagate test particles or plot saved trajectories."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_prop = sub.add_parser("propagate", help="Run Boris–Buneman propagation.")
    p_prop.add_argument("config", metavar="TOML")

    p_plot = sub.add_parser("plot", help="Plot trajectories from a saved CSV.")
    p_plot.add_argument("config", metavar="TOML")
    p_plot.add_argument("csv",    metavar="CSV")

    args = parser.parse_args()
    if args.mode == "propagate":
        main("propagate", args.config)
    else:
        main("plot", args.config, args.csv)
