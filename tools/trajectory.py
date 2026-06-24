"""trajectory.py
Defines the :class:`Trajectory` dataclass – a parametric curve through the
simulation domain – and factory functions / generators that produce one.

A ``Trajectory`` is the shared data structure consumed by:

* ``field_along_path``  – interpolate field values at each waypoint of a
  spatial curve (single snapshot)
* ``field_time_series`` – interpolate field values at one or several fixed
  points (or along a curve) through all snapshots
* ``particle_tools``    – read back test-particle paths saved by
  :func:`particle_tools.save_trajectories` as ``Trajectory`` objects

Structure
---------
coords : (N, 3) float array   – x, y, z positions [m]
param  : (N,)   float array   – curve parameter (time [s], arc-length, phase …)
label  : str                  – human-readable tag used in file names / legends

Iteration
---------
Iterating a ``Trajectory`` yields ``(param_i, coord_i)`` tuples, where
``coord_i`` is a length-3 array.  This makes it easy to loop over waypoints::

    for t, r in traj:
        ...

Factories
---------
``Trajectory.from_csv``         – read coords + param from a two-column CSV
``Trajectory.from_orbit``       – circular orbit at a given radius (convenience)
``Trajectory.from_line``        – straight line between two points
``particle_trajectory_reader``  – generator that yields one ``Trajectory`` per
                                  particle from a CSV written by
                                  :func:`particle_tools.save_trajectories`
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A parametric curve through the simulation domain.

    Parameters
    ----------
    coords:
        Shape ``(N, 3)`` array of (x, y, z) positions in metres.
    param:
        Shape ``(N,)`` array of the curve parameter (physical time in seconds
        for particle tracks; arbitrary for spatial paths).
    label:
        Short identifier used in file names and plot legends.
    """

    coords: np.ndarray          # (N, 3) float64  [m]
    param:  np.ndarray          # (N,)   float64
    label:  str = "trajectory"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        self.param  = np.asarray(self.param,  dtype=float)

        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(
                f"Trajectory.coords must be shape (N, 3), got {self.coords.shape}"
            )
        if self.param.shape != (len(self.coords),):
            raise ValueError(
                f"Trajectory.param length {len(self.param)} does not match "
                f"coords length {len(self.coords)}"
            )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield ``(param_i, coord_i)`` pairs."""
        return zip(self.param, self.coords)

    def __len__(self) -> int:
        return len(self.param)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Save to a CSV file.

        Columns: ``param, x, y, z``.  The ``label`` is written as a comment
        in the first line so that :meth:`from_csv` can reconstruct it.
        """
        path = Path(path)
        header = f"# label={self.label}\nparam,x,y,z"
        data = np.column_stack([self.param, self.coords])
        np.savetxt(path, data, delimiter=",", header=header, comments="")

    @classmethod
    def from_csv(cls, path: str | Path, label: str | None = None) -> "Trajectory":
        """Read a ``Trajectory`` from a CSV file written by :meth:`to_csv`.

        The first line may be a ``# label=<name>`` comment; if so, *label* is
        inferred from it.  A *label* argument overrides the comment.
        """
        path = Path(path)
        inferred_label = path.stem      # fallback

        rows: list[list[str]] = []
        with open(path, newline="") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line.startswith("# label="):
                    inferred_label = line.split("=", 1)[1].strip()
                elif line.startswith("#") or line == "param,x,y,z":
                    continue        # skip header / other comments
                else:
                    rows.append(line.split(","))

        arr = np.array(rows, dtype=float)
        return cls(
            coords=arr[:, 1:4],
            param=arr[:, 0],
            label=label if label is not None else inferred_label,
        )

    # ------------------------------------------------------------------
    # Factories for common spatial curves
    # ------------------------------------------------------------------

    @classmethod
    def from_orbit(
        cls,
        radius: float,
        n_points: int = 100,
        plane: str = "xy",
        label: str = "orbit",
    ) -> "Trajectory":
        """Circular orbit of *radius* metres in the requested plane.

        Parameters
        ----------
        radius:
            Orbit radius [m].
        n_points:
            Number of waypoints.
        plane:
            One of ``"xy"``, ``"xz"``, ``"yz"``.
        label:
            Trajectory label.
        """
        phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        c, s = np.cos(phi), np.sin(phi)
        z = np.zeros(n_points)

        plane_map = {
            "xy": np.column_stack([radius * c, radius * s, z]),
            "xz": np.column_stack([radius * c, z, radius * s]),
            "yz": np.column_stack([z, radius * c, radius * s]),
        }
        if plane not in plane_map:
            raise ValueError(f"plane must be one of {list(plane_map)}, got {plane!r}")

        param = np.linspace(0.0, 1.0, n_points, endpoint=False)
        return cls(coords=plane_map[plane], param=param, label=label)

    @classmethod
    def from_line(
        cls,
        start: np.ndarray | list,
        end:   np.ndarray | list,
        n_points: int = 100,
        label: str = "line",
    ) -> "Trajectory":
        """Straight line from *start* to *end* (both in metres).

        The parameter is normalised arc-length ∈ [0, 1].
        """
        start = np.asarray(start, dtype=float)
        end   = np.asarray(end,   dtype=float)
        t     = np.linspace(0.0, 1.0, n_points)
        coords = start[None, :] + t[:, None] * (end - start)[None, :]
        return cls(coords=coords, param=t, label=label)


# ---------------------------------------------------------------------------
# Generator: iterate particle trajectories from a CSV
# ---------------------------------------------------------------------------

def particle_trajectory_reader(
    path: str | Path,
) -> Generator[Trajectory, None, None]:
    """Yield one :class:`Trajectory` per particle from a CSV produced by
    :func:`particle_tools.save_trajectories`.

    File format (written by ``save_trajectories``)::

        particle_id,param,x,y,z,vx,vy,vz
        0,0.0,x0,y0,z0,vx0,vy0,vz0
        0,0.5,...
        1,0.0,...

    The generator groups rows by ``particle_id`` and yields each group as a
    :class:`Trajectory` whose ``label`` is ``"particle_<id>"``.  Velocity
    columns (if present) are ignored; they are stored separately by
    :func:`particle_tools.save_trajectories`.
    """
    path = Path(path)

    def _rows() -> Generator[dict, None, None]:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            yield from reader

    for pid, group in itertools.groupby(_rows(), key=lambda r: r["particle_id"]):
        rows = list(group)
        param  = np.array([float(r["param"]) for r in rows])
        coords = np.array([[float(r["x"]), float(r["y"]), float(r["z"])]
                           for r in rows])
        yield Trajectory(coords=coords, param=param, label=f"particle_{pid}")
