import itertools
import socket
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
import tomllib
import os

import matplotlib
from analysator.vlsvfile import VlsvReader

matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Wedge

from plot_parameters import PlotParams
from rhybrid_configparser import RhybridConfigParser
from simrun import RhybridRun, get_time_param
from vlsv_data_reducers import read_variable_data

plt.switch_backend('agg')
HN = '(hostname = ' + socket.gethostname() + ') '

# --------------------------------------------------------------------------------------------------------
# SlicePlotJobConfig & SlicePlotJob  — Utilities for parallelized slice plot jobs over a full simulation
# --------------------------------------------------------------------------------------------------------

@dataclass
class SlicePlotJobConfig:
    """All configuration for one complete slice-plot job.

    Constructed directly or via the SlicePlotJobConfig.from_toml() factory.
    Holds references to the parameter registry and the list of runs so
    that the plotting classes only need to receive a single object.
    """

    n_cores: int = 1

    # --- output ---
    output_dir: str = './png/'

    # --- figure style ---
    fig_dpi: int = 100
    fig_resolution: tuple[int, int] = (1920, 1080)
    tick_dir: str = 'out'
    tick_length: int = 2
    tick_width: int = 1
    x_tick_angle: int = 45

    # --- slice planes (3-D runs only, in plot axis units) ---
    x_plane: float = 0.0
    y_plane: float = 0.0
    z_plane: float = 0.0

    # --- zoom axis limits (in plot axis units): None means use the full simulation domain ---
    axis_lims_zoom: Optional[list[float]] = None

    # --- axis unit: conversion factor from data coordinate units to plot axis units ---
    axis_unit: Optional[float] = None
    axis_unit_label: Optional[str] = None

    # --- planet disk: (xz plane, xy plane, yz plane) ---
    show_planet: tuple[bool, bool, bool] = (True, True, False)

    # --- time range for this process ---
    t_start: int = 0
    t_end: int = 1_000_000

    # --- aggregated objects (not part of the TOML header scalars) ---
    plot_params: PlotParams = field(default_factory=PlotParams)
    runs: list[RhybridRun] = field(default_factory=list)

    @classmethod
    def from_toml(cls, toml_path: str) -> 'SlicePlotJobConfig':
        """Build a SlicePlotJobConfig from a TOML plot-config file."""
        header = load_plotter_settings(toml_path)

        run_config = RhybridConfigParser()
        with open(Path(header['runConfig']).resolve()) as f:
            run_config.read_file(f)

        t_start=int(header.get('tStartThisProcess', 0))
        t_end=int(header.get('tEndThisProcess', 1_000_000))
        run = RhybridRun(run_config, header['runFolder'], header['runDescr'], step_range=(t_start, t_end))
        plot_params = PlotParams(toml_path)

        zoom = header.get('axisLimsZoom', None)
        if zoom == -1:          # sentinel value used in the TOML
            zoom = None

        axis_unit = header.get('axisUnit', None)
        axis_unit_label = header.get('axisUnitLabel', None)
        if not axis_unit:
            axis_unit = float(run.config_params['r_object'])
            axis_unit_label = "$R_p$"
        else:
            axis_unit = float(axis_unit)

        return cls(
            n_cores=header.get("Ncores", 1),
            output_dir=header.get('outputFolder', './png/'),
            fig_dpi=header.get('figDpi', 100),
            x_plane=float(header.get('xPlane', 0.0)),
            y_plane=float(header.get('yPlane', 0.0)),
            z_plane=float(header.get('zPlane', 0.0)),
            axis_lims_zoom=zoom,
            axis_unit=axis_unit,
            axis_unit_label=axis_unit_label,
            show_planet=tuple(header.get('showPlanet', (1, 1, 0))),
            t_start=t_start,
            t_end=t_end,
            plot_params=plot_params,
            runs=[run]
        )


def load_plotter_settings(toml_path: str) -> dict:
    """ Read *toml_path* and return a validated dict of plotter settings from the header section. """

    if not os.path.isfile(toml_path):
        raise FileNotFoundError(f'plot_parameters: config file not found: {toml_path}')

    with open(toml_path, 'rb') as fh:
        raw = tomllib.load(fh)

    header = raw.get('header', {})
    if not header:
        raise ValueError(f'plot_parameters: no [header] block found in {toml_path}')

    _validate_header(header)

    return header


def _validate_header(header: dict) -> None:
    """Raise ValueError with a clear message if *header* is malformed."""
    run_cfg = header['runConfig']
    if not os.path.exists(run_cfg):
        raise FileNotFoundError(f'plot_parameters: run config not found: {run_cfg}')

    run_folder = header['runFolder']
    if not os.path.isdir(run_folder):
        raise FileNotFoundError(f'plot_parameters: run folder not found: {run_folder}')

    if not any(
            f.startswith('state') and f.endswith('.vlsv')
            for f in os.listdir(run_folder)
    ):
        raise FileNotFoundError(f'plot_parameters: no state*.vlsv found in folder: {run_folder}')

    if header['tStartThisProcess'] > header['tEndThisProcess']:
        raise ValueError('plot_parameters: tStart > tEnd (this process)')

    for key in ('tStartThisProcess', 'tEndThisProcess'):
        if header[key] < 0:
            raise ValueError(f'plot_parameters: negative time value: {key} = {header[key]}')


_plot_configs = {}


def _render_step(plot_configs: list[SlicePlotConfig], vlsv_path: str):
    for cfg in plot_configs:
        plotter = SlicePlot(VlsvReader(vlsv_path), cfg)
        plotter.render_plot()


class SlicePlotJob:
    """Renders xz / xy / yz slice panels for a 3D or 2D RHybrid run.

    One instance per plotting job (one TOML config).  The public interface
    is just two methods:

        plotter = SlicePlotJob(cfg)
        plotter.plot_step(400)          # one time step
        plotter.save_all([0, 400, 800]) # all steps, optionally parallel
    """

    def __init__(self, cfg: SlicePlotJobConfig) -> None:
        if len(cfg.runs) != 1:
            raise ValueError(
                f'SlicePlotJob expects exactly one run, got {len(cfg.runs)}')

        self._job_cfg = cfg
        sim_box_lims, axis_lims, ticks = self._init_geometry()
        slice_points = [(0, self._job_cfg.x_plane * self._job_cfg.axis_unit),
                        (1, self._job_cfg.y_plane * self._job_cfg.axis_unit),
                        (2, self._job_cfg.z_plane * self._job_cfg.axis_unit)]
        self._global_plot_params = dict(
            sim_name=self._job_cfg.runs[0].run_descr,

            # --- output ---
            output_dir=self._job_cfg.output_dir,

            # --- figure style ---
            fig_dpi=self._job_cfg.fig_dpi,
            fig_resolution=self._job_cfg.fig_resolution,

            # --- simulation box and axis limits in data units ---
            sim_box_lims=sim_box_lims,
            axis_lims=axis_lims,
            x_unit=self._job_cfg.axis_unit,
            x_unit_label=self._job_cfg.axis_unit_label,

            # --- axis tick locations in data units ---
            tick_locs=ticks,

            # --- slice points for 3D simulations in format (axis, location in data coord units) ---
            slice_points=slice_points,

            # --- planet disk: (xz plane, xy plane, yz plane) ---
            show_planet=self._job_cfg.show_planet,
            r_planet=float(self._job_cfg.runs[0].config_params['r_object']),
            rp_str="$R_p"
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_geometry(self) -> tuple[list, list, np.ndarray]:
        """Resolve global axis parameters."""
        domain = self._job_cfg.runs[0].config_params['domain']
        sim_box_lims = [(domain['x_min'], domain['x_max']),
                        (domain['y_min'], domain['y_max']),
                        (domain['z_min'], domain['z_max'])]

        if self._job_cfg.axis_lims_zoom:
            zoom = [lim * self._job_cfg.axis_unit for lim in self._job_cfg.axis_lims_zoom]
        else:
            zoom = [domain['x_min'], domain['x_max'], domain['y_min'], domain['y_max'],
                    domain['z_min'], domain['z_max']]

        axis_lims = [(zoom[0], zoom[1]), (zoom[2], zoom[3]), (zoom[4], zoom[5])]

        # The tick computation might work better in the plot axis units:
        ticks = _compute_tick_positions([v / self._job_cfg.axis_unit for v in zoom],
                                        domain['x_size'], domain['y_size'], domain['z_size'])
        ticks = ticks * self._job_cfg.axis_unit

        return sim_box_lims, axis_lims, ticks

    def save_all(self, steps: list[int] = None, n_cores: int = None) -> None:
        if not steps:
            steps = self.get_steps()
        if not n_cores:
            n_cores = self._job_cfg.n_cores

        self._init_worker_args(steps)
        with Pool(n_cores) as pool:
            pool.starmap(_render_step, [(item["cfgs"], item["vlsv_path"]) for item in _plot_configs.values()])

    def _init_worker_args(self, steps: list[int]) -> None:
        """Initialize SlicePlot arguments for each worker."""

        _plot_configs.clear()
        for step in steps:
            _plot_configs[step] = {
                "cfgs": [
                    SlicePlotConfig(
                        var_name=var_opts["param"],
                        var_type=var_opts["type"],
                        var_unit=var_opts["unit"],
                        var_label=var_opts["str"],
                        var_vmin=var_opts["lims"][0] * var_opts["unit"],
                        var_vmax=var_opts["lims"][1] * var_opts["unit"],
                        smooth_sig=var_opts["sigma"],
                        colormap=var_opts["colormap"],
                        log_col_scale=var_opts["log"],
                        output_name=f'{var_opts["filename"]}_state{step:08d}.png',
                        **self._global_plot_params
                    )
                    for var_opts in self._job_cfg.plot_params
                ],
                "vlsv_path": self._job_cfg.runs[0].get_vlsv_file_path(step)
            }


    def get_steps(self):
        return self._job_cfg.runs[0].get_steps_in_range(self._job_cfg.t_start, self._job_cfg.t_end)


    def plot_step(self, step: int, vlsv_path: str=None) -> None:
        """Render and save all parameter panels for one time step.

        vlsv_path can be read
        """

        if not vlsv_path:
            vlsv_path = self._job_cfg.runs[0].get_vlsv_file_path(step)
        vr = VlsvReader(vlsv_path)
        for var_opts in self._job_cfg.plot_params:
            out_name = (
                f'{var_opts["filename"]}_state{step:08d}.png'
            )
            plot_cfg = SlicePlotConfig(
                var_name=var_opts["param"],
                var_type=var_opts["type"],
                var_unit=var_opts["unit"],
                var_label=var_opts["str"],
                var_vmin=var_opts["lims"][0],
                var_vmax=var_opts["lims"][1],
                smooth_sig=var_opts["sigma"],
                colormap=var_opts["colormap"],
                log_col_scale=var_opts["log"],
                output_name=out_name,
                **self._global_plot_params
            )


def _compute_tick_positions(zoom: list[float],
                            nx: int, ny: int, nz: int) -> np.ndarray:
    """Return a 'nice' array of tick positions for the zoomed domain."""
    ranges = [
        zoom[1] - zoom[0] if nx > 1 else -1,
        zoom[3] - zoom[2] if ny > 1 else -1,
        zoom[5] - zoom[4] if nz > 1 else -1,
    ]
    max_range = max(ranges)
    max_coord = max(abs(v) for v in zoom)
    scale = pow(10, np.ceil(np.log10(max_coord))) if max_coord > 0 else 1

    tick_step, n_ticks = scale, -1
    for ii, jj, kk in itertools.product(range(1, 100), (1, 2, 5), (-1, 1)):
        tick_step = scale / (kk * jj * ii)
        n_ticks = max_range / tick_step
        if 5 <= n_ticks <= 15:
            break
    if not (5 <= n_ticks <= 15):
        print(HN + f'WARNING: no good tick step (n_ticks={n_ticks:.1f}, step={tick_step})')

    return np.arange(-scale, +scale, step=tick_step)


# ----------------------------------------------------------------------------------
# SlicePlotConfig & SlicePlot  — Plotter utilities for single snapshot slice plots
# ----------------------------------------------------------------------------------

@dataclass
class SlicePlotConfig:
    """ All configuration for a single slice-plot figure. """

    sim_name: str = ""

    # --- plot variable specs ---
    var_name: str = ""
    var_type: str = ""
    var_unit: float = 1.0   # conversion factor from data to plot units
    var_label: str = ""
    var_vmin: float = 0.0   # in data units
    var_vmax: float = 1.0   # in data units
    smooth_sig: float = -1
    colormap: Optional[str] = None
    log_col_scale: bool = False

    # --- output ---
    output_dir: str = './png'
    output_name: str = 'slice_plot.png'

    # --- figure style ---
    fig_dpi: int = 100
    fig_resolution: tuple[int, int] = (1920, 1080)

    # --- simulation box and axis limits in data units ---
    #       - simulation box limits are used as the bounding box in data coordinates for the image
    #       - axis limits are used as the bounds of the actual figure axis inside the image (can be e.g. zoomed)
    sim_box_lims: list[tuple[float, float]] = field(default_factory=list)
    axis_lims: Optional[list[tuple[float, float]]] = None
    x_unit: float = 1.0     # conversion factor from data coordinate units to plot axis units
    x_unit_label: str = "$R_p$"

    # --- axis ticks ---
    tick_locs: Optional[np.ndarray] = None
    tick_dir: str = 'out'
    tick_length: int = 2
    tick_width: int = 1
    x_tick_angle: int = 45

    # --- slice points for 3D simulations in format (axis, location in data coord units) ---
    slice_points: Optional[list[tuple[int, float]]] = None

    # --- planet disk: (xz plane, xy plane, yz plane) ---
    show_planet: tuple[bool, ...] = (True, True, False)
    r_planet: float = 1.0   # in data coord units
    rp_str: str = "$R_p$"

    # --- derived / cached ---
    fig_size: tuple[float, float] = field(init=False, repr=False)
    n_rows: int = field(init=False, repr=False)
    n_cols: int = field(init=False, repr=False)

    def __post_init__(self):
        # Compute figure size once so callers don't repeat the arithmetic.
        w, h = self.fig_resolution
        self.fig_size = (w / self.fig_dpi, h / self.fig_dpi)
        self.n_rows = 1
        self.n_cols = 2 if not self.slice_points else len(self.slice_points)

        if not self.axis_lims:
            self.axis_lims = self.sim_box_lims

class SlicePlot:
    """ Renders slice panels for a single snapshot of a 3D or 2D Rhybrid run. """

    def __init__(self, vr: VlsvReader, cfg: SlicePlotConfig):
        self._cfg = cfg

        # Get simulation data:
        self.time = vr.read_parameter(get_time_param(vr))
        self.timestep = vr.read_parameter("timestep")
        self.var_data = read_variable_data(vr, cfg.var_name, cfg.var_type)
        self.n_cell = self.var_data.shape
        self.sim_dim = len(self.n_cell)

        # Determine slice columns in data array in format (axis, idx, location in m):
        self.slice_points = [(-1, -1, -1)] if not cfg.slice_points else \
            [(axis, self._choose_ncol(axis, sloc), sloc) for (axis, sloc) in cfg.slice_points]

    def _choose_ncol(self, axis, sloc) -> int:
        """Return cell index for a slice plane."""
        (r_min, r_max) = self._cfg.sim_box_lims[axis]
        n = self.n_cell[axis]
        saxis_name = ("x", "y", "z")[axis]

        if sloc > r_max or sloc < r_min:
            sloc = (r_max + r_min) / 2.0
            print(HN + f'WARNING: {saxis_name}-plane out of domain, '
                       f'clamping to {sloc / self._cfg.x_unit:.2f} {self._cfg.x_unit_label}.')
        dr = (r_max - r_min) / n
        col = int(np.floor((sloc - r_min) / dr))
        return max(0, min(col, n - 1))

    def render_plot(self) -> None:

        print(HN + f'{self._cfg.sim_name} | {self.timestep} | '
                   f'{self._cfg.var_name} {self._cfg.var_type}')

        vmin = self._cfg.var_vmin / self._cfg.var_unit
        vmax = self._cfg.var_vmax / self._cfg.var_unit
        rp_str = self._cfg.rp_str

        fig, axes = plt.subplots(
            nrows=self._cfg.n_rows, ncols=self._cfg.n_cols,
            figsize=self._cfg.fig_size,
            frameon=True, squeeze=False)

        is_first_col = True
        a_cbar = None
        for ax, (saxis, sidx, sloc), show_planet in zip(
                axes[0], self.slice_points, self._cfg.show_planet):
            plot_data = self.var_data.take(sidx, axis=saxis) if saxis != -1 else self.var_data
            plot_data = self._maybe_smooth(plot_data) / self._cfg.var_unit

            # Helpers:
            list_drop_i = lambda l, i_drop: [item for i, item in enumerate(l) if i != i_drop]
            flatten_lims = lambda limlist: sum([list(lims) for lims in limlist], [])

            # Get axis details for the slice:
            sbox_lims = [(l[0] / self._cfg.x_unit, l[1] / self._cfg.x_unit) for l in self._cfg.sim_box_lims]
            ax_lims = [(l[0] / self._cfg.x_unit, l[1] / self._cfg.x_unit) for l in self._cfg.axis_lims]
            image_extent = flatten_lims(list_drop_i(sbox_lims, saxis)) if saxis != -1 else flatten_lims(sbox_lims)
            ax_lims = list_drop_i(ax_lims, saxis) if saxis != -1 else ax_lims
            xyz = ["x", "y", "z"]
            saxis_name = xyz[saxis]
            ax_names = list_drop_i(xyz, saxis) if saxis != -1 else ["x", "y"]   # TODO: Confirm the 2D array axis order!

            a_cbar = ax.imshow(
                plot_data.transpose(),
                vmin=vmin,
                vmax=vmax,
                cmap=self._cfg.colormap,
                extent=image_extent,
                aspect='equal',
                origin='lower',
                interpolation='nearest',
            )

            self._configure_axes(
                ax,
                f'${ax_names[0]}$ [{self._cfg.x_unit_label}]',
                f'${ax_names[1]}$ [{self._cfg.x_unit_label}]',
                f'{self.sim_dim}D: ${"".join(ax_names)}$ '
                f'(${saxis_name}=${self._plane_title(sloc / self._cfg.x_unit, self._cfg.x_unit_label)})',
                ax_lims[0], ax_lims[1],
                is_bottom_row=True, is_first_col=is_first_col)
            self._configure_panel(ax, vmin, vmax, a_cbar, show_planet)

            is_first_col = False

        fig.suptitle(f"$t = {str(round(self.time * 10) / 10)}$ s")
        fig.tight_layout()
        self._add_colorbar(fig, axes, a_cbar)

        fig_path = Path(self._cfg.output_dir) / self._cfg.output_name
        fig.savefig(fig_path, dpi=self._cfg.fig_dpi, transparent=False)
        plt.clf()
        plt.close(fig)

    def _maybe_smooth(self, arr: np.ndarray) -> np.ndarray:
        sigma = self._cfg.smooth_sig
        if sigma > 0:
            return sp.ndimage.gaussian_filter(arr, sigma=sigma, mode='constant')
        return arr

    def _configure_axes(self, ax, xlabel: str, ylabel: str,
                        title: Optional[str], xlim, ylim,
                        is_bottom_row: bool, is_first_col: bool) -> None:
        if is_bottom_row:
            ax.set_xlabel(xlabel)
        if is_first_col:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.title.set_text(title)
        if self._cfg.tick_locs is not None:
            ax.set_xticks(self._cfg.tick_locs / self._cfg.x_unit)
            ax.set_yticks(self._cfg.tick_locs / self._cfg.x_unit)
        ax.tick_params('x', labelrotation=self._cfg.x_tick_angle)
        ax.axis('scaled')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def _configure_panel(self, ax, vmin: float, vmax: float, artist, show_planet: bool) -> None:
        """Overlay planet disk and set log normalisation if needed."""
        if show_planet:
            ax.add_artist(Wedge((0, 0), self._cfg.r_planet / self._cfg.x_unit, 90, 270, fc='dimgray'))
            ax.add_artist(Wedge((0, 0), self._cfg.r_planet / self._cfg.x_unit, 270, 90, fc='w'))
            ax.add_artist(plt.Circle((0, 0), self._cfg.r_planet / self._cfg.x_unit, color='k', fill=False, lw=0.5))
        if self._cfg.log_col_scale:
            artist.set_norm(colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.tick_params(which='both', direction=self._cfg.tick_dir,
                       length=self._cfg.tick_length, width=self._cfg.tick_width)

    def _add_colorbar(self, fig, axes, artist) -> None:

        # Try to compute a reasonably sized space for the colorbar wrt. the starting point of 0.15 times the
        # std. axis size of a 1920 x 1080 figure:
        cbar_space = 0.15 / ((fig.get_figwidth() / 1920) / (fig.get_figheight() / 1080))
        clb = fig.colorbar(artist, ax=axes.flatten(), shrink=0.5, fraction=cbar_space)
        clb.ax.set_title(self._cfg.var_label)

    def _plane_title(self, coord: float, unit_str: str) -> str:
        val = str(round(coord * 10) / 10) if abs(coord) > 0 else '0'
        return val + unit_str