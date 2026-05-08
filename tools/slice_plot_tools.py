import itertools
import socket
from dataclasses import dataclass, field
from multiprocessing import Pool, Value
from pathlib import Path
from typing import Optional
import tomllib
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Wedge

from plot_parameters import PlotParams
from rhybrid_configparser import RhybridConfigParser
from simrun import RhybridRun

plt.switch_backend('agg')
HN = '(hostname = ' + socket.gethostname() + ') '


# ---------------------------------------------------------------------------
# SlicePlotConfig  — all figure-wide settings in one place
# ---------------------------------------------------------------------------

@dataclass
class SlicePlotConfig:
    """All configuration for one complete slice-plot job.

    Constructed directly or via the SlicePlotConfig.from_toml() factory.
    Holds references to the parameter registry and the list of runs so
    that the plotting classes only need to receive a single object.
    """

    # --- output ---
    output_dir: str = './png/'

    # --- figure style ---
    fig_dpi: int = 100
    fig_resolution: tuple[int, int] = (1920, 1080)
    tick_dir: str = 'out'
    tick_length: int = 2
    tick_width: int = 1
    x_tick_angle: int = 45

    # --- slice planes (3-D runs only, in units of Rp) ---
    x_plane: float = 0.0
    y_plane: float = 0.0
    z_plane: float = 0.0

    # --- zoom: None means use the full simulation domain ---
    axis_lims_zoom: Optional[list[float]] = None

    # --- planet disk: (xz plane, xy plane, yz plane) ---
    show_planet: tuple[bool, bool, bool] = (True, True, False)

    # --- time range for this process ---
    t_start: int = 0
    t_end: int = 1_000_000

    # --- aggregated objects (not part of the TOML header scalars) ---
    plot_params: PlotParams = field(default_factory=PlotParams)
    runs: list[RhybridRun] = field(default_factory=list)

    # --- derived / cached ---
    _fig_size: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self):
        # Compute figure size once so callers don't repeat the arithmetic.
        w, h = self.fig_resolution
        self._fig_size = (w / self.fig_dpi, h / self.fig_dpi)

    @classmethod
    def from_toml(cls, toml_path: str) -> 'SlicePlotConfig':
        """Build a SlicePlotConfig from a TOML plot-config file."""
        header = load_plotter_settings(toml_path)

        run_config = RhybridConfigParser()
        with open(Path(header['runConfig']).resolve()) as f:
            run_config.read_file(f)

        run = RhybridRun(run_config, header['runFolder'], header['runDescr'])
        plot_params = PlotParams(toml_path)

        zoom = header.get('axisLimsZoom', None)
        if zoom == -1:          # sentinel value used in the TOML
            zoom = None

        return cls(
            output_dir=header.get('outputFolder', './png/'),
            fig_dpi=header.get('figDpi', 100),
            x_plane=float(header.get('xPlane', 0.0)),
            y_plane=float(header.get('yPlane', 0.0)),
            z_plane=float(header.get('zPlane', 0.0)),
            axis_lims_zoom=zoom,
            show_planet=tuple(header.get('showPlanet', (1, 1, 0))),
            t_start=int(header.get('tStartThisProcess', 0)),
            t_end=int(header.get('tEndThisProcess', 1_000_000)),
            plot_params=plot_params,
            runs=[run],
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
    
    runFolder = header['runFolder']
    if not os.path.isdir(runFolder):
        raise FileNotFoundError(f'plot_parameters: run folder not found: {runFolder}')
    
    if not any(
        f.startswith('state') and f.endswith('.vlsv')
        for f in os.listdir(runFolder)
    ):    
        raise FileNotFoundError(f'plot_parameters: no state*.vlsv found in folder: {runFolder}')
    
    if header['tStartThisProcess'] > header['tEndThisProcess']:
        raise ValueError('plot_parameters: tStart > tEnd (this process)')
    
    if header['tStartGlobal'] > header['tEndGlobal']:
        raise ValueError('plot_parameters: tStartGlobal > tEndGlobal')
    
    for key in ('tStartThisProcess', 'tEndThisProcess',
                'tStartGlobal', 'tEndGlobal'):
        if header[key] < 0:
            raise ValueError(f'plot_parameters: negative time value: {key} = {header[key]}')


# ---------------------------------------------------------------------------
# Shared rendering helpers  — pure functions, no class state
# ---------------------------------------------------------------------------

def _choose_ncol(r_plane: float, r_min: float, r_max: float,
                 n: int, axis: str, r_object: float) -> tuple[int, float]:
    """Return (cell index, clamped plane coordinate in Rp) for a slice plane."""
    if r_plane > r_max or r_plane < r_min:
        r_plane = (r_max + r_min) / 2.0
        print(HN + f'WARNING: {axis}Plane out of domain, '
              f'clamping to {r_plane / r_object:.2f} Rp')
    dr = (r_max - r_min) / n
    col = int(np.floor((r_plane - r_min) / dr))
    return max(0, min(col, n - 1)), r_plane / r_object


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


def _maybe_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma > 0:
        return sp.ndimage.gaussian_filter(arr, sigma=sigma, mode='constant')
    return arr


def _plot_slice(ax, mesh: np.ndarray, var_set: dict, extent: list) -> object:
    """Call imshow for one 2-D slice; return the artist for the colorbar."""
    unit = var_set['unit']
    return ax.imshow(
        mesh / unit,
        vmin=var_set['lims'][0] / unit,
        vmax=var_set['lims'][1] / unit,
        cmap=var_set['colormap'],
        extent=extent,
        aspect='equal',
        origin='lower',
        interpolation='nearest',
    )


def _configure_axes(ax, xlabel: str, ylabel: str,
                    title: Optional[str],
                    xlim, ylim, ticks: np.ndarray,
                    is_bottom_row: bool, is_first_col: bool,
                    cfg: SlicePlotConfig) -> None:
    if is_bottom_row:
        ax.set_xlabel(xlabel)
    if is_first_col:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.title.set_text(title)
    ax.set_xticks(ticks)
    ax.tick_params('x', labelrotation=cfg.x_tick_angle)
    ax.set_yticks(ticks)
    ax.axis('scaled')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _configure_panel(ax, var_set: dict, artist, show_planet: bool,
                     cfg: SlicePlotConfig) -> None:
    """Overlay planet disk and set log normalisation if needed."""
    if show_planet:
        ax.add_artist(Wedge((0, 0), 1.0, 90, 270, fc='dimgray'))
        ax.add_artist(Wedge((0, 0), 1.0, 270, 90, fc='w'))
        ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, lw=0.5))
    if var_set['log']:
        unit = var_set['unit']
        artist.set_norm(colors.LogNorm(
            vmin=var_set['lims'][0] / unit,
            vmax=var_set['lims'][1] / unit))
    ax.tick_params(which='both', direction=cfg.tick_dir,
                   length=cfg.tick_length, width=cfg.tick_width)


def _add_colorbar(fig, axes, artist, label: str, shrink: float = 0.5) -> None:
    clb = fig.colorbar(artist, ax=axes.flatten(), shrink=shrink)
    clb.ax.set_title(label)


def _round_str(x: float) -> str:
    return str(round(x * 10) / 10)


def _plane_title(coord: float, unit_str: str) -> str:
    val = str(round(coord * 10) / 10) if abs(coord) > 0 else '0'
    return val + unit_str

def _render_step(toml_path: str, step: int) -> None:
    """Top-level worker function: reconstructs the plotter from config.
    
    For pickling the SlicePlot3D class with multiprocessing, a new instance 
    needs to be created at each step. This might look a bit hacky, but does 
    not compromise efficiency as the bottlenecks in computation and memory 
    usage are vlsv file I/O operations and figure rendering.
    """
    cfg = SlicePlotConfig.from_toml(toml_path)
    plotter = SlicePlot3D(cfg)
    plotter.plot_step(step)

# ---------------------------------------------------------------------------
# SlicePlot3D
# ---------------------------------------------------------------------------

class SlicePlot3D:
    """Renders xz / xy / yz slice panels for a 3-D RHybrid run.

    One instance per plotting job (one TOML config).  The public interface
    is just two methods:

        plotter = SlicePlot3D(cfg)
        plotter.plot_step(400)          # one time step
        plotter.save_all([0, 400, 800]) # all steps, optionally parallel
    """

    def __init__(self, cfg: SlicePlotConfig) -> None:
        if len(cfg.runs) != 1:
            raise ValueError(
                f'SlicePlot3D expects exactly one run, got {len(cfg.runs)}')

        self._cfg = cfg
        self._run = cfg.runs[0]
        self._rp = float(self._run.config_params['r_object'])
        self._rp_str = '$R_p$'

        # Figure grid: one row per run (here always 1), three columns (xz, xy, yz)
        self._n_rows = 1
        self._n_cols = 3

        # Multiprocessing-safe file-open counter (shared across workers)
        self._open_cnt = Value('i', 0)

        # Resolve and cache the slice column indices once, from the first step.
        # These are fixed for the entire run (the grid doesn't change between
        # steps), so there is no need to recompute them per step.
        self._yz_col: Optional[int] = None   # column index for the x-plane cut
        self._xz_col: Optional[int] = None   # column index for the y-plane cut
        self._xy_col: Optional[int] = None   # column index for the z-plane cut
        self._clamped_x: Optional[float] = None  # actual x_plane after clamping (Rp)
        self._clamped_y: Optional[float] = None
        self._clamped_z: Optional[float] = None
        self._zoom: Optional[list[float]] = None
        self._ticks: Optional[np.ndarray] = None

        self._init_geometry()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_geometry(self) -> None:
        """Resolve slice indices and zoom limits from the first VLSV file."""
        domain = self._run.config_params['domain']
        nx = domain['x_size']
        ny = domain['y_size']
        nz = domain['z_size']
        xmin, xmax = domain['x_min'], domain['x_max']
        ymin, ymax = domain['y_min'], domain['y_max']
        zmin, zmax = domain['z_min'], domain['z_max']

        rp = self._rp
        cfg = self._cfg

        self._yz_col, self._clamped_x = _choose_ncol(
            cfg.x_plane * rp, xmin, xmax, nx, 'x', rp)
        self._xz_col, self._clamped_y = _choose_ncol(
            cfg.y_plane * rp, ymin, ymax, ny, 'y', rp)
        self._xy_col, self._clamped_z = _choose_ncol(
            cfg.z_plane * rp, zmin, zmax, nz, 'z', rp)

        # Axis limits in Rp
        full_zoom = [xmin/rp, xmax/rp, ymin/rp, ymax/rp, zmin/rp, zmax/rp]
        self._zoom = cfg.axis_lims_zoom if cfg.axis_lims_zoom is not None \
                     else full_zoom
        self._axis_lims = full_zoom
        self._ticks = _compute_tick_positions(self._zoom, nx, ny, nz)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot_step(self, step: int) -> None:
        """Render and save all parameter panels for one time step."""
        file_time = self._read_time(step)

        for var_set in self._cfg.plot_params:
            
            print(HN + f'{self._run.run_out_dir.name} | step {step} | '
                  f'{var_set["param"]} {var_set["type"]}')
            
            # with self._open_cnt.get_lock():
            #     self._open_cnt.value += 1
            #     print(HN + f'[{self._open_cnt.value}] '
            #           f'{self._run.run_out_dir.name} | step {step} | '
            #           f'{var_set["param"]} {var_set["type"]}')

            fig, axes = plt.subplots(
                nrows=self._n_rows, ncols=self._n_cols,
                figsize=self._cfg._fig_size,
                frameon=True, squeeze=False)

            self._render_panels(fig, axes, var_set, step, file_time)

            out_name = (
                f'{var_set["filename"]}_state{step:08d}.vlsv.png'
            )
            out_path = Path(self._cfg.output_dir) / out_name
            fig.savefig(out_path, dpi=self._cfg.fig_dpi, transparent=False)
            plt.clf()
            plt.close(fig)
                
    def save_all(self, steps: list[int] = None, n_cores: int = 1) -> None:
        """Render and save all time steps, optionally in parallel."""
        if n_cores > 1:
            toml_path = self._cfg.plot_params.toml_path
            with Pool(n_cores) as pool:
                pool.starmap(_render_step, [(toml_path, s) for s in steps])
        else:
            for step in steps:
                self.plot_step(step)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _read_time(self, step: int) -> float:
        result = self._run.read_parameter(['t'], [step])
        return float(result['t'][0])

    def _load_data(self, var_set: dict, step: int) -> np.ndarray:
        """Read variable data for one step and return as a (nz, ny, nx) array."""

        # read_variable_data already applies CellID ordering and reshapes,
        # so data is (nz, ny, nx) — just index [0] for the step axis.
        var, vtype = var_set['param'], var_set['type']
        data = self._run.read_variable_data(var, vtype, step)[".".join([var, vtype])][0]

        return data

    def _render_panels(self, fig, axes, var_set: dict,
                       step: int, file_time: float) -> None:
        """Render the three slice panels (xz, xy, yz) into axes[0][0..2]."""
        D = self._load_data(var_set, step)
        sigma = var_set['sigma']
        lims = self._axis_lims
        zoom = self._zoom
        ticks = self._ticks
        rp_str = self._rp_str
        cfg = self._cfg
        is_only_row = True   # SlicePlot3D always has exactly one run / one row

        # --- xz panel (column 0) ---
        mesh_xz = _maybe_smooth(D[:, self._xz_col, :], sigma)
        a = _plot_slice(axes[0][0], mesh_xz, var_set,
                        [lims[0], lims[1], lims[4], lims[5]])
        _configure_axes(
            axes[0][0],
            f'$x$ [{rp_str}]', f'$z$ [{rp_str}]',
            f'3D: $xz$ ($y=$ {_plane_title(self._clamped_y, rp_str)})',
            zoom[0:2], zoom[4:6], ticks,
            is_bottom_row=is_only_row, is_first_col=True, cfg=cfg)
        _configure_panel(axes[0][0], var_set, a, cfg.show_planet[0], cfg)

        # --- xy panel (column 1, carries the time stamp in its title) ---
        mesh_xy = _maybe_smooth(D[self._xy_col, :, :], sigma)
        a = _plot_slice(axes[0][1], mesh_xy, var_set,
                        [lims[0], lims[1], lims[2], lims[3]])
        _configure_axes(
            axes[0][1],
            f'$x$ [{rp_str}]', f'$y$ [{rp_str}]',
            (f'$t=$ {_round_str(file_time)} s\n'
             f'3D: $xy$ ($z=$ {_plane_title(self._clamped_z, rp_str)})'),
            zoom[0:2], zoom[2:4], ticks,
            is_bottom_row=is_only_row, is_first_col=False, cfg=cfg)
        _configure_panel(axes[0][1], var_set, a, cfg.show_planet[1], cfg)

        # --- yz panel (column 2) ---
        mesh_yz = _maybe_smooth(D[:, :, self._yz_col], sigma)
        a = _plot_slice(axes[0][2], mesh_yz, var_set,
                        [lims[2], lims[3], lims[4], lims[5]])
        _configure_axes(
            axes[0][2],
            f'$y$ [{rp_str}]', f'$z$ [{rp_str}]',
            f'3D: $yz$ ($x=$ {_plane_title(self._clamped_x, rp_str)})',
            zoom[2:4], zoom[4:6], ticks,
            is_bottom_row=is_only_row, is_first_col=False, cfg=cfg)
        _configure_panel(axes[0][2], var_set, a, cfg.show_planet[2], cfg)

        fig.tight_layout()
        _add_colorbar(fig, axes, a, var_set['str'], shrink=0.5)
        

class SlicePlot2D:
    def __init__(self, cfg: SlicePlotConfig): ...

    def plot_step(self, step: int) -> None:
        """Render and save all parameter panels for one time step."""

    def save_all(self, steps: list[int], n_cores: int = 1) -> None:
        """Dispatch plot_step over all steps, optionally in parallel."""
