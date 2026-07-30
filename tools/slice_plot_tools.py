import itertools
import socket
from dataclasses import dataclass, field
from multiprocessing import Pool, Value
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


# ---------------------------------------------------------------------------
# SlicePlotConfig  — all figure-wide settings in one place
# ---------------------------------------------------------------------------

@dataclass
class SlicePlotConfig:
    """ All configuration for a single slice-plot figure. """

    sim_name: str= ""

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
    output_path: str = './png/slice_plot.png'

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

    def __init__(self, vlsv_file_path: Path, cfg: SlicePlotConfig):
        self._cfg = cfg

        # Get simulation data:
        vr = VlsvReader(vlsv_file_path)
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
                       f'clamping to {sloc / self._cfg.r_planet:.2f} Rp')
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
            plot_data = self._maybe_smooth(plot_data, self._cfg.smooth_sig) / self._cfg.var_unit

            # Helpers:
            list_drop_i = lambda l, i_drop: [item for i, item in enumerate(l) if i != i_drop]
            flatten_lims = lambda limlist: sum([list(lims) for lims in limlist], [])

            # Get axis details for the slice:
            sbox_lims = [(l[0] / self._cfg.x_unit, l[1] / self._cfg.x_unit) for l in self._cfg.sim_box_lims]
            ax_lims = [(l[0] / self._cfg.x_unit, l[1] / self._cfg.x_unit) for l in self._cfg.axis_lims]
            image_extent = flatten_lims(list_drop_i(sbox_lims, saxis)) if saxis != -1 else sbox_lims
            ax_lims = list_drop_i(ax_lims, saxis) if saxis != -1 else ax_lims
            xyz = ["z", "y", "x"]
            saxis_name = xyz[saxis]
            ax_names = list_drop_i(xyz, saxis) if saxis != -1 else ["x", "y"]   # TODO: Confirm the 2D array axis order!

            a_cbar = ax.imshow(
                plot_data,
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
                f'(${saxis_name}=${_plane_title(sloc / self._cfg.x_unit, self._cfg.x_unit_label)})',
                ax_lims[0], ax_lims[1],
                is_bottom_row=True, is_first_col=is_first_col)
            self._configure_panel(ax, vmin, vmax, a_cbar, show_planet)

            is_first_col = False

        fig.tight_layout()
        self._add_colorbar(fig, axes, a_cbar)

        fig.savefig(self._cfg.output_path, dpi=self._cfg.fig_dpi, transparent=False)
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
        if self._cfg.tick_locs:
            ax.set_xticks(self._cfg.tick_locs)
            ax.set_yticks(self._cfg.tick_locs)
        ax.tick_params('x', labelrotation=self._cfg.x_tick_angle)
        ax.axis('scaled')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def _configure_panel(self, ax, vmin: float, vmax: float, artist, show_planet: bool) -> None:
        """Overlay planet disk and set log normalisation if needed."""
        if show_planet:
            ax.add_artist(Wedge((0, 0), 1.0, 90, 270, fc='dimgray'))
            ax.add_artist(Wedge((0, 0), 1.0, 270, 90, fc='w'))
            ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, lw=0.5))
        if self._cfg.log_col_scale:
            artist.set_norm(colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.tick_params(which='both', direction=self._cfg.tick_dir,
                       length=self._cfg.tick_length, width=self._cfg.tick_width)

    def _add_colorbar(self, fig, axes, artist) -> None:
        clb = fig.colorbar(artist, ax=axes.flatten(), shrink=0.5)
        clb.ax.set_title(self._cfg.var_label)
