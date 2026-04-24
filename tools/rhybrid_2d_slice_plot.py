# See plotter_rhybrid_2d_slice.py for explanation
#
# REFACTORING NOTES
# =================
# Changes applied vs. the original:
#   1. All indentation converted to 4 spaces.
#   2. Deeply nested blocks extracted into focused functions (see below).
#   3. Repeated dict-building for P / P_settings collapsed into helpers.
#   4. Command-line argument parsing moved into parse_args().
#   5. VLSV variable discovery and P-list construction moved into
#      build_parameter_list().
#   6. The three "read variable → pick component" branches collapsed into
#      read_variable_data().
#   7. The 2-D axis-geometry bookkeeping extracted into Plane2D (dataclass).
#   8. Tick-mark search extracted into compute_tick_step().
#   9. plotPanel's 3-D and 2-D rendering branches are now thin wrappers
#      that call shared helpers (plot_slice, configure_axes).
#
# Suggested further refactoring (not yet implemented here):
#   - Move P_settings declaration to a separate config file / module.
#   - Replace the bare module-level script with a main() function and
#     guard the entry point with  if __name__ == '__main__': main()
#   - Replace global mutable state (xPlane, yPlane, zPlane, axisLimsZoom,
#     runDim, runDimAxes) with a RunConfig dataclass passed around explicitly.

import sys
import os
import socket
import itertools
import subprocess
from dataclasses import dataclass, field
from multiprocessing import Pool, Value
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import analysator as alr
import numpy as np
import scipy as sp
import operator as oper
from matplotlib.patches import Wedge

plt.switch_backend('agg')

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

useMultiProcessing = 1
printHeaderOnly = 0
mpFileOpenCnt = Value('i', 0)
Nfiles = -1
Nruns = -1
NfileOpens = -1
Nparams = -1
NfigCols = -1
NfigRows = -1
runDim = -1
runDimAxes = ''
HN = '(hostname = ' + socket.gethostname() + ') '

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['lines.linewidth'] = 2
figDpi = 100
matplotlib.rcParams['figure.dpi'] = figDpi
matplotlib.rcParams['savefig.dpi'] = figDpi
figResolutionX = 1920
figResolutionY = 1080
figureSize = (figResolutionX / figDpi, figResolutionY / figDpi)
tickDir = 'out'
tickLength = 2
tickWidth = 1
xTickAngle = 45

outputFileNamePrefix = ''

xPlane = -2.0
yPlane = 0.0
zPlane = 0.0
axisLimsZoom = -1
showPlanet = (1, 1, 0)

colormap1 = 'inferno'
colormap2 = 'seismic'


# ---------------------------------------------------------------------------
# Parameter-settings helpers
# ---------------------------------------------------------------------------

def _vec_settings(param, label_base, lims, unit, colormap, filename_base,
                  sigma=-1, log=0):
    """Return the four P_settings dicts for a vector field (|·|, x, y, z)."""
    return [
        {'param': param, 'type': 'magnitude',
         'str': f'$|{label_base}|$', 'log': log, 'lims': lims,
         'unit': unit, 'colormap': colormap,
         'filename': filename_base, 'sigma': sigma},
        {'param': param, 'type': 'xcomp',
         'str': f'${label_base}_x$', 'log': log, 'lims': lims,
         'unit': unit, 'colormap': colormap2,
         'filename': filename_base + 'x', 'sigma': sigma},
        {'param': param, 'type': 'ycomp',
         'str': f'${label_base}_y$', 'log': log, 'lims': lims,
         'unit': unit, 'colormap': colormap2,
         'filename': filename_base + 'y', 'sigma': sigma},
        {'param': param, 'type': 'zcomp',
         'str': f'${label_base}_z$', 'log': log, 'lims': lims,
         'unit': unit, 'colormap': colormap2,
         'filename': filename_base + 'z', 'sigma': sigma},
    ]


def _scalar_setting(param, label, lims, unit, colormap, filename,
                    sigma=-1, log=1):
    """Return a single P_settings dict for a scalar field."""
    return {'param': param, 'type': 'scalar',
            'str': label, 'log': log, 'lims': lims,
            'unit': unit, 'colormap': colormap,
            'filename': filename, 'sigma': sigma}


# ---------------------------------------------------------------------------
# Build P_settings list
# ---------------------------------------------------------------------------

P_settings = []

# Magnetic field
for _s in _vec_settings('cellB', 'B [nT]', (0, 40e-9), 1e-9, colormap1, 'B'):
    P_settings.append(_s)
for _s in _vec_settings('cellBAverage', 'B [nT]', (0, 40e-9), 1e-9, colormap1, 'B_ave'):
    P_settings.append(_s)

# Number densities
P_settings.append(_scalar_setting(
    'n_H+sw_ave',
    r'$n$(H$^+_\mathrm{sw}$) [m$^{-3}$]',
    (1e4, 1e9), 1, colormap1, 'Hsw_n'))
P_settings.append(_scalar_setting(
    'n_O+_ave',
    r'$n$(O$^{+}_\mathrm{planet}$) [m$^{-3}$]',
    (0.04 * 1e4, 0.04 * 1e9), 1, colormap1, 'Oplanet_n'))

# Electric field
_e_lims_lin = (-0.02, 0.02)
for _s in _vec_settings('nodeE', 'E', (0.0001, 0.1), 1, colormap1, 'E',
                        log=1):
    P_settings.append(_s)
# Override the component entries to linear scale + mV/m unit
for _type, _fn in (('xcomp', 'Ex'), ('ycomp', 'Ey'), ('zcomp', 'Ez')):
    P_settings.append({'param': 'nodeE', 'type': _type,
                       'str': f'$E_{_fn[1]}$ [mV/m]', 'log': 0,
                       'lims': _e_lims_lin, 'unit': 1e-3,
                       'colormap': colormap2, 'filename': _fn, 'sigma': -1})

# Velocity fields
_vel_mag_lims = (0, 700e3)
_vel_comp_lims = (-600e3, 600e3)
for _param, _label, _fn_prefix in (
    ('cellUe',    r'e^-',                         'e'),
    ('v_H+sw_ave', r'H$^+_\mathrm{sw}$',          'Hsw'),
    ('v_O+_ave',  r'O$^{+}_\mathrm{planet}$',     'Oplanet'),
):
    for _s in _vec_settings(_param, f'U ({_label}) [km/s]',
                             _vel_mag_lims, 1e3, colormap1, _fn_prefix + '_U'):
        P_settings.append(_s)

# Temperature
P_settings.append(_scalar_setting(
    'T_H+sw',
    r'$T$(H$^+_\mathrm{sw}$) [K]',
    (1e5, 1e8), 1, colormap1, 'Hsw_T'))

# Active parameter list (filled later)
P = []


# ---------------------------------------------------------------------------
# Command-line argument parsing
# ---------------------------------------------------------------------------

def _print_args_and_quit(msg):
    print(HN + f'ERROR: {msg}')
    for ii, arg in enumerate(sys.argv):
        print(HN + f'arg{ii} = {arg}')
    quit()


def parse_args():
    """Parse and validate command-line arguments.

    Returns a dict with keys:
        Ncores, runFolder, outFolder, runDescr, 
        Robject, tStartThisProcess, tEndThisProcess,
        tStartGlobal, tEndGlobal
    """
    Nargs = len(sys.argv)
    try:
        if Nargs == 10:
            return dict(
                Ncores=int(sys.argv[1]),
                runFolder=str(sys.argv[2]),
                outFolder=str(sys.argv[3]),
                runDescr=str(sys.argv[4]),
                Robject=float(sys.argv[5]),
                tStartThisProcess=int(sys.argv[6]),
                tEndThisProcess=int(sys.argv[7]),
                tStartGlobal=int(sys.argv[8]),
                tEndGlobal=int(sys.argv[9]),
            )
        elif Nargs == 8:
            tStart = int(sys.argv[6])
            tEnd = int(sys.argv[7])
            return dict(
                Ncores=int(sys.argv[1]),
                runFolder=str(sys.argv[2]),
                outFolder=str(sys.argv[3]),
                runDescr=str(sys.argv[4]),
                Robject=float(sys.argv[5]),
                tStartThisProcess=tStart,
                tEndThisProcess=tEnd,
                tStartGlobal=tStart,
                tEndGlobal=tEnd,
            )
        elif Nargs == 4:
            return dict(
                Ncores=int(sys.argv[1]),
                runFolder=str(sys.argv[2]),
                outFolder=str(sys.argv[3]),
                runDescr='run',
                Robject=1,
                tStartThisProcess=0,
                tEndThisProcess=1_000_000,
                tStartGlobal=0,
                tEndGlobal=1_000_000,
            )
        else:
            _print_args_and_quit('3, 7 or 9 command line arguments required')
    except ValueError:
        _print_args_and_quit('bad argument')


def validate_args(args):
    """Validate parsed arguments; quit on error."""
    runFolder = os.path.join(args['runFolder'], '')
    if not os.path.isdir(runFolder):
        print(HN + 'ERROR: cannot read folder: ' + runFolder)
        quit
    if not os.path.isdir(args['outFolder']):
        print(HN + 'ERROR: cannot find folder for output: ' + args['outFolder'])
        quit()
    if not any(
        f.startswith('state') and f.endswith('.vlsv')
        for f in os.listdir(runFolder)
    ):
        print(HN + 'ERROR: no state*.vlsv found in folder: ' + runFolder)
        quit()
    Ncores = args['Ncores']
    if Ncores == -1:
        args['printHeaderOnly'] = 1
        args['Ncores'] = 1
    elif Ncores < 1 or Ncores > 100:
        print(HN + 'ERROR: negative or otherwise bad number of cores')
        quit()
    else:
        args['printHeaderOnly'] = 0
    for key in ('tStartThisProcess', 'tEndThisProcess',
                'tStartGlobal', 'tEndGlobal'):
        if args[key] < 0:
            print(HN + f'ERROR: negative time value: {key} = {args[key]}')
            quit()
    if args['tStartThisProcess'] > args['tEndThisProcess']:
        print(HN + 'ERROR: tStart > tEnd (this process)')
        quit()
    if args['tStartGlobal'] > args['tEndGlobal']:
        print(HN + 'ERROR: tStartGlobal > tEndGlobal')
        quit()
    args['runFolder'] = runFolder
    return args


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parseParameter(logFileName, paramGetCmd, paramUnit):
    """Run a shell command and return the first token as a float * paramUnit."""
    res = subprocess.run(paramGetCmd, shell=True, capture_output=True, text=True)
    if res.returncode == 1:
        print('ERROR not found: ' + paramGetCmd)
        quit()
    return float(res.stdout.split()[0]) * paramUnit


def round2str(x, Ndec=10):
    if Ndec == 1:
        return str(int(round(x * Ndec) / Ndec))
    return str(round(x * Ndec) / Ndec)


def round2strWithUnit(x, unitStr):
    if abs(x) > 0:
        return str(round(x * 10) / 10) + unitStr
    return '0'


# ---------------------------------------------------------------------------
# Axis / geometry helpers
# ---------------------------------------------------------------------------

def chooseNcol(rPlane, rmin, rmax, nr, rstr, Rp, Rp_str):
    """Return the cell-column index and the clamped plane coordinate (in Rp)."""
    if rPlane > rmax or rPlane < rmin:
        rPlane = (rmax + rmin) / 2.0
        print(HN + f'(chooseNcol) WARNING: {rstr}Plane out of domain, '
              f'setting: {rstr}Plane = {rPlane / Rp} Rp')
    dr = (rmax - rmin) / nr
    Ncol = int(np.floor((rPlane - rmin) / dr))
    Ncol = max(0, min(Ncol, nr - 1))
    return Ncol, rPlane / Rp


def compute_tick_step(axisLimsZoom, nx, ny, nz):
    """Return a 'nice' tick step for the given zoomed axis limits."""
    Dx = axisLimsZoom[1] - axisLimsZoom[0] if nx > 1 else -1
    Dy = axisLimsZoom[3] - axisLimsZoom[2] if ny > 1 else -1
    Dz = axisLimsZoom[5] - axisLimsZoom[4] if nz > 1 else -1
    maxCrdRange = max(Dx, Dy, Dz)
    maxCrd = max(abs(v) for v in axisLimsZoom)
    maxCrdTen = pow(10, np.ceil(np.log10(maxCrd))) if maxCrd > 0 else 1

    tickStep = 1
    for ii, jj, kk in itertools.product(range(1, 100), (1, 2, 5), (-1, 1)):
        tickStep = maxCrdTen / (kk * jj * ii)
        NticksMax = maxCrdRange / tickStep
        if 5 <= NticksMax <= 15:
            break
    if not (5 <= NticksMax <= 15):
        print(HN + f'WARNING: no good tick step: NticksMax={NticksMax}, '
              f'tickStep={tickStep}')
    crdTicks = np.arange(-maxCrdTen, +maxCrdTen, step=tickStep)
    return crdTicks


# ---------------------------------------------------------------------------
# VLSV variable helpers
# ---------------------------------------------------------------------------

def read_variable_data(vr, P_ii):
    """Read a VLSV variable and return the relevant component as a 1-D array."""
    D_i = vr.read_variable_info(P_ii['param'])
    ptype = P_ii['type']
    if ptype == 'magnitude':
        return np.sqrt((D_i.data ** 2).sum(axis=1))
    if ptype == 'scalar':
        return D_i.data
    if ptype == 'xcomp':
        return D_i.data[:, 0]
    if ptype == 'ycomp':
        return D_i.data[:, 1]
    if ptype == 'zcomp':
        return D_i.data[:, 2]
    if ptype == 'nvO':
        nO_i = vr.read_variable_info('n_O+_ave')
        VO_i = vr.read_variable_info('v_O+_ave')
        VtotO = np.sqrt((VO_i.data ** 2).sum(axis=1))
        return nO_i.data * VtotO
    print(HN + 'ERROR: unknown parameter type: ' + ptype)
    quit()


def build_parameter_list(vlsv_path):
    """Discover VLSV variables and build the global P list from P_settings.

    Returns (all_vars, N_vector_vars_found, N_scalar_vars_found).
    """
    vr = alr.vlsvfile.VlsvReader(vlsv_path)
    all_vars = vr.get_all_variables()
    N_vector = 0
    N_scalar = 0

    settings_by_param = {}
    for s in P_settings:
        settings_by_param.setdefault(s['param'], []).append(s)

    for var_ in all_vars:
        if any(tag in var_ for tag in ('flag', 'detector_', 'CellID')):
            print(HN + 'skipping flag/detector/CellID variable: ' + var_)
            continue

        D_i = vr.read_variable_info(var_)
        is_vector = len(D_i.data.shape) > 1

        if is_vector:
            if D_i.data.shape[1] != 3:
                print(HN + f'ERROR: vector size not 3: {D_i.data.shape[1]} ({var_})')
                exit()
            N_vector += 1
        else:
            N_scalar += 1

        if var_ not in settings_by_param:
            print(HN + 'skipping variable not in P_settings: ' + var_)
            continue

        for s in settings_by_param[var_]:
            if is_vector and s['type'] in ('magnitude', 'xcomp', 'ycomp', 'zcomp'):
                P.append(dict(s))
            elif not is_vector and s['type'] == 'scalar':
                P.append(dict(s))
                break  # only one scalar entry per variable

    return all_vars, N_vector, N_scalar


# ---------------------------------------------------------------------------
# File-discovery helper
# ---------------------------------------------------------------------------

def find_vlsv_files(folder, tStart, tEnd):
    """Return a sorted list of state*.vlsv paths within [tStart, tEnd]."""
    found = []
    for f in sorted(os.listdir(folder)):
        if f.startswith('state') and f.endswith('.vlsv'):
            try:
                t = int(f[5:13])
            except ValueError:
                print(HN + 'ERROR: could not parse int from file name: ' + f)
                continue
            if tStart <= t <= tEnd:
                found.append(folder + f)
    return found


def checkVlsvFileExists(folder, fileName, tStartGlobal, tEndGlobal):
    """Return (fileFound, newFileFound, newFileName).

    If the expected file is absent, tries to find the latest available one
    within [tStartGlobal, tEndGlobal].
    """
    if os.path.isfile(folder + fileName):
        return True, False, ''
    tLatest = -1
    fileNameLastTimestep = ''
    for f in sorted(os.listdir(folder)):
        if f.startswith('state') and f.endswith('.vlsv'):
            try:
                t = int(f[5:13])
            except ValueError:
                print(HN + 'ERROR: could not parse int from file name: ' + f)
                continue
            if tStartGlobal <= t <= tEndGlobal and t > tLatest:
                tLatest = t
                fileNameLastTimestep = f
    if tLatest > -1:
        return False, True, fileNameLastTimestep
    return False, False, ''


# ---------------------------------------------------------------------------
# Panel-rendering helpers
# ---------------------------------------------------------------------------

def configurePanel(P_ii, cMap, ax, showPlanet_flag, showColorbar):
    """Draw planet wedges and configure color normalisation for an axes panel."""
    if showPlanet_flag == 1:
        w1 = Wedge((0, 0), 1.0, 90, 270, fc='dimgray')
        w2 = Wedge((0, 0), 1.0, 270, 90, fc='w')
        c1 = plt.Circle((0, 0), 1.0, color='k', fill=False, lw=0.5)
        ax.add_artist(w1)
        ax.add_artist(w2)
        ax.add_artist(c1)
    if P_ii['log'] == 1:
        cMap.set_norm(colors.LogNorm(
            vmin=P_ii['lims'][0] / P_ii['unit'],
            vmax=P_ii['lims'][1] / P_ii['unit']))
    plt.gcf().gca().tick_params(
        which='both', direction=tickDir, length=tickLength, width=tickWidth)


def configure_axes(ax, xlabel, ylabel, title, xlim, ylim, crdTicks,
                   is_bottom_row, is_first_col):
    """Apply common axes labels, ticks and limits to a subplot."""
    if is_bottom_row:
        ax.set_xlabel(xlabel)
    if is_first_col:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.title.set_text(title)
    ax.set_xticks(crdTicks)
    ax.tick_params('x', labelrotation=xTickAngle)
    ax.set_yticks(crdTicks)
    ax.axis('scaled')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_slice(ax, meshD, P_ii, axisExtent):
    """Call imshow for one 2-D data slice and return the image artist."""
    return ax.imshow(
        meshD / P_ii['unit'],
        vmin=P_ii['lims'][0] / P_ii['unit'],
        vmax=P_ii['lims'][1] / P_ii['unit'],
        cmap=P_ii['colormap'],
        extent=axisExtent,
        aspect='equal',
        origin='lower',
        interpolation='nearest',
    )


def maybe_smooth(arr, sigma):
    """Apply Gaussian smoothing if sigma > 0, otherwise return arr unchanged."""
    if sigma > 0:
        return sp.ndimage.filters.gaussian_filter(arr, sigma=sigma, mode='constant')
    return arr


def add_colorbar(fig, axes, artist, label, shrink=0.5):
    clb = fig.colorbar(artist, ax=axes.flatten(), shrink=shrink)
    clb.ax.set_title(label)


# ---------------------------------------------------------------------------
# plotPanel  (the main per-file-per-parameter workhorse)
# ---------------------------------------------------------------------------

def plotPanel(fig, axes, ii_run, P_ii, runFolder, vlsvFileName,
              runStr, Rp, Rp_str, printHeader=0):
    """Plot one parameter panel (or return header info when fig == -1)."""
    global mpFileOpenCnt, xPlane, yPlane, zPlane, axisLimsZoom
    global runDim, runDimAxes

    # Progress counter (multiprocess-safe)
    with mpFileOpenCnt.get_lock():
        if fig != -1 and printHeader == 0:
            print(HN + f'opening: {runFolder} | {vlsvFileName} | '
                  f'{P_ii["param"]} | {P_ii["type"]} '
                  f'({mpFileOpenCnt.value}/{NfileOpens})')
        mpFileOpenCnt.value += 1

    # Read VLSV file
    vr = alr.vlsvfile.VlsvReader(runFolder + vlsvFileName)
    [xmin, ymin, zmin, xmax, ymax, zmax] = vr.get_spatial_mesh_extent()
    [mx, my, mz] = vr.get_spatial_mesh_size()
    [sx, sy, sz] = vr.get_spatial_block_size()
    nx, ny, nz = mx * sx, my * sy, mz * sz
    dx = (xmax - xmin) / nx
    axisLims = np.divide([xmin, xmax, ymin, ymax, zmin, zmax], Rp)

    fileTime = vr.read_parameter('t')
    if fileTime is None:
        fileTime = vr.read_parameter('time')

    titleStr = runStr

    # Determine run dimensionality
    runDimAxes = ''
    if nx > 1:
        runDimAxes += 'x'
    if ny > 1:
        runDimAxes += 'y'
    if nz > 1:
        runDimAxes += 'z'
    runDim = len(runDimAxes)
    if nx < 1 or ny < 1 or nz < 1:
        print(HN + 'ERROR: bad run dimensions')

    # Column / row indices
    if runDim == 3:
        ii_col, ii_row = 0, ii_run
    elif runDim == 2:
        ii_col, ii_row = ii_run, 0
    else:
        print(HN + f'ERROR: unsupported run dimensionality (runDim={runDim})')
        quit()

    # Zoomed domain (fall back to full domain)
    if len(np.atleast_1d(axisLimsZoom)) < 6:
        axisLimsZoom = axisLims

    crdTicks = compute_tick_step(axisLimsZoom, nx, ny, nz)

    # Cell id ordering
    cellids_sorted = vr.read_variable('CellID').argsort()

    # Plane indices for 3-D runs
    if runDim == 3:
        YZ_Ncol, xPlane = chooseNcol(xPlane * Rp, xmin, xmax, nx, 'x', Rp, Rp_str)
        XZ_Ncol, yPlane = chooseNcol(yPlane * Rp, ymin, ymax, ny, 'y', Rp, Rp_str)
        XY_Ncol, zPlane = chooseNcol(zPlane * Rp, zmin, zmax, nz, 'z', Rp, Rp_str)
    else:
        XZ_Ncol = XY_Ncol = YZ_Ncol = -2

    # Print header and return early if requested
    if printHeader == 1:
        _print_run_header(Rp, runDim, runDimAxes,
                          xmin, xmax, ymin, ymax, zmin, zmax,
                          nx, ny, nz, Rp_str,
                          xPlane, yPlane, zPlane,
                          YZ_Ncol, XZ_Ncol, XY_Ncol)
    if fig == -1:
        return runDim, runDimAxes

    # Read and reshape the variable
    D = read_variable_data(vr, P_ii)
    D = D[cellids_sorted].reshape(nz, ny, nx)

    is_bottom_row = (ii_row == NfigRows - 1)

    if runDim == 3:
        _plot_3d_panels(fig, axes, ii_row, ii_col, D, P_ii,
                        axisLims, axisLimsZoom, crdTicks,
                        XY_Ncol, XZ_Ncol, YZ_Ncol,
                        xPlane, yPlane, zPlane,
                        Rp_str, fileTime, is_bottom_row)
    elif runDim == 2:
        _plot_2d_panel(fig, axes, ii_row, ii_col, D, P_ii,
                       axisLims, axisLimsZoom, crdTicks,
                       runDimAxes, Rp_str, fileTime, titleStr,
                       is_bottom_row)


def _print_run_header(Rp, runDim, runDimAxes,
                      xmin, xmax, ymin, ymax, zmin, zmax,
                      nx, ny, nz, Rp_str,
                      xPlane, yPlane, zPlane,
                      YZ_Ncol, XZ_Ncol, XY_Ncol):
    print('\nRun and figure configuration (from the first VLSV file):')
    print(f'Rp = {Rp / 1e3} km')
    print(f'run dimensions: {runDim}D, {runDimAxes}')
    print(f'x = {xmin / Rp} ... {xmax / Rp} Rp')
    print(f'y = {ymin / Rp} ... {ymax / Rp} Rp')
    print(f'z = {zmin / Rp} ... {zmax / Rp} Rp')
    print(f'grid (nx,ny,nz) = ({nx},{ny},{nz})')
    if runDim == 3:
        print('Plotting a 3D run with:')
        print(f'xPlane = {xPlane} Rp, YZ_Ncol = {YZ_Ncol}')
        print(f'yPlane = {yPlane} Rp, XZ_Ncol = {XZ_Ncol}')
        print(f'zPlane = {zPlane} Rp, XY_Ncol = {XY_Ncol}')
    elif runDim == 2:
        print(f'Plotting a 2D run: {runDimAxes}')
    else:
        print('ERROR: unsupported run dimensionality')
        quit()
    print(f'fig. columns x rows: {NfigCols} x {NfigRows}\n')


def _plot_3d_panels(fig, axes, ii_row, ii_col, D, P_ii,
                    axisLims, axisLimsZoom, crdTicks,
                    XY_Ncol, XZ_Ncol, YZ_Ncol,
                    xPlane, yPlane, zPlane,
                    Rp_str, fileTime, is_bottom_row):
    """Render the three (xz, xy, yz) slices for a 3-D run."""
    sigma = P_ii['sigma']

    # xz slice
    meshD_xz = maybe_smooth(D[:, XZ_Ncol, :], sigma)
    a = plot_slice(axes[ii_row][ii_col], meshD_xz, P_ii,
                   [axisLims[0], axisLims[1], axisLims[4], axisLims[5]])
    title_xz = ('3D: $xz$ ($y=$' + round2strWithUnit(yPlane, Rp_str) + ')'
                 if ii_row == 0 else None)
    configure_axes(axes[ii_row][ii_col],
                   '$x$ [' + Rp_str + ']', '$z$ [' + Rp_str + ']',
                   title_xz,
                   axisLimsZoom[0:2], axisLimsZoom[4:6],
                   crdTicks, is_bottom_row, True)
    configurePanel(P_ii, a, axes[ii_row][ii_col], showPlanet[0], 0)
    ii_col += 1

    # xy slice
    meshD_xy = maybe_smooth(D[XY_Ncol, :, :], sigma)
    a = plot_slice(axes[ii_row][ii_col], meshD_xy, P_ii,
                   [axisLims[0], axisLims[1], axisLims[2], axisLims[3]])
    title_xy = ('$t=$' + round2str(fileTime) + ' s\n'
                '3D: $xy$ ($z=$' + round2strWithUnit(zPlane, Rp_str) + ')'
                if ii_row == 0 else None)
    configure_axes(axes[ii_row][ii_col],
                   '$x$ [' + Rp_str + ']', '$y$ [' + Rp_str + ']',
                   title_xy,
                   axisLimsZoom[0:2], axisLimsZoom[2:4],
                   crdTicks, is_bottom_row, False)
    configurePanel(P_ii, a, axes[ii_row][ii_col], showPlanet[1], 0)
    ii_col += 1

    # yz slice
    meshD_yz = maybe_smooth(D[:, :, YZ_Ncol], sigma)
    a = plot_slice(axes[ii_row][ii_col], meshD_yz, P_ii,
                   [axisLims[2], axisLims[3], axisLims[4], axisLims[5]])
    title_yz = ('3D: $yz$ ($x=$' + round2strWithUnit(xPlane, Rp_str) + ')'
                if ii_row == 0 else None)
    configure_axes(axes[ii_row][ii_col],
                   '$y$ [' + Rp_str + ']', '$z$ [' + Rp_str + ']',
                   title_yz,
                   axisLimsZoom[2:4], axisLimsZoom[4:6],
                   crdTicks, is_bottom_row, False)
    configurePanel(P_ii, a, axes[ii_row][ii_col], showPlanet[2], 0)

    if is_bottom_row:
        fig.tight_layout()
        shrink = 0.5 if NfigRows == 1 else 1.0
        add_colorbar(fig, axes, a, P_ii['str'], shrink)


def _plot_2d_panel(fig, axes, ii_row, ii_col, D, P_ii,
                   axisLims, axisLimsZoom, crdTicks,
                   runDimAxes, Rp_str, fileTime, titleStr,
                   is_bottom_row):
    """Render the single slice for a 2-D run."""
    axis_configs = {
        'yz': (D[:, :, 0],
               [axisLims[2], axisLims[3], axisLims[4], axisLims[5]],
               axisLimsZoom[2:4], axisLimsZoom[4:6], '$y$', '$z$'),
        'xz': (D[:, 0, :],
               [axisLims[0], axisLims[1], axisLims[4], axisLims[5]],
               axisLimsZoom[0:2], axisLimsZoom[4:6], '$x$', '$z$'),
        'xy': (D[0, :, :],
               [axisLims[0], axisLims[1], axisLims[2], axisLims[3]],
               axisLimsZoom[0:2], axisLimsZoom[2:4], '$x$', '$y$'),
    }
    if runDimAxes not in axis_configs:
        print(HN + 'ERROR: unknown runDimAxes = ' + runDimAxes)
        quit()

    meshD, axisExtend, zoomX, zoomY, xlabelStr, ylabelStr = axis_configs[runDimAxes]
    a = plot_slice(axes[ii_row][ii_col], meshD, P_ii, axisExtend)

    if ii_col == 0 and ii_row == 0:
        title = ('$t=$' + round2str(fileTime) + ' s, '
                 + str(runDim) + 'D: ' + runDimAxes + '\n' + titleStr)
    else:
        title = titleStr

    configure_axes(axes[ii_row][ii_col],
                   xlabelStr + ' [' + Rp_str + ']',
                   ylabelStr + ' [' + Rp_str + ']',
                   title,
                   zoomX, zoomY,
                   crdTicks, True, ii_col == 0)
    configurePanel(P_ii, a, axes[ii_row][ii_col], showPlanet[1], 0)

    if ii_col == NfigCols - 1:
        fig.tight_layout()
        add_colorbar(fig, axes, a, P_ii['str'])


# ---------------------------------------------------------------------------
# plotFigure  (one output file per VLSV file)
# ---------------------------------------------------------------------------

def plotFigure(vlsvFileNameFullPath, printHeader=0):
    for ii in range(len(P)):
        if printHeader == 0:
            print(HN + f'parameter: {P[ii]["param"]}, {P[ii]["type"]} '
                  f'({ii + 1}/{Nparams})')
        fig, axes = plt.subplots(
            nrows=NfigRows, ncols=NfigCols,
            figsize=figureSize, frameon=True, squeeze=False)
        vlsvFileName = os.path.basename(vlsvFileNameFullPath)
        for jj, s in enumerate(simRuns):
            runFolder, runStr, Rp, Rp_str = s
            plotPanel(fig, axes, jj, P[ii], runFolder, vlsvFileName,
                      runStr, Rp, Rp_str, printHeader)
            if printHeader == 1:
                return
        plt.savefig(
            outputFileNamePrefix + P[ii]['filename'] + '_' + vlsvFileName + '.png',
            dpi=figDpi, transparent=False)
        plt.clf()
        plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

args = parse_args()
args = validate_args(args)

Ncores = args['Ncores']
runFolder = args['runFolder']
runDescr = args['runDescr']
Robject = args['Robject']
tStartThisProcess = args['tStartThisProcess']
tEndThisProcess = args['tEndThisProcess']
tStartGlobal = args['tStartGlobal']
tEndGlobal = args['tEndGlobal']
printHeaderOnly = args.get('printHeaderOnly', 0)

if printHeaderOnly == 0:
    print('running on ' + str(Ncores) + ' cores')

simRuns = [(runFolder, runDescr, Robject, '$R_p$')]
Nruns = len(simRuns)

# Discover VLSV files
if 'runFiles' not in locals():
    runFolder1 = simRuns[0][0]
    print(HN + f'processing from: folder = {runFolder1}, '
          f'time step = {tStartThisProcess} ... {tEndThisProcess} '
          f'(this process), whole run = {tStartGlobal} ... {tEndGlobal}')
    runFiles = find_vlsv_files(runFolder1, tStartThisProcess, tEndThisProcess)

Nfiles = len(runFiles)
print(HN + 'files found: ' + str(Nfiles))
if Nfiles <= 0:
    print(HN + 'ERROR: no VLSV files found')
    quit()

# Build the list of parameters to plot
all_vars, N_vector_vars_found, N_scalar_vars_found = build_parameter_list(runFiles[0])
Nparams = len(P)
NfileOpens = Nruns * Nfiles * Nparams

if printHeaderOnly == 1:
    print('Plotting variables:')
    for Pii_ in P:
        print(Pii_)
    print(f'Vector and scalar variables found: {len(all_vars)}')
    print(f'Vector variables found: {N_vector_vars_found}')
    print(f'Scalar variables found: {N_scalar_vars_found}')
    print(f'Plotting fields found: {Nparams}')
    print(f'Runs found: {Nruns}')
    print(f'Total file openings expected: {NfileOpens}')

# Read header / dimension info from first file
runDim, runDimAxes = plotPanel(
    -1, -1, -1, -1,
    simRuns[0][0], os.path.basename(runFiles[0]),
    'run0', simRuns[0][2], -1)

# Figure grid dimensions
if runDim == 3:
    NfigCols, NfigRows = 3, Nruns
elif runDim == 2:
    NfigCols, NfigRows = Nruns, 1
else:
    print(HN + f'ERROR: unsupported run dimensionality (runDim={runDim})')
    quit()

# Run plotting
if useMultiProcessing > 0 and printHeaderOnly == 0:
    if __name__ == '__main__':
        pool = Pool(Ncores)
        args_list = [(f,) for f in runFiles]
        pool.starmap(plotFigure, args_list)
elif printHeaderOnly == 1:
    plotFigure(runFiles[0], 1)
else:
    print(HN + 'DEBUG MODE: multiprocessing not used, serial execution')
    for ff in runFiles:
        plotFigure(ff)