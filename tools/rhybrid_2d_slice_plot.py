# plotter_rhybrid_2d_slice.py
#
# Plots 2-D slices of RHybrid simulation output (VLSV format).

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from slice_plot_tools import SlicePlot2D, SlicePlot3D, SlicePlotConfig

plt.switch_backend('agg')

def print_run_header(cfg: SlicePlotConfig):
    return

def print_parameter_header(cfg: SlicePlotConfig):
    return

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(param_paths: list[str], n_cores: int = 1, header_only: bool = False):
    configs = [SlicePlotConfig.from_toml(p) for p in param_paths]
    
    if header_only:
        for cfg in configs:
            print_run_header(cfg)
            print_parameter_header(cfg)
        return

    for cfg in configs:
        PlotterClass = SlicePlot2D if cfg.runs[0].config_params['domain']['n_dims'] == 2 \
                       else SlicePlot3D
        plotter = PlotterClass(cfg)
        steps = cfg.runs[0].get_steps_in_range(cfg.t_start, cfg.t_end)
        plotter.save_all(steps, n_cores=n_cores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--plotconfig", help="Plot configuration file", type=str,
                        default="./tools/config_2d_slice_plot/mars_example.toml")
    parser.add_argument("-np", "--nprocesses", help="Number of parallel processes", type=int,
                        default=1)
    args = parser.parse_args()
    
    cfg_path = str(Path(args.plotconfig).resolve())
    main([cfg_path], n_cores=args.nprocesses)
    