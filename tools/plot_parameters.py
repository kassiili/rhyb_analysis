"""plot_parameters.py
Loads and validates the plot-parameter registry from a toml file. Parameter
entries are searched from [[parameter]] sections (an array of tables). Each
parameter entry describes one quantity to be plotted and  must include the 
following set of keys:

    param     - variable name as it appears in the VLSV file
    type      - 'scalar' | 'magnitude' | 'xcomp' | 'ycomp' | 'zcomp' 
                | (custom definition implemented in vlsv_data_reducers.py)
    str       - colorbar / panel label (LaTeX accepted)
    log       - use logarithmic colour scale: 0 = linear, 1 = log
    lims      - (vmin, vmax) in the *original* VLSV units (before unit scaling)
    unit      - divide the raw data by this before plotting
                (i.e. 1e-9 turns Tesla into nT, 1e3 turns m/s into km/s)
    colormap  - any Matplotlib colormap name
    filename  - output PNG filename prefix (no path, no extension)
    sigma     - Gaussian smoothing std-dev passed to gaussian_filter;
                set to -1 to disable smoothing

Usage
-----
    from plot_parameters import load_parameter_settings

    P_settings = load_parameter_settings()          # uses default path
    P_settings = load_parameter_settings('my.toml') # custom path
"""

import sys
import os

# tomllib is in the standard library from Python 3.11 onward.
# For older interpreters fall back to the third-party tomli package.
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python < 3.11 requires the 'tomli' package: pip install tomli"
        )
        
from analysator.vlsvfile import VlsvReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOML = os.path.join(os.path.dirname(__file__), 'config_2d_slice_plot/mars_example.toml')

_VALID_PARAM_TYPES = {'magnitude', 'scalar', 'xcomp', 'ycomp', 'zcomp'}

_REQUIRED_PARAM_KEYS = {'param', 'type', 'label', 'log', 'lims', 'unit',
                        'colormap', 'filename', 'sigma'}

class PlotParams:
    
    def __init__(self, toml_path: str = _DEFAULT_TOML):
        """Read *toml_path* and validate the list of parameter-setting dicts.

        Each returned dict has the keys in _REQUIRED_PARAM_KEYS expected by the plotter.

        The TOML key 'label' is renamed to 'str' to match the plotter's
        internal convention.

        Raises
        ------
        FileNotFoundError
            If *toml_path* does not exist.
        ValueError
            If any parameter block fails validation.
        """
        self.toml_path = toml_path
        with open(toml_path, 'rb') as fh:
            raw = tomllib.load(fh)
            
        entries = raw.get('parameter', [])
        if not entries:
            raise ValueError(f'plot_parameters: no [[parameter]] blocks found in {toml_path}')

        self.params = []
        for i, entry in enumerate(entries):
            self._validate_param_entry(entry, i)
            self.params.append(self._normalise(entry))
           
        
    def _validate_param_entry(self, entry: dict, index: int) -> None:
        """Raise ValueError with a clear message if *entry* is malformed."""
        missing = _REQUIRED_PARAM_KEYS - entry.keys()
        if missing:
            raise ValueError(
                f'plot_parameters: [[parameter]] block {index} in {self.toml_path} '
                f'is missing keys: {sorted(missing)}'
            )

        ptype = entry['type']
        if ptype not in _VALID_PARAM_TYPES:
            raise ValueError(
                f'plot_parameters: block {index} has unknown type {ptype!r}. '
                f'Valid types: {sorted(_VALID_PARAM_TYPES)}'
            )

        lims = entry['lims']
        if len(lims) != 2:
            raise ValueError(
                f'plot_parameters: block {index} ({entry["param"]}/{ptype}): '
                f"'lims' must be a 2-element list, got {lims!r}"
            )
        if lims[0] >= lims[1]:
            raise ValueError(
                f'plot_parameters: block {index} ({entry["param"]}/{ptype}): '
                f"'lims[0]' must be less than 'lims[1]', got {lims!r}"
            )

        if entry['unit'] == 0:
            raise ValueError(
                f'plot_parameters: block {index} ({entry["param"]}/{ptype}): '
                f"'unit' must not be zero"
            )


    def _normalise(self, entry: dict) -> dict:
        """Convert a raw TOML dict to the internal plotter format."""
        return {
            'param':    entry['param'],
            'type':     entry['type'],
            'str':      entry['label'],   # rename 'label' → 'str'
            'log':      int(entry['log']),
            'lims':     tuple(entry['lims']),
            'unit':     float(entry['unit']),
            'colormap': entry['colormap'],
            'filename': entry['filename'],
            'sigma':    float(entry['sigma']),
        }
        
        
    def find_in_vlsv(self, vlsv_reader: VlsvReader) -> tuple[set, set]:
        """ Look for plot variables in vlsv file. Returns tuple with found and missing var names. """
        
        print(f"Searching vlsv file {vlsv_reader.file_name} for plot variables given in "
              f"{self.toml_path}.")
        
        vars_in_vlsv = vlsv_reader.get_all_variables()
        vars_in_self = self.get_all_variables()
        
        found = set(vars_in_self).intersection(set(vars_in_vlsv))
        missing = set(vars_in_self) - set(vars_in_vlsv)
        
        if any([var not in vars_in_vlsv for var in vars_in_self]):
            print(f"The following plot variables were not found in {vlsv_reader.file_name}: "
                  + f"{list(missing)}")
        else:
            print("All plot variables found in the VLSV file.")
        
        return (found, missing)
    
    def get_all_variables(self):
        return [entry["param"] for entry in self.params]
    
    def describe(self):
        # print('Plotting variables:')
        # for Pii_ in P:
        #     print(Pii_)
        # print(f'Vector and scalar variables found: {len(all_vars)}')
        # print(f'Vector variables found: {N_vector_vars_found}')
        # print(f'Scalar variables found: {N_scalar_vars_found}')
        # print(f'Plotting fields found: {Nparams}')
        # print(f'Runs found: {Nruns}')
        # print(f'Total file openings expected: {NfileOpens}')
        return 
    
    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)


if __name__ == "__main__":
    plot_params = PlotParams()
    
    print(plot_params.params)