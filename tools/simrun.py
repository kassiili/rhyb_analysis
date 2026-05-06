import numpy as np
from pathlib import Path
from typing import Union
import re
from analysator.vlsvfile import VlsvReader

from rhybrid_configparser import RhybridConfigParser
from vlsv_data_reducers import reduce_vslv_data

class RhybridRun:
    
    _vlsv_name_pattern = re.compile(r'^state\d{8}\.vlsv$')
    
    def __init__(self, run_config: RhybridConfigParser, run_out_dir: str, run_descr: str=""):
        self._run_config = run_config
        self.config_params = self._read_run_config()
        
        self.run_out_dir = Path(run_out_dir).resolve()
        self.run_out_files = self._find_vlsv_files()
        self._validate_vlsv_files()
        
        self.run_descr = run_descr
        self.cell_id_order = {k: None for k in self.run_out_files.keys()}
                
    def _read_run_config(self):
        """ Read needed information from run config at initiation. """
        params = {}
        params["output_params"] = self._run_config.resolve_output_params()
        params["r_object"] = self._run_config["Hybrid"]["R_object"]
        
        # Read domain parameters:
        params['domain'] = {k: float(v) for k, v in self._run_config['LogicallyCartesian'].items()
                            if k in ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']}
        n_cells = {k: int(v) for k, v in self._run_config['LogicallyCartesian'].items()
                   if k in ['x_size', 'y_size', 'z_size']}
        params['domain'].update(n_cells)
        params['domain']['n_dims'] = sum([ax_size > 1 for ax_size in n_cells.values()])
        
        return params
                
    def _snap_time_to_file_name(t: Union[int, list[int]]) -> list[str]:
        if isinstance(t, int):
            t = [t]
        return [f'{ti:08d}' for ti in t]
                
    def _get_snap_times(self):
        return [i for i in range(0, int(self._run_config["Simulation"]["maximum_timesteps"]) + 1, 
                                 int(self._run_config["Simulation"]["data_save_interval"]))]
    
    def _find_vlsv_files(self) -> dict:
        """ Collect vlsv files and check that snap times match run config. """
        files_expected = [f"state{t:08d}.vlsv" for t in self._get_snap_times()]
        found = [f.name for f in sorted(self.run_out_dir.iterdir()) if self._vlsv_name_pattern.match(f.name)]
        
        if any([f not in found for f in files_expected]):
            raise FileNotFoundError(
                f"Some of the expected vlsv files were not found in {self.run_out_dir}. "
                + f"The following files were not found: {list(set(files_expected) - set(found))}"
            )
        
        print(f'Files found: {len(found)}')
        return {int(f.split("state")[1].split(".")[0]): VlsvReader(self.run_out_dir / f) for f in found}
    
    def _validate_vlsv_files(self):
        """ Check the vlsv file variables match run config. """
        
        # Check that all files contain the same variable set:
        var_sets = [set(vr.get_all_variables()) for vr in self.run_out_files.values()]
        if not all([si == var_sets[0] for si in var_sets]):
            raise ValueError("Inconsistent VLSV files: saved variables differ across files.")
        
        # Check that files contain configured outputs:
        if not all([set(si).issuperset(self.config_params['output_params']) for si in var_sets]):
            raise ValueError("Inconsistent VLSV files: configured output variables missing.")
        
        # Check domain:
        domain_keys_to_check = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max', 
                                'x_size', 'y_size', 'z_size']
        vlsv_domain_params = self._get_vlsv_domain_params()
        for k in domain_keys_to_check:
            if vlsv_domain_params[k] != self.config_params['domain'][k]:
                raise ValueError(
                    f'Inconsistent VLSV files: domain parameters do not match for {k}, '
                    + f'value in file {self.run_out_files[0].file_name}: {vlsv_domain_params[k]}, '
                    + f'value in config: {self.config_params['domain'][k]}.'
                )
        
        
    # TODO: write a function to get the needed domain characteristics for different use cases:
    def _get_vlsv_domain_params(self) -> dict:
        params = {}
        vr = self.run_out_files[0]
        
        [xmin, ymin, zmin, xmax, ymax, zmax] = vr.get_spatial_mesh_extent()
        params['x_min'] = xmin
        params['x_max'] = xmax
        params['y_min'] = ymin
        params['y_max'] = ymax
        params['z_min'] = zmin
        params['z_max'] = zmax                
        
        # Get total number of cells per axis over full domain:
        [mx, my, mz] = vr.get_spatial_mesh_size()
        [sx, sy, sz] = vr.get_spatial_block_size()
        nx, ny, nz = mx * sx, my * sy, mz * sz
        params['x_size'] = nx
        params['y_size'] = ny
        params['z_size'] = nz
        
        return params
        
    def read_variable_data(self, var_names: Union[str, list[str]], var_types: Union[str, list[str]],
                           steps: Union[int, list[int]], order="CellID") -> dict[np.ndarray]:
        """ Read data from vlsv files and return as arrays. 
        
        The first dimension of the returned arrays is the time step. Note that the same variable name
        may be repeated in *var_names*.
        """
        if isinstance(var_names, str):
            var_names = [var_names]
        if isinstance(var_types, str):
            var_types = [var_types]
        if isinstance(steps, int):
            steps = [steps]
            
        # Helper function:
        if order == "CellID":
            nx, ny, nz = [self.config_params['domain'][k] for k in ['x_size', 'y_size', 'z_size']]
            
            # Order and reshape the array in compliance with Rhybrid convention:
            def read_var(var, vtype, step): 
                return (reduce_vslv_data(var, vtype, self.run_out_files[step])
                        [self.get_cell_id_order(step)]
                        .reshape(nz, ny, nx))
        else:
            def read_var(var, vtype, step): 
                return reduce_vslv_data(var, vtype, self.run_out_files[step])
        
        out = {"step": np.array(steps)}
        for var, vtype in zip(var_names, var_types):
            out[".".join([var, vtype])] = np.array([read_var(var, vtype, s) for s in steps])

        return out
    
    def read_parameter(self, params: Union[str, list[str]], steps: Union[int, list[int]]
                       ) -> dict[np.ndarray]:
        if isinstance(params, str):
            params = [params]
        if isinstance(steps, str):
            steps = [steps]
            
        # Replace time parameter with the correct one (as historically there are two conventions):
        in_params = [
            'time' if p == 't' and not self.run_out_files[0].check_parameter('t')
            else 't' if p == 'time' and not self.run_out_files[0].check_parameter('time')
            else p for p in params
        ]
        
        out = {"step": np.array(steps)}
        for pi, po in zip(in_params, params):
            out[po] = np.array([self.run_out_files[s].read_parameter(pi) for s in steps])
        
        return out
            
    def get_cell_id_order(self, step):
        if self.cell_id_order[step] is None:
            self.cell_id_order[step] = self.run_out_files[step].read_variable('CellID').argsort()

        return self.cell_id_order[step]
    
    def get_steps_in_range(self, t_start: int, t_end: int):
        """ Get list of time steps with vlsv records between *t_start* and *t_end* (both inclusive). """
        all_steps = sorted(self.run_out_files.keys())
        return [s for s in all_steps if s >= t_start and s <= t_end]
        
    def describe_vlsv_files(self):
        return


if __name__ == "__main__":
    config = RhybridConfigParser()
    with open("mars_example/mars.cfg") as f:
        config.read_file(f)
    run = RhybridRun(config, "mars_example/sim_data")
    
    print(run.config_params)