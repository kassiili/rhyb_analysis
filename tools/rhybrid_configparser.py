import configparser
import io

class RhybridConfigParser(configparser.ConfigParser):
    
    _REQUIRED_CFG_ENTRIES = {
        "Simulation": ["time_initial", "maximum_timesteps", "dt", "data_save_interval", 
                       "data_save_interval_unit"],
        "Hybrid": ["R_object"]
    }
    
    # List of possible variables that are written particle-wise in the output (might not be up to date):
    _PARTICLE_OUTPUT_PARAMETERS = ['T', 'T_ave', 'n', 'n_ave', 'v', 'v_ave', 'nodeJi', 'nodeRhoQi']
    
    def __init__(self, *args, **kwargs):
        # Set default behaviour to non-strict to allow e.g. duplicate keys:
        if 'strict' not in kwargs:
            kwargs['strict'] = False
        super().__init__(*args, **kwargs)
    
    # Add default header for first items:
    def read_file(self, f, source=None):
        try:
            super().read_file(f, source)
        except configparser.MissingSectionHeaderError:
            content = f.read()
            sfile = io.StringIO("[Header]\n" + content)
            super().read_file(sfile, source)
            
        self._validate_config_file()

    def _validate_config_file(self):
        missing = {}
        for section, keys in self._REQUIRED_CFG_ENTRIES.items():
            if not self.has_section(section) and section != self.default_section:
                missing[section] = keys
                continue
            missing_keys = [k for k in keys if not self.has_option(section, k)]
            if missing_keys:
                missing[section] = missing_keys
        if missing:
            lines = [f"  [{s}]: {', '.join(keys)}" for s, keys in missing.items()]
            raise ValueError("Missing required config options:\n" + "\n".join(lines))
        
    def get_particle_sections(self):
        p_sec_names = [s.split('injector_')[1] for s in self.sections() if s.startswith('injector_')]
        return {s:dict(self.items(s)) for s in p_sec_names}
    
    def resolve_output_params(self) -> list[str]:
        """ Create a list of output variable names expected to be found in VLSV outputs. """
        
        p_names = [d['output_str'] for d in self.get_particle_sections().values()]
        out_params_base = set(self["Hybrid"]["output_parameters"].split())
        out_params = []
    
        # Just a help function:
        def list_insert(ls, i, v):
            ls_out = ls.copy()
            ls_out.insert(i, v)
            return ls_out
        
        for var in out_params_base:
            if var not in self._PARTICLE_OUTPUT_PARAMETERS:
                out_params.append(var)
                continue
            
            # For each particle type p and var X, add output params in format 'X_p(_ave)':
            var_parts = var.split('_')  # p should be of format 'X' or 'X_ave'
            out_params += ['_'.join(list_insert(var_parts, 1, p)) for p in p_names]
        
        return out_params

if __name__ == "__main__":
    config = RhybridConfigParser()
    with open("mars_example/mars.cfg") as f:
        config.read_file(f)
        
    p_secs = config.get_particle_sections()
    out_params = config.resolve_output_params()
    sim_setup = config["Simulation"]
    
