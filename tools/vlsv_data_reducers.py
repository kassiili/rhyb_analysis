import numpy as np

from analysator.vlsvfile import VlsvReader

def reduce_vslv_data(var_name, out_type, vr: VlsvReader):
    # Basic types:
    if out_type == 'scalar':
        return _scalar(vr.read_variable_info(var_name).data)
    if out_type == 'magnitude':
        return _magnitude(vr.read_variable_info(var_name).data)
    if out_type == 'xcomp':
        return _vec_component(0, vr.read_variable_info(var_name).data)
    if out_type == 'ycomp':
        return _vec_component(1, vr.read_variable_info(var_name).data)
    if out_type == 'zcomp':
        return _vec_component(2, vr.read_variable_info(var_name).data)
    
    # Custom types:
    if out_type == 'nvO':
        return _nvO(vr.read_variable_info('n_O+_ave').data,
                    vr.read_variable_info('v_O+_ave').data)
    
    raise ValueError('Unknown parameter type: ' + out_type)
    
def _scalar(data):
    return data

def _magnitude(data):
    return np.sqrt((data ** 2).sum(axis=1))

def _vec_component(i, data):
    return data[:, i]

def _nvO(nO_i, VO_i):
    """ Oxygen ion momentum density per ion mass. """
    VtotO = np.sqrt((VO_i ** 2).sum(axis=1))
    return nO_i * VtotO