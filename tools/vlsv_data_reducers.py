import numpy as np

from analysator.vlsvfile import VlsvReader

def read_variable_data(vr: VlsvReader, var_name: str, var_type: str, order="CellID")\
        -> np.ma.MaskedArray:
    """ Read data from vlsv file and return as array.

    3D simulation grid data arrays are ordered as (x, y, z).
    """
    if order == "CellID":
        [mx, my, mz] = vr.get_spatial_mesh_size()
        [sx, sy, sz] = vr.get_spatial_block_size()
        nx, ny, nz = mx * sx, my * sy, mz * sz
        cell_id_order = vr.read_variable('CellID').argsort()

        # Order and reshape the array in compliance with Rhybrid convention:
        out = reduce_vslv_data(var_name, var_type, vr)[cell_id_order].reshape(nz, ny, nx)
    else:
        out = reduce_vslv_data(var_name, var_type, vr)

    # In Rhybrid snapshot files, the axes are ordered as (z,y,x) so we permute to (x,y,z):
    return out.transpose(2, 1, 0)

def reduce_vslv_data(var_name: str, out_type: str, vr: VlsvReader) -> np.ma.MaskedArray:
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
    
def _scalar(data: np.ma.MaskedArray) -> np.ma.MaskedArray:
    return data

def _magnitude(data: np.ma.MaskedArray) -> np.ma.MaskedArray:
    return np.ma.sqrt((data ** 2).sum(axis=1))

def _vec_component(i: int, data: np.ma.MaskedArray) -> np.ma.MaskedArray:
    return data[:, i]

def _nvO(n_o: np.ma.MaskedArray, v_o: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """ Oxygen ion momentum density per ion mass. """
    vtot_o = np.ma.sqrt((v_o ** 2).sum(axis=1))
    return n_o * vtot_o