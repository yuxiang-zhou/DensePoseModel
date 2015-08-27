from menpo.shape import PointCloud, TriMesh

import numpy as np


def grid_triangulation(shape):
    height, width = shape
    row_to_index = lambda x: x * width
    top_triangles = lambda x: np.concatenate([
        np.arange(row_to_index(x), row_to_index(x) + width - 1)[..., None],
        np.arange(row_to_index(x) + 1, row_to_index(x) + width)[..., None],
        np.arange(row_to_index(x + 1),
                  row_to_index(x + 1) + width - 1)[..., None]], axis=1)
 
    # Half edges are opposite directions
    bottom_triangles = lambda x: np.concatenate([
        np.arange(row_to_index(x + 1),
                  row_to_index(x + 1) + width - 1)[..., None],
        np.arange(row_to_index(x) + 1, row_to_index(x) + width)[..., None],
        np.arange(row_to_index(x + 1) + 1,
                  row_to_index(x + 1) + width)[..., None]], axis=1)
 
    trilist = []
    for k in xrange(height - 1):
        trilist.append(top_triangles(k))
        trilist.append(bottom_triangles(k))
        
    return np.concatenate(trilist)


def zero_flow_grid_pcloud(shape, triangulated=False, mask=None, grid_size=1):
    point_grid = np.meshgrid(range(0, shape[0], grid_size),
                             range(0, shape[1], grid_size), indexing='ij')
    point_grid_vec = np.vstack([p.ravel() for p in point_grid]).T
    # point_grid_im = point_grid_vec.reshape(shape + (2,))

    if triangulated:
        trilist = grid_triangulation(shape)
        pcloud = TriMesh(point_grid_vec, trilist=trilist)
    else:
        pcloud = PointCloud(point_grid_vec)
    
    if mask is not None:
        return pcloud.from_mask(mask.pixels.ravel())
    else:
        return pcloud