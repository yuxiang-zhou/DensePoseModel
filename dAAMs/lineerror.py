__author__ = 'yz4009'

import numpy as np
from scipy.interpolate import interp1d
from menpo.shape import PointCloud


def arclen_polyl(cnt):

    tang = np.diff(cnt, axis=0)
    seg_len = np.sqrt(np.power(tang[:, 0], 2) + np.power(tang[:, 1], 2))
    seg_len = np.hstack((0, seg_len))
    alparam = np.cumsum(seg_len)
    cntLen = alparam[-1]
    return alparam, cntLen


def interpolate(points, step, kind='slinear'):
    alparam, cntLen = arclen_polyl(points)
    
    #[_,index_x]=np.unique(points[:, 0], return_index=True)
    #index_x = np.sort(index_x)
    #f_x = interp1d(
    #    alparam[index_x], points[index_x, 0], kind='cubic'
    #)
    
    #[_,index_y]=np.unique(points[:, 1], return_index=True)
    #index_y = np.sort(index_y)
    #f_y = interp1d(
    #    alparam[index_y], points[index_y, 1], kind='cubic'
    #)
    
    f_x = interp1d(
        alparam, points[:, 0], kind=kind
    )
    
    f_y = interp1d(
        alparam, points[:, 1], kind=kind
    )

    points_dense_x = f_x(np.arange(0, cntLen, step))
    points_dense_y = f_y(np.arange(0, cntLen, step))

    points_dense = np.hstack((
        points_dense_x[:, None], points_dense_y[:, None]
    ))

    return points_dense


def line_diff(l1, l2):

    Na = l1.shape[0]
    Nb = l2.shape[0]

    diffs_x = np.repeat(l1[:, 0][:, None], Nb, axis=1) \
        - np.repeat(l2[:, 0][None, :], Na, axis=0)
    diffs_y = np.repeat(l1[:, 1][:, None], Nb, axis=1) \
        - np.repeat(l2[:, 1][None, :], Na, axis=0)

    euclidien = diffs_x*diffs_x+diffs_y*diffs_y

    msd_ab = np.mean(np.min(euclidien, 0))
    msd_ba = np.mean(np.min(euclidien, 1))

    return msd_ab, msd_ba


def compute_line_error(pts1, pts2, gp):
    pts1 = pts1.points if isinstance(pts1, PointCloud) else pts1
    pts2 = pts2.points if isinstance(pts2, PointCloud) else pts2

    error = 0
    length = 0
    for g in gp:
        gl1 = pts1[g, :]
        gl2 = pts2[g, :]
        gl1 = interpolate(gl1, 0.5)
        gl2 = interpolate(gl2, 0.5)
        _, tl1 = arclen_polyl(gl1)
        _, tl2 = arclen_polyl(gl2)
        d1, d2 = line_diff(gl1, gl2)
        error += np.sum(np.sqrt([d1*tl1, d2*tl2])) / (tl1 + tl2)
        length += (tl1 + tl2) / 2
    return error / length