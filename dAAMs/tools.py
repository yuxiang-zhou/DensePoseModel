import os
import sys
import glob
import uuid
import subprocess
import numpy as np
import menpo.io as mio
import scipy.io as sio

from .svs import SVS
from .icp import nicp, icp, SICP, SNICP
from .lineerror import interpolate
from .MatlabExecuter import MatlabExecuter
from .transforms import OpticalFlowTransform

from skimage import filters
from os.path import isfile
from matplotlib.path import Path as matpath
from scipy.spatial.distance import euclidean as dist

from menpo.image import Image, BooleanImage
from menpo.shape import TriMesh, PointCloud
from menpo.transform import Translation, AlignmentSimilarity

from menpofit.builder import build_reference_frame

def group_from_labels(lmg):
    # return [[<group>],[<group>],[<group>] ...]
    groups = []
    labels = lmg.items()
    lindex = 0

    for l in labels:
        g_size = l[1].n_points
        groups.append(range(lindex, lindex + g_size))
        lindex += g_size

    return groups


def minimum_distance(v, w, p, tolerance=1.0):
#     Return minimum distance between line segment (v,w) and point p
    l2 = dist(v, w)  # i.e. |w-v|^2 -  avoid a sqrt
    if l2 == 0.0:
        return dist(p, v)

#     Consider the line extending the segment, parameterized as v + t (w - v).
#     We find projection of point p onto the line.
#     It falls where t = [(p-v) . (w-v)] / |w-v|^2
    t = np.dot((p - v) / l2, (w - v) / l2)
    if t < 0.0:
        return dist(p, v) + tolerance      # // Beyond the 'v' end of the segment
    elif t > 1.0:
        return dist(p, w) + tolerance  # // Beyond the 'w' end of the segment

    projection = v + t * (w - v)  # // Projection falls on the segment
    return dist(p, projection)


def sample_points(target, range_x, range_y, edge=None):
    ret_img = Image.init_blank((range_x, range_y))

    if edge is None:
        edge = [range(len(target))]

    for eg in edge:
        for pts in interpolate(target[eg], 0.1):
            try:
                ret_img.pixels[0, pts[0], pts[1]] = 1
            except:
                print 'Index out of Bound'

    return ret_img


def FastNICP(sources, target):
    aligned = []
    corrs = []
    for source in sources:
        pts, _ = icp(source.points, target.points)
        mesh = TriMesh(pts)
        try:
            a_s, corr = nicp(mesh, target, us=2001, ls=1, step=100)
        except:
            print 'Failed Fast NICP'
            nicp_result = SNICP([PointCloud(pts)], target)
            corr = nicp_result.point_correspondence[0]
            a_s = nicp_result.aligned_shapes[0]
        aligned.append(a_s)
        corrs.append(corr)
    return aligned, corrs


def align_shapes(shapes, target_shape, lms_shapes=None, align_target=None):
    if align_target:
        lms_target = align_target

        forward_transform = [
            AlignmentSimilarity(ls, lms_target) for ls in lms_shapes
        ]
        aligned_shapes = np.array([
            t.apply(s) for t, s in zip(forward_transform, shapes)
        ])
        removed_transform = [t.pseudoinverse() for t in forward_transform]
        target_shape = align_target
        _icp = None

    else:
        # Align Shapes Using ICP
        _icp = SICP(shapes, target_shape)
        aligned_shapes = _icp.aligned_shapes
        # Store Removed Transform
        removed_transform = []
        forward_transform = []
        for a_s, s in zip(aligned_shapes, shapes):
            ast = AlignmentSimilarity(a_s, s)
            removed_transform.append(ast)
            icpt = AlignmentSimilarity(s, a_s)
            forward_transform.append(icpt)

    return aligned_shapes, target_shape, removed_transform, forward_transform, _icp


def _build_svs(svs_path_in, _norm_imgs, target_shape, aligned_shapes, align_t,
               reference_frame, _icp_transform, _is_mc=False, group=None,
               target_align_shape=None, _shape_desc='draw_gaussian'):
    svs_path_in = '{}'.format(svs_path_in)
    if not os.path.exists(svs_path_in):
        os.makedirs(svs_path_in)
    # Build Transform Using SVS
    xr, yr = reference_frame.shape

    if ((
            not glob.glob(svs_path_in + '/*.gif')
            and _is_mc)
            or (not glob.glob(svs_path_in + '/svs_*.png')
                and not _is_mc)):
        for j, (a_s, tr) in enumerate(
                zip(
                    [target_shape] + aligned_shapes.tolist(),
                    [AlignmentSimilarity(target_shape, target_shape)] + _icp_transform
                )
        ):
            print ("  - SVS Training {} out of {}".format(
                j, len(aligned_shapes) + 1)
            )
            # Align shapes with reference frame
            temp_as = align_t.apply(a_s)
            points = temp_as.points

            # Store SVS Landmarks
            svsLmsPath = '{}/svs_{:04d}_lms.pts'.format(svs_path_in, j)
            svsLms = target_align_shape if j == 0 else _norm_imgs[j-1].landmarks['align'].lms
            svsLms = align_t.apply(tr.apply(svsLms))
            if not os.path.exists(
                svsLmsPath
            ):
                tempRef = reference_frame.copy()
                tempRef.landmarks['temp'] = svsLms
                mio.export_landmark_file(tempRef.landmarks['temp'], svsLmsPath)

            # Construct tplt_edge
            tplt_edge = None
            lindex = 0
            # Get Grouped Landmark Indexes
            if j > 0:
                g_i = _norm_imgs[j-1].landmarks[group].items()
            else:
                g_i = _norm_imgs[j].landmarks[group].items()
                if not g_i[0][1].n_points == a_s.n_points:
                    g_i = [['Reference', a_s]]

            edge_g = []
            edge_ig = []
            for g in g_i:
                g_size = g[1].n_points
                rindex = g_size+lindex
                edges_range = np.array(range(lindex, rindex))
                edge_ig.append(edges_range)
                edges = np.hstack((
                    edges_range[:g_size-1, None], edges_range[1:, None]
                ))
                edge_g.append(edges)
                tplt_edge = edges if tplt_edge is None else np.vstack((
                    tplt_edge, edges
                ))
                lindex = rindex

            tplt_edge = np.concatenate(edge_g)
            #
            # Store SVS Image
            if _shape_desc == 'SVS':
                svs = SVS(
                    points, tplt_edge=tplt_edge, tolerance=3, nu=0.8,
                    gamma=0.8, max_f=20
                )
                store_image = svs.svs_image(range(xr), range(yr))
            elif _shape_desc == 'draw':
                store_image = sample_points(points, xr, yr, edge_ig)
            elif _shape_desc == 'draw_gaussian':
                ni = sample_points(points, xr, yr, edge_ig)
                store_image = Image.init_blank(ni.shape)
                store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
            elif _shape_desc == 'sample_gaussian':
                ni = Image.init_blank((xr, yr))
                for pts in points:
                    ni.pixels[0, pts[0], pts[1]] = 1
                store_image = Image.init_blank(ni.shape)
                store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
            elif _shape_desc == 'sample':
                store_image = Image.init_blank((xr, yr))
                for pts in points:
                    store_image.pixels[0, pts[0], pts[1]] = 1
            else:
                raise Exception('Undefined Shape Descriptor: {}'.format(_shape_desc))

            mio.export_image(
                store_image,
                '{}/svs_{:04d}.png'.format(svs_path_in, j),
                overwrite=True
            )

            # Train Group SVS
            for ii, g in enumerate(edge_ig):
                g_size = points[g].shape[0]
                edges_range = np.array(range(g_size))
                edges = np.hstack((
                    edges_range[:g_size-1, None], edges_range[1:, None]
                ))

                # Store SVS Image
                if _shape_desc == 'SVS':
                    svs = SVS(
                        points[g], tplt_edge=edges, tolerance=3, nu=0.8,
                        gamma=0.8, max_f=20
                    )
                    store_image = svs.svs_image(range(xr), range(yr))
                elif _shape_desc == 'draw':
                    store_image = sample_points(points[g], xr, yr)
                elif _shape_desc == 'draw_gaussian':
                    ni = sample_points(points[g], xr, yr, edge_ig)
                    store_image = Image.init_blank(ni.shape)
                    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                elif _shape_desc == 'sample_gaussian':
                    ni = Image.init_blank((xr, yr))
                    for pts in points[g]:
                        ni.pixels[0, pts[0], pts[1]] = 1
                    store_image = Image.init_blank(ni.shape)
                    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                elif _shape_desc == 'sample':
                    store_image = Image.init_blank((xr, yr))
                    for pts in points[g]:
                        store_image.pixels[0, pts[0], pts[1]] = 1
                else:
                    raise Exception('Undefined Shape Descriptor: {}'.format(_shape_desc))

                mio.export_image(
                    store_image,
                    '{}/svs_{:04d}.png'.format(svs_path_in, j),
                    overwrite=True
                )

            # Create gif from svs group
            #     convert -delay 10 -loop 0 svs_0001_g*.png test.gif
            subprocess.Popen([
                'convert',
                '-delay', '10', '-loop', '0',
                '{0}/svs_{1:04d}_g*.png'.format(svs_path_in, j),
                '{0}/svs_{1:04d}.gif'.format(svs_path_in, j)])


def rescale_images_to_reference_shape(images, group, reference_shape,
                                      verbose=False):
    r"""
    """
    _has_lms_align = False
    _n_align_points = None
    _is_mc = False
    group_align = group
    _db_path = images[0].path.parent
    reference_align_shape = reference_shape
    n_landmarks = reference_shape.n_points
    # Normalize the scaling of all images wrt the reference_shape size
    for i in images:
        if glob.glob('{}/*_lms.pts'.format(i.path.parent)):
            _has_lms_align = True
            i.landmarks['align'] = mio.import_landmark_file(
                '{}/{}_lms.pts'.format(i.path.parent, i.path.stem)
            )
            if not _n_align_points:
                _n_align_points = i.landmarks['align'].lms.n_points

    if _has_lms_align:
        group_align = 'align'
        reference_align_shape = PointCloud(
            reference_shape.points[:_n_align_points]
        )
        reference_shape = PointCloud(
            reference_shape.points[_n_align_points:]
        )

    normalized_images = [i.rescale_to_pointcloud(reference_align_shape, group=group_align)
                         for i in images]

    # Global Parameters
    alpha = 30
    pdm = 0
    lms_shapes = [i.landmarks['align'].lms for i in normalized_images]
    shapes = [i.landmarks[group].lms for i in normalized_images]
    n_shapes = len(shapes)


    # groups label
    sample_groups = group_from_labels(
        normalized_images[0].landmarks[group]
    )

    # Align Shapes Using ICP
    aligned_shapes, target_shape, _removed_transform, _icp_transform, _icp\
        = align_shapes(shapes, reference_shape, lms_shapes=lms_shapes, align_target=reference_align_shape)

    # Build Reference Frame from Aligned Shapes
    bound_list = []
    for s in aligned_shapes:
        bmin, bmax = s.bounds()
        bound_list.append(bmin)
        bound_list.append(bmax)
        bound_list.append(np.array([bmin[0], bmax[1]]))
        bound_list.append(np.array([bmax[0], bmin[1]]))
    bound_list = PointCloud(np.array(bound_list))

    reference_frame = build_reference_frame(bound_list, boundary=15)

    # Translation between reference shape and aliened shapes
    align_centre = target_shape.centre_of_bounds()
    align_t = Translation(
        reference_frame.centre() - align_centre
    )

    _rf_align = Translation(
        align_centre - reference_frame.centre()
    )

    # Set All True Pixels for Mask
    reference_frame.mask.pixels = np.ones(
        reference_frame.mask.pixels.shape, dtype=np.bool)

    # Mask Reference Frame
    # reference_frame.landmarks['sparse'] = reference_shape
    # self.reference_frame.constrain_mask_to_landmarks(group='sparse')

    # Get Dense Shape from Masked Image
    # dense_reference_shape = PointCloud(
    #     self.reference_frame.mask.true_indices()
    # )

    # Set Dense Shape as Reference Landmarks
    # self.reference_frame.landmarks['source'] = dense_reference_shape
    # self._shapes = shapes
    # self._aligned_shapes = []

    # Create Cache Directory
    home_dir = os.getcwd()
    dir_hex = uuid.uuid1()

    svs_path_in = _db_path if _db_path else '{}/.cache/{}/svs_training'.format(home_dir, dir_hex)
    svs_path_out = svs_path_in

    matE = MatlabExecuter()
    mat_code_path = '/vol/atlas/homes/yz4009/gitdev/mfsfdev'

    # Skip building svs is path specified
    _build_svs(svs_path_in, normalized_images, reference_shape, aligned_shapes, align_t,
               reference_frame, _icp_transform, _is_mc=_is_mc, group='PTS',
               target_align_shape=reference_align_shape, _shape_desc='draw_gaussian')


    # self._build_trajectory_basis(sample_groups, target_shape,
    #     aligned_shapes, dense_reference_shape, align_t)

    # Call Matlab to Build Flows
    if not isfile('{}/result.mat'.format(_db_path)):
        print('  - Building Shape Flow')
        matE.cd(mat_code_path)
        ext = 'gif' if _is_mc else 'png'
        isLms = 1
        isBasis = 0
        fstr = 'addpath(\'{0}/{1}\');' \
               'addpath(\'{0}/{2}\');' \
               'build_flow(\'{3}\', \'{4}\', \'{5}\', {6}, {7}, ' \
               '{8}, \'{3}/{9}\', {10}, {11}, {14}, {15}, {12}, \'{13}\')'.format(
                    mat_code_path, 'cudafiles', 'tools',
                    svs_path_in, svs_path_out, 'svs_%04d.{}'.format(ext),
                    0,
                    1, n_shapes, 'bas.mat',
                    alpha, pdm, 30, 'svs_%04d_lms.pts', isBasis, isLms
               )
        sys.stderr.write(fstr)
        p = matE.run_function(fstr)
        p.wait()
    else:
        svs_path_out = _db_path

    # Retrieve Results
    mat = sio.loadmat(
        '{}/result.mat'.format(svs_path_out)
    )

    _u, _v = mat['u'], mat['v']

    # Build Transforms
    print ("  - Build Transform")
    transforms = []
    for i in range(n_shapes):
        transforms.append(
            OpticalFlowTransform(_u[:, :, i], _v[:, :, i])
        )

    # build dense shapes
    print ("  - Build Dense Shapes")

    testing_points = reference_frame.mask.true_indices()
    close_mask = BooleanImage(matpath(
        align_t.apply(reference_shape).points
    ).contains_points(testing_points).reshape(
        reference_frame.mask.mask.shape
    ))
    reference_frame.mask = close_mask
    # self.reference_frame.constrain_mask_to_landmarks(group='sparse')

    # Get Dense Shape from Masked Image
    dense_reference_shape = PointCloud(
        np.vstack((
            align_t.apply(reference_align_shape).points,
            align_t.apply(reference_shape).points,
            reference_frame.mask.true_indices()
        ))
    )

    # Set Dense Shape as Reference Landmarks
    reference_frame.landmarks['source'] = dense_reference_shape
    dense_shapes = []
    for i, t in enumerate(transforms):
        warped_points = t.apply(dense_reference_shape)
        dense_shape = warped_points
        dense_shapes.append(dense_shape)

    ni = []
    for i, ds, t in zip(normalized_images, dense_shapes, _removed_transform):
        img = i.warp_to_shape(i.shape, _rf_align.compose_before(t), warp_landmarks=True)
        img.landmarks[group] = ds
        ni.append(img)

    return ni, transforms, reference_frame, n_landmarks, _n_align_points#, reference_frame, dense_reference_shape, reference_shape, testing_points,target_shape,align_t