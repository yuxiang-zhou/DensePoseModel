import os
import sys
import glob
import uuid
import copy
import math
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
from menpo.transform import Translation, AlignmentSimilarity, Scale, AlignmentRotation, AlignmentUniformScale
from menpo.visualize import print_dynamic

from menpofit.builder import build_reference_frame


def group_from_labels(lmg):
    # return [[<group>],[<group>],[<group>] ...]
    groups = []
    labels = lmg.items()
    lindex = 0

    for l in labels:
        if not l[0] == 'all':
            g_size = l[1].n_points
            groups.append(range(lindex, lindex + g_size))
            lindex += g_size

    groups = np.array(groups) if lindex > 0 else None

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


def sample_points(target, range_x, range_y, edge=None, x=0, y=0):
    ret_img = Image.init_blank((range_x, range_y))

    if edge is None:
        edge = [range(len(target))]

    for eg in edge:
        for pts in interpolate(target[eg], 0.1):
            try:
                ret_img.pixels[0, pts[0]-y, pts[1]-x] = 1
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
        print 'Using AlignmentSimilarity'
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
        print 'Using ICP'
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

    # Draw Mask
    mask_shape = mask_pc(align_t.apply(target_shape))
    mask_image = Image.init_blank((xr, yr))
    for pts in mask_shape.points:
        mask_image.pixels[0, pts[0], pts[1]] = 1
    mio.export_image(
        mask_image,
        '{}/ref_mask.png'.format(svs_path_in),
        overwrite=True
    )

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
            print_dynamic("  - SVS Training {} out of {}".format(
                j, len(aligned_shapes) + 1)
            )
            # Align shapes with reference frame
            temp_as = align_t.apply(a_s)
            points = temp_as.points

            # Store SVS Landmarks
            svsLmsPath = '{}/svs_{:04d}_lms.pts'.format(svs_path_in, j)
            svsLms = target_align_shape if j == 0 else _norm_imgs[j-1].landmarks['align'].lms
            svsLms = align_t.apply(tr.apply(svsLms))
            if not os.path.exists(svsLmsPath):
                tempRef = reference_frame.copy()
                tempRef.landmarks['temp'] = svsLms
                mio.export_landmark_file(tempRef.landmarks['temp'], svsLmsPath)

            img_label = _norm_imgs[j-1] if j > 0 else _norm_imgs[j]
            groups = group_from_labels(img_label.landmarks[group])
            store_image = Image.init_blank((xr,yr))
            ni = sample_points(points, xr, yr, groups)
            store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)

            # # Construct tplt_edge
            # tplt_edge = None
            # lindex = 0
            # # Get Grouped Landmark Indexes
            # if j > 0:
            #     g_i = _norm_imgs[j-1].landmarks[group].items()
            # else:
            #     g_i = _norm_imgs[j].landmarks[group].items()
            #     if not g_i[0][1].n_points == a_s.n_points:
            #         g_i = [['Reference', a_s]]
            #
            # edge_g = []
            # edge_ig = []
            # for g in g_i:
            #     g_size = g[1].n_points
            #     rindex = g_size+lindex
            #     edges_range = np.array(range(lindex, rindex))
            #     edge_ig.append(edges_range)
            #     edges = np.hstack((
            #         edges_range[:g_size-1, None], edges_range[1:, None]
            #     ))
            #     edge_g.append(edges)
            #     tplt_edge = edges if tplt_edge is None else np.vstack((
            #         tplt_edge, edges
            #     ))
            #     lindex = rindex
            #
            # tplt_edge = np.concatenate(edge_g)
            # print tplt_edge
            # #
            # # Store SVS Image
            # if _shape_desc == 'SVS':
            #     svs = SVS(
            #         points, tplt_edge=tplt_edge, tolerance=3, nu=0.8,
            #         gamma=0.8, max_f=20
            #     )
            #     store_image = svs.svs_image(range(xr), range(yr))
            # elif _shape_desc == 'draw':
            #     store_image = sample_points(points, xr, yr, edge_ig)
            # elif _shape_desc == 'draw_gaussian':
            #     ni = sample_points(points, xr, yr, edge_ig)
            #     store_image = Image.init_blank(ni.shape)
            #     store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
            # elif _shape_desc == 'sample_gaussian':
            #     ni = Image.init_blank((xr, yr))
            #     for pts in points:
            #         ni.pixels[0, pts[0], pts[1]] = 1
            #     store_image = Image.init_blank(ni.shape)
            #     store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
            # elif _shape_desc == 'sample':
            #     store_image = Image.init_blank((xr, yr))
            #     for pts in points:
            #         store_image.pixels[0, pts[0], pts[1]] = 1
            # else:
            #     raise Exception('Undefined Shape Descriptor: {}'.format(_shape_desc))

            mio.export_image(
                store_image,
                '{}/svs_{:04d}.png'.format(svs_path_in, j),
                overwrite=True
            )
            #
            # # Train Group SVS
            # for ii, g in enumerate(edge_ig):
            #     g_size = points[g].shape[0]
            #     edges_range = np.array(range(g_size))
            #     edges = np.hstack((
            #         edges_range[:g_size-1, None], edges_range[1:, None]
            #     ))
            #
            #     # Store SVS Image
            #     if _shape_desc == 'SVS':
            #         svs = SVS(
            #             points[g], tplt_edge=edges, tolerance=3, nu=0.8,
            #             gamma=0.8, max_f=20
            #         )
            #         store_image = svs.svs_image(range(xr), range(yr))
            #     elif _shape_desc == 'draw':
            #         store_image = sample_points(points[g], xr, yr)
            #     elif _shape_desc == 'draw_gaussian':
            #         ni = sample_points(points[g], xr, yr, edge_ig)
            #         store_image = Image.init_blank(ni.shape)
            #         store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
            #     elif _shape_desc == 'sample_gaussian':
            #         ni = Image.init_blank((xr, yr))
            #         for pts in points[g]:
            #             ni.pixels[0, pts[0], pts[1]] = 1
            #         store_image = Image.init_blank(ni.shape)
            #         store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
            #     elif _shape_desc == 'sample':
            #         store_image = Image.init_blank((xr, yr))
            #         for pts in points[g]:
            #             store_image.pixels[0, pts[0], pts[1]] = 1
            #     else:
            #         raise Exception('Undefined Shape Descriptor: {}'.format(_shape_desc))
            #
            #     mio.export_image(
            #         store_image,
            #         '{}/svs_{:04d}.png'.format(svs_path_in, j),
            #         overwrite=True
            #     )

            # Create gif from svs group
            #     convert -delay 10 -loop 0 svs_0001_g*.png test.gif

            # subprocess.Popen([
            #     'convert',
            #     '-delay', '10', '-loop', '0',
            #     '{0}/svs_{1:04d}_g*.png'.format(svs_path_in, j),
            #     '{0}/svs_{1:04d}.gif'.format(svs_path_in, j)])


def rescale_images_to_reference_shape(images, group, reference_shape,
                                        tight_mask=True,
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
        if 'LMS' in i.landmarks.keys():
            _has_lms_align = True
            i.landmarks['align'] = i.landmarks['LMS']
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

    scales = np.max(bound_list.points, axis=0) - np.min(bound_list.points, axis=0)
    max_scale = np.max(scales)
    bound_list = PointCloud(np.array([
        [max_scale, max_scale], [max_scale, 0], [0, max_scale], [0, 0]
    ]))

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

    # Create Cache Directory
    home_dir = os.getcwd()
    dir_hex = uuid.uuid1()

    svs_path_in = _db_path if _db_path else '{}/.cache/{}/svs_training'.format(home_dir, dir_hex)
    svs_path_out = svs_path_in

    matE = MatlabExecuter()
    mat_code_path = '/vol/atlas/homes/yz4009/gitdev/mfsfdev'

    # Skip building svs is path specified
    _build_svs(svs_path_in, normalized_images, reference_shape, aligned_shapes, align_t,
               reference_frame, _icp_transform, _is_mc=_is_mc, group=group,
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
    ref_sparse_lms = align_t.apply(reference_shape)
    close_mask = BooleanImage(matpath(
        ref_sparse_lms.points
    ).contains_points(testing_points).reshape(
        reference_frame.mask.mask.shape
    ))

    if tight_mask:
        reference_frame.mask = close_mask
    else:
        reference_frame.landmarks['sparse'] = ref_sparse_lms
        reference_frame.constrain_mask_to_landmarks(group='sparse')

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
        img = i.warp_to_shape(reference_frame.shape, _rf_align.compose_before(t), warp_landmarks=True)
        img.landmarks[group] = ds
        ni.append(img)

    return ni, transforms, reference_frame, n_landmarks, _n_align_points, [
        # _rf_align, _removed_transform, aligned_shapes, target_shape,
        # reference_frame, dense_reference_shape, testing_points,
        # align_t, normalized_images, shapes, lms_shapes,
        # reference_shape, reference_align_shape
    ]


def mask_pc(pc):
    t = Translation(-pc.centre())
    p = t.apply(pc)
    (y1,x1),(y2,x2) = p.bounds()

    a, b = np.meshgrid(
        np.arange(np.floor(y1), np.ceil(y2)),
        np.arange(np.floor(x1), np.ceil(x2))
    )

    mask = np.vstack([a.flatten(), b.flatten()]).T

    return PointCloud(
        t.pseudoinverse().apply(
            mask[matpath(p.points).contains_points(mask)]
        ).astype(int)
    )


def pc_intersection(pts1, pts2):
    intersects = []
    lst1 = pts1.points.tolist()
    lst2 = pts2.points.tolist()
    for p in lst1:
        if p in lst2:
            intersects.append(p)
    return PointCloud(np.array(intersects))


def flip_aam(aam):
    # flip reference shape
    faam = copy.deepcopy(aam)
    faam.reference_shape = Scale([1,-1]).apply(aam.reference_shape)

    # flip models
    for sm, am, ps in zip(faam.shape_models, faam.appearance_models, aam.patch_shape):
        # flip shape model mean
        sm.mean_vector = Scale([1,-1]).apply(sm.mean()).points.flatten()
        # flip appearance model mean
        img = am.mean()
        am.mean_vector = img.pixels[:,:,:,:,-1::-1].flatten()
        # flip shape model components
        ncomponents, _ = sm._components.shape
        sc = sm._components.reshape(ncomponents, -1, 2)
        sc[:, :, 1] *= -1
        sm._components = sc.reshape(ncomponents, -1)
        # flip appearance components
        ncomponents, _ = am._components.shape
        am._components = am._components.reshape((ncomponents,)+img.pixels.shape)[:,:,:,:,:,-1::-1].reshape(ncomponents,-1)
    return faam


def rotation_alignment_angle_ccw(pts1, pts2):
    t = AlignmentRotation(
        Translation(-pts1.centre()).apply(pts1),
        Translation(-pts2.centre()).apply(pts2)
    )

    return 360 - math.degrees(t.axis_and_angle_of_rotation()[-1])


def line_to_sparse(training_images, rs, group='PTS'):
    ni, transforms, reference_frame, n_landmarks, _n_align_points, [
        _rf_align, _removed_transform, aligned_shapes, target_shape,
        reference_frame, dense_reference_shape, testing_points,
        align_t, normalized_images, shapes, lms_shapes,
        reference_shape, reference_align_shape
    ] = rescale_images_to_reference_shape(training_images,group, rs)

    for i,n,t,norm in zip(training_images,ni,_removed_transform,normalized_images):
        scale = AlignmentUniformScale(norm.landmarks[group].lms, i.landmarks[group].lms)
        pts = PointCloud(n.landmarks[group].lms.points[:rs.n_points])
        i.landmarks['SPARSE'] = scale.apply(t.apply(_rf_align.apply(pts)))

    return training_images


from dAAMs.gridview import zero_flow_grid_pcloud, grid_triangulation
def subsample_dense_pointclouds(dense_pclouds, shape, mask=None, sampling=6,
                               pclouds_is_grid=False):
   if mask and not np.all(np.allclose(shape, mask.shape)):
       raise ValueError('Shape must match mask shape if masked is passed')

   if mask is not None:
       zero_pcloud = zero_flow_grid_pcloud(shape)
       zero_points = zero_pcloud.points.reshape(shape + (2,))
       zero_points = zero_points[::sampling, ::sampling, :]
       zero_points = zero_points.reshape([-1, 2])
       sampled = mask.sample(zero_points, order=0).ravel()

   pclouds = []
   for p in dense_pclouds:
       if not pclouds_is_grid:
           im = MaskedImage.init_blank(shape, mask=mask, n_channels=2)
           pixels = im.from_vector(p.points.T.ravel()).rolled_channels()
       else:
           pixels = p.points.reshape(shape + (2,))
       pixels = pixels[::sampling, ::sampling, :]
       points = pixels.reshape([-1, 2])
       if mask is not None:
           pcloud = PointCloud(points[sampled, :])
       else:
           pcloud = PointCloud(points)
       pclouds.append(pcloud)
   return pclouds, sampled


def grid_view(daams, fitter, final_shape, image=None, angle=0, indexes=None, trilist=None):
    t = fitter.algorithms[-1]
    t = t.transform

    t.set_target(final_shape)

    sub,sampled=subsample_dense_pointclouds(
        [PointCloud(daams.reference_frame.landmarks['source'].lms.points[daams.n_landmarks:])],
        daams.reference_frame.shape, mask=daams.reference_frame.mask
    )

    index = []
    for p in sub[0].points:
        for i, pp in enumerate(daams.reference_frame.landmarks['source'].lms.points):
            if np.all(p == pp):
                index.append(i)

    if indexes:
        index = indexes + index
    else:
        index = range(daams.n_landmarks) + index

    tm = TriMesh(daams.reference_frame.landmarks['source'].lms.points[index])

    if not trilist is None:
        tm_fit = TriMesh(t.target.points[index], trilist)
    else:
        tm_fit = TriMesh(t.target.points[index], tm.trilist)


    if image:
        image = build_reference_frame(final_shape)
        image.landmarks['DENSE'] = t.target
        img = image

        img.landmarks['trim'] = tm_fit
        img = img.rotate_ccw_about_centre(angle-360)
        img = img.crop_to_landmarks_proportion(1, group='trim')
        img.view_widget()
    else:
        tm_fit.view()
        return tm_fit, t.target
