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
from scipy.spatial.distance import euclidean

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
from menpo.transform import GeneralizedProcrustesAnalysis
from menpo.model import PCAModel
from menpo.feature import igo, hog, no_op, double_igo as digo, dsift
from menpofit.builder import build_reference_frame
from scipy.ndimage.morphology import distance_transform_edt

def normalise_image(img):
    ret_img = img.copy()
    hp = img.pixels
    nhp = (hp-np.min(hp))
    nhp = nhp/np.max(nhp)
    ret_img.pixels = nhp
    return ret_img


# binary based shape descriptor ----------------------------
def binary_shape(pc, xr, yr, groups=None):
    return sample_points(pc.points, xr, yr, groups)


def distance_transform_shape(pc, xr, yr, groups=None):
    rsi = binary_shape(pc, xr, yr, groups)
    ret = distance_transform_edt(rsi.rolled_channels().squeeze())

    return Image.init_from_rolled_channels(ret.T)


def shape_context_shape(pc, xr, yr, groups=None, sampls=None, r_inner=0.125, r_outer=2, nbins_r=5, nbins_theta=12):
    nbins = nbins_r*nbins_theta

    def get_angle(p1,p2):
        """Return angle in radians"""
        return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))

    def compute_one(pt, points, mean_dist):

        distances = np.array([euclidean(pt,p) for p in points])
        r_array_n = distances / mean_dist

        r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)

        r_array_q = np.zeros(len(points))
        for m in xrange(nbins_r):
           r_array_q +=  (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        def _get_angles(self, x):
            result = zeros((len(x), len(x)))
            for i in xrange(len(x)):
                for j in xrange(len(x)):
                    result[i,j] = get_angle(x[i],x[j])
            return result

        theta_array = np.array([get_angle(pt,p)for p in points])
        # 2Pi shifted
        theta_array_2 = theta_array + 2*math.pi * (theta_array < 0)
        theta_array_q = 1 + np.floor(theta_array_2 /(2 * math.pi / nbins_theta))

        sn = np.zeros((nbins_r, nbins_theta))
        for j in xrange(len(points)):
            if (fz[j]):
                sn[r_array_q[j] - 1, theta_array_q[j] - 1] += 1

        return sn.reshape(nbins)


    rsi = binary_shape(pc, xr, yr, groups)
    pixels = rsi.pixels.squeeze()
    pts = np.argwhere(pixels > 0)
    if sampls:
        pts = pts[np.random.randint(0,pts.shape[0],sampls)]
    mean_dist = dist([0,0],[xr,yr]) / 2

    sc = np.zeros((xr,yr,nbins))

    for x in xrange(xr):
        for y in xrange (yr):
            sc[x,y,:] = compute_one(np.array([x,y]), pts, mean_dist)

    return Image.init_from_rolled_channels(sc)
# ----------------------------------------------------------


# SVS based Shape Descriptor -------------------------------
def svs_shape(pc, xr, yr, groups=None):
    store_image = Image.init_blank((xr,yr))
    ni = binary_shape(pc, xr, yr, groups)
    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
    return store_image


def hog_svs_shape(pc, xr, yr, groups=None):
    store_image = hog(svs_shape(pc, xr, yr, groups))
    return store_image


def sift_svs_shape(pc, xr, yr, groups=None):
    store_image = dsift(svs_shape(pc, xr, yr, groups))
    return store_image


def sample_gaussian_shape(pc, xr, yr, groups=None):
    ni = Image.init_blank((xr, yr))
    for pts in pc.points:
        ni.pixels[0, pts[0], pts[1]] = 1
    store_image = Image.init_blank(ni.shape)
    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 4)
    return store_image

# ------------------------------------------------------------

def sample_shape(pc, xr, yr, groups=None):
    store_image = Image.init_blank((xr, yr))
    for pts in pc.points:
        store_image.pixels[0, pts[0], pts[1]] = 1
    return store_image


def build_shape_model(shapes, max_components=None):
    r"""
    Builds a shape model given a set of shapes.

    Parameters
    ----------
    shapes: list of :map:`PointCloud`
        The set of shapes from which to build the model.
    max_components: None or int or float
        Specifies the number of components of the trained shape model.
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

    Returns
    -------
    shape_model: :class:`menpo.model.pca`
        The PCA shape model.
    """
    # centralize shapes
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source() for s in gpa.transforms]

    # build shape model
    shape_model = PCAModel(aligned_shapes)
    if max_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_components)

    return shape_model


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

    groups = np.array(groups) if lindex > 0 else [range(lmg.lms.n_points)]

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
        for pts in target:
            ret_img.pixels[0, pts[0]-y, pts[1]-x] = 1
    else:
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


def _build_shape_desc(sd_path_in, _norm_imgs, target_shape, aligned_shapes, align_t,
               reference_frame, _icp_transform, _is_mc=False, group=None,
               target_align_shape=None, _shape_desc=svs_shape, align_group='align', target_group=None):
    sd_path_in = '{}'.format(sd_path_in)
    if not os.path.exists(sd_path_in):
        os.makedirs(sd_path_in)
    # Build Transform Using SVS
    xr, yr = reference_frame.shape

    # Draw Mask
    # mask_shape = mask_pc(align_t.apply(target_shape))
    # mask_image = Image.init_blank((xr, yr))
    # for pts in mask_shape.points:
    #     mask_image.pixels[0, pts[0], pts[1]] = 1
    # mio.export_image(
    #     mask_image,
    #     '{}/ref_mask.png'.format(sd_path_in),
    #     overwrite=True
    # )

    if (not glob.glob(sd_path_in + '/sd_*.gif')):

        target_group = target_group if not target_group is None  else [range(target_shape.n_points)]
        for j, (a_s, tr, svsLms,groups) in enumerate(
                zip(
                    [target_shape] + aligned_shapes.tolist(),
                    [AlignmentSimilarity(target_shape, target_shape)] + _icp_transform,
                    [target_align_shape]+[ni.landmarks[align_group].lms for ni in _norm_imgs],
                    [target_group]+[group_from_labels(ni.landmarks[group]) for ni in _norm_imgs]
                )
        ):
            print_dynamic("  - Shape Descriptor Training {} out of {}".format(
                j, len(aligned_shapes) + 1)
            )
            # Align shapes with reference frame
            temp_as = align_t.apply(a_s)
            points = temp_as.points

            # Store SVS Landmarks
            svsLmsPath = '{}/sd_{:04d}_lms.pts'.format(sd_path_in, j)
            svsLms = align_t.apply(tr.apply(svsLms))
            if not os.path.exists(svsLmsPath):
                tempRef = reference_frame.copy()
                tempRef.landmarks['temp'] = svsLms
                mio.export_landmark_file(tempRef.landmarks['temp'], svsLmsPath)


            store_image = normalise_image(_shape_desc(temp_as, xr, yr, groups))

            # Create gif from svs group
            #     convert -delay 10 -loop 0 sd_0001_g*.png test.gif

            for ch in range(store_image.n_channels):
                channel_img = store_image.extract_channels(ch)
                mio.export_image(
                    channel_img,
                    '{}/sd_{:04d}_g{:02d}.png'.format(sd_path_in, j, ch),
                    overwrite=True
                )

            subprocess.Popen([
                'convert',
                '-delay', '10', '-loop', '0',
                '{0}/sd_{1:04d}_g*.png'.format(sd_path_in, j),
                '{0}/sd_{1:04d}.gif'.format(sd_path_in, j)
            ])


def rescale_images_to_reference_shape(images, group, reference_shape,
                                        tight_mask=True, sd=svs_shape, target_group=None,
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
    else:
        group_align = '_nicp'
        for i in images:
            source_shape = TriMesh(reference_shape.points)
            _, points_corr = nicp(source_shape, i.landmarks[group].lms)
            i.landmarks[group_align] = PointCloud(i.landmarks[group].lms.points[points_corr])

    print('  - Normalising')
    normalized_images = [i.rescale_to_pointcloud(reference_align_shape, group=group_align)
                         for i in images]

    # Global Parameters
    alpha = 30
    pdm = 0
    lms_shapes = [i.landmarks[group_align].lms for i in normalized_images]
    shapes = [i.landmarks[group].lms for i in normalized_images]
    n_shapes = len(shapes)


    # Align Shapes Using ICP
    aligned_shapes, target_shape, _removed_transform, _icp_transform, _icp\
        = align_shapes(shapes, reference_shape, lms_shapes=lms_shapes, align_target=reference_align_shape)
    # Build Reference Frame from Aligned Shapes
    bound_list = []
    for s in [reference_shape] + aligned_shapes.tolist():
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

    sd_path_in = '{}/shape_discriptor'.format(_db_path) if _db_path else '{}/.cache/{}/sd_training'.format(home_dir, dir_hex)
    sd_path_out = sd_path_in

    matE = MatlabExecuter()
    mat_code_path = '/vol/atlas/homes/yz4009/gitdev/mfsfdev'

    # Skip building svs is path specified
    _build_shape_desc(sd_path_in, normalized_images, reference_shape, aligned_shapes, align_t,
               reference_frame, _icp_transform, _is_mc=_is_mc, group=group,
               target_align_shape=reference_align_shape, _shape_desc=sd,
               align_group=group_align, target_group=target_group
               )


    # self._build_trajectory_basis(sample_groups, target_shape,
    #     aligned_shapes, dense_reference_shape, align_t)

    # Call Matlab to Build Flows
    if not isfile('{}/result.mat'.format(sd_path_in)):
        print('  - Building Shape Flow')
        matE.cd(mat_code_path)
        ext = 'gif'
        isLms = _has_lms_align + 0
        isBasis = 0
        fstr =  'gpuDevice(1);' \
                'addpath(\'{0}/{1}\');' \
                'addpath(\'{0}/{2}\');' \
                'build_flow(\'{3}\', \'{4}\', \'{5}\', {6}, {7}, ' \
                '{8}, \'{3}/{9}\', {10}, {11}, {14}, {15}, {12}, \'{13}\')'.format(
                    mat_code_path, 'cudafiles', 'tools',
                    sd_path_in, sd_path_out, 'sd_%04d.{}'.format(ext),
                    0,
                    1, n_shapes, 'bas.mat',
                    alpha, pdm, 30, 'sd_%04d_lms.pts', isBasis, isLms
               )
        sys.stderr.write(fstr)
        sys.stderr.write(fstr.replace('build_flow','build_flow_test'))
        p = matE.run_function(fstr)
        p.wait()
    else:
        sd_path_out = sd_path_in

    # Retrieve Results
    mat = sio.loadmat(
        '{}/result.mat'.format(sd_path_out)
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

    return ni, transforms, reference_frame, n_landmarks, _n_align_points, _removed_transform, normalized_images, _rf_align, reference_shape, [
        align_t
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


def line_to_sparse(training_images,sparse_shape, rs, group='PTS', sd='draw_gaussian'):
    ni, icp_transforms, reference_frame, n_landmarks, _n_align_points, _removed_transform, normalized_images, _rf_align, rs, [align_t] = rescale_images_to_reference_shape(training_images,group, rs, sd='sample_gaussian')

    path_to_db = '{}'.format(training_images[0].path.parent)

    # Retrieve Results
    mat = sio.loadmat(
        '{}/result.mat'.format(path_to_db)
    )

    _u, _v = mat['u'], mat['v']

    # Build Transforms
    print ("  - Build Transform")
    transforms = []
    for i in range(_u.shape[-1]):
        transforms.append(
            OpticalFlowTransform(_u[:, :, i], _v[:, :, i])
            )

    for i,n,t,norm, icpt, oft in zip(training_images,ni,_removed_transform, normalized_images, icp_transforms, transforms):
        scale = AlignmentUniformScale(norm.landmarks[group].lms, i.landmarks[group].lms)
        pts = oft.apply(align_t.apply(sparse_shape))
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
