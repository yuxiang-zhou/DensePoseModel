from menpofit.aam import AAMBuilder
from menpofit.aam.lineerror import interpolate
from menpofit.base import create_pyramid
from menpofit.transform import DifferentiableThinPlateSplines
from menpo.transform.base import Transform
from menpofit.deformationfield import SVS
from menpofit.builder import normalization_wrt_reference_shape
from menpofit.deformationfield.MatlabExecuter import MatlabExecuter
from menpo.feature import igo
from menpo.image import Image, BooleanImage
from menpo.shape import PointCloud
from menpo.transform.icp import SICP, SNICP
from menpo.model import PCAModel
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.transform import Translation, AlignmentSimilarity
from menpo.transform.icp import nicp, icp
from menpo.shape import TriMesh
from scipy.spatial.distance import euclidean as dist
from skimage import filters
from os.path import isfile
from matplotlib.path import Path as matpath

import os
import sys
import glob
import uuid
import subprocess
import numpy as np
import menpo.io as mio
import scipy.io as sio


# Optical Flow Transform
class OpticalFlowTransform(Transform):
    def __init__(self, u, v):
        super(OpticalFlowTransform, self).__init__()
        self._u = v
        self._v = u

    def _apply(self, x, **kwargs):
        ret = x.copy()
        for p in ret:
            i, j = p[0].astype(int), p[1].astype(int)
            p += np.array([self._u[i, j], self._v[i, j]])
        return ret


# Deformation Field using ICP, NICP --------------
class DeformationFieldBuilder(AAMBuilder):
    def __init__(self, features=igo, transform=DifferentiableThinPlateSplines,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=False,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=0, template=0):
        super(DeformationFieldBuilder, self).__init__(
            features, transform, trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models, max_shape_components,
            max_appearance_components, boundary)
        self.template = template
        self._db_path = None

    def build(self, images, group=None, label=None, verbose=False,
              target_shape=None, db_path=None):
        r"""
        Builds a Multilevel Active Appearance Model from a list of
        landmarked images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the AAM.

        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        aam : :map:`AAM`
            The AAM object. Shape and appearance models are stored from lowest
            to highest level
        """

        self._db_path = db_path
        target_align_shape = target_shape
        group_align = group
        # compute reference_shape and normalize images size
        # search for *_lms.pts, use them to normalise if exist
        if glob.glob(self._db_path + '/*_lms.pts'):
            self._has_lms_align = True
            for i in images:
                i.landmarks['align'] = mio.import_landmark_file(
                    '{}/{}_lms.pts'.format(self._db_path, i.path.stem)
                )

            group_align = 'align'
            label = None
            target_align_shape = mio.import_landmark_file(
                '{}/{}_lms.pts'.format(
                    self._db_path, images[self.template].path.stem
                )
            ).lms

        self.reference_shape, normalized_images = \
            normalization_wrt_reference_shape(
                images, group_align, label, self.normalization_diagonal,
                target_align_shape, verbose
            )

        self._norm_imgs = normalized_images

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        appearance_models = []
        self._feature_images = []
        self._warpped_images = []
        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get feature images of current level
            feature_images = []
            for c, g in enumerate(generators):
                if verbose:
                    print_dynamic(
                        '{}Computing feature space/rescaling - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                feature_images.append(next(g))

            self._feature_images.append(feature_images)

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # define shapes that will be used for training
            if j == 0:
                original_shapes = shapes
                train_shapes = shapes
            else:
                if self.scaled_shape_models:
                    train_shapes = shapes
                else:
                    train_shapes = original_shapes

            # train shape model and find reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            if j == 0:
                shape_model = self._build_shape_model(
                    train_shapes, self.max_shape_components[rj],
                    target_shape
                )
            else:
                if self.scaled_shape_models:
                    shape_model = self._build_shape_model(
                        train_shapes, self.max_shape_components[rj],
                        target_shape
                    )
                else:
                    shape_model = shape_models[-1].copy()

            reference_frame = self._build_reference_frame(shape_model.mean())

            # add shape model to the list
            shape_models.append(shape_model)

            # compute transforms
            transforms = self._compute_transforms(reference_frame,
                                                  feature_images, group,
                                                  label, verbose, level_str)

            # warp images to reference frame
            warped_images = []
            for c, (i, t) in enumerate(zip(feature_images, transforms)):
                if verbose:
                    print_dynamic('{}Warping images - {}'.format(
                        level_str,
                        progress_bar_str(float(c + 1) / len(feature_images),
                                         show_bar=False)))
                si = self._image_pre_process(i, j, c)

                # mask = reference_frame.mask
                wimg = si.warp_to_mask(reference_frame.mask, t)

                warped_images.append(wimg)
            self._warpped_images.append(warped_images)

            # attach reference_frame to images' source shape
            for i in warped_images:
                i.landmarks['source'] = reference_frame.landmarks['source']

            # build appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components[rj] is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[rj])

            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        n_training_images = len(images)

        return self._build_aam(shape_models, appearance_models,
                               n_training_images)

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns a DeformationField object.

        Parameters
        ----------
        shape_models : :map:`PCAModel`
            The trained multilevel shape models.

        appearance_models : :map:`PCAModel`
            The trained multilevel appearance models.

        n_training_images : `int`
            The number of training images.

        Returns
        -------
        aam : :map:`DeformationField`
            The trained DeformationField object.
        """
        from .base import DeformationField
        return DeformationField(shape_models, appearance_models,
                                n_training_images,
                                DifferentiableThinPlateSplines,
                                self.features, self.reference_shape,
                                self.downscale, self.scaled_shape_models,
                                self.reference_frame, self._icp,
                                self.normalization_diagonal,
                                self.n_landmarks, self.group_corr)

    def _image_pre_process(self, img, scale, index):
        si = img.rescale(np.power(self.downscale, scale))
        return si

    def _compute_transforms(self, reference_frame, feature_images, group,
                            label, verbose, level_str):
        if verbose:
            print_dynamic('{}Computing transforms'.format(level_str))

        transforms = []
        for t, rt in zip(self.transforms, self._removed_transform):
            ct = t.compose_before(self._rf_align).compose_before(rt)
            transforms.append(ct)

        return transforms

    def _build_reference_frame(self, mean_shape, sparsed=True):
        r"""
        Generates the reference frame given a mean shape.

        Parameters
        ----------
        mean_shape : :map:`PointCloud`
            The mean shape to use.

        Returns
        -------
        reference_frame : :map:`MaskedImage`
            The reference frame.
        """

        return self.reference_frame


# Deformation Field using SVS, Optical Flow
class OpticalFieldBuilder(DeformationFieldBuilder):

    def __init__(self, features=igo, transform=DifferentiableThinPlateSplines,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=False,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=10, template=0):
        self._is_mc = True
        self._alpha = 15
        self._shape_desc = 'SVS'
        super(OpticalFieldBuilder, self).__init__(
            features, transform,
            trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models,
            max_shape_components, max_appearance_components,
            boundary, template
        )

    def build(self, images, group=None, label=None, verbose=False,
              target_shape=None, multi_channel=True, db_path=None,
              alpha=30, shape_desc='SVS'):

        self._is_mc = multi_channel
        self._alpha = alpha
        self._shape_desc = shape_desc
        if target_shape is None:
            target_shape = images[self.template].landmarks['PTS'].lms
        else:
            self.template = -1

        return super(OpticalFieldBuilder, self).build(
            images, group, label, verbose, target_shape, db_path
        )

    def _build_shape_model(self, shapes, max_components, target_shape):

        # Global Parameters
        alpha = self._alpha
        pdm = 0
        lms_shapes = [i.landmarks['align'].lms for i in self._norm_imgs]
        self.n_shapes = n_shapes = len(shapes)
        self._shapes = shapes
        n_landmarks = target_shape.points.shape[0] + 3

        # Simulate inconsist annotation
        sample_groups = group_from_labels(
            self._feature_images[0][self.template].landmarks['groups']
        )

        # Align Shapes Using ICP
        aligned_shapes, target_shape, self._removed_transform, self._icp_transform, self._icp\
            = self.align_shapes(shapes, target_shape)
        self._as = aligned_shapes

        # Build Reference Frame from Aligned Shapes
        bound_list = []
        for s in aligned_shapes:
            bmin, bmax = s.bounds()
            bound_list.append(bmin)
            bound_list.append(bmax)
            bound_list.append(np.array([bmin[0], bmax[1]]))
            bound_list.append(np.array([bmax[0], bmin[1]]))
        bound_list = PointCloud(np.array(bound_list))

        self.reference_frame = super(
            DeformationFieldBuilder, self
        )._build_reference_frame(bound_list)

        # Translation between reference shape and aliened shapes
        align_centre = target_shape.centre_of_bounds()
        align_t = Translation(
            self.reference_frame.centre() - align_centre
        )
        self._align_t = align_t

        self._rf_align = Translation(
            align_centre - self.reference_frame.centre()
        )

        # Set All True Pixels for Mask
        self.reference_frame.mask.pixels = np.ones(
            self.reference_frame.mask.pixels.shape, dtype=np.bool)

        # Mask Reference Frame
        self.reference_frame.landmarks['sparse'] = PointCloud(np.vstack((
            align_t.apply(lms_shapes[self.template]).points,
            align_t.apply(target_shape).points
        )))
        # self.reference_frame.constrain_mask_to_landmarks(group='sparse')

        self.reference_shape = self.reference_frame.landmarks['sparse'].lms

        # Get Dense Shape from Masked Image
        # dense_reference_shape = PointCloud(
        #     self.reference_frame.mask.true_indices()
        # )

        # Set Dense Shape as Reference Landmarks
        # self.reference_frame.landmarks['source'] = dense_reference_shape
        self._shapes = shapes
        self._aligned_shapes = []

        # Create Cache Directory
        home_dir = os.getcwd()
        dir_hex = uuid.uuid1()

        svs_path_in = '{}/.cache/{}/svs_training'.format(home_dir, dir_hex) if \
            self._db_path is None else self._db_path
        svs_path_out = svs_path_in
        self._db_path = svs_path_in

        matE = MatlabExecuter()
        mat_code_path = '/vol/atlas/homes/yz4009/gitdev/mfsfdev'
        
        # Skip building svs is path specified
        self._build_svs(svs_path_in, target_shape, aligned_shapes, align_t)

        print_dynamic('  - Building Trajectory Basis')

        # self._build_trajectory_basis(sample_groups, target_shape,
        #     aligned_shapes, dense_reference_shape, align_t)

        # Call Matlab to Build Flows
        if not isfile(self._db_path + '/result.mat'):
            print_dynamic('  - Building Shape Flow')
            matE.cd(mat_code_path)
            ext = 'gif' if self._is_mc else 'png'
            isLms = 1
            isBasis = 0
            fstr = 'addpath(\'{0}/{1}\');' \
                   'addpath(\'{0}/{2}\');' \
                   'build_flow(\'{3}\', \'{4}\', \'{5}\', {6}, {7}, ' \
                   '{8}, \'{3}/{9}\', {10}, {11}, {14}, {15}, {12}, \'{13}\')'.format(
                        mat_code_path, 'cudafiles', 'tools',
                        svs_path_in, svs_path_out, 'svs_%04d.{}'.format(ext),
                        self.template+1,
                        1, n_shapes, 'bas.mat',
                        alpha, pdm, 30, 'svs_%04d_lms.pts', isBasis, isLms
                   )
            sys.stderr.write(fstr)
            p = matE.run_function(fstr)
            p.wait()
        else:
            svs_path_out = self._db_path

        # Retrieve Results
        mat = sio.loadmat(
            '{}/result.mat'.format(svs_path_out)
        )

        _u, _v = mat['u'], mat['v']

        # Build Transforms
        print_dynamic("  - Build Transform")
        transforms = []
        for i in range(n_shapes):
            transforms.append(
                OpticalFlowTransform(_u[:, :, i], _v[:, :, i])
            )
        self.transforms = transforms

        # build dense shapes
        print_dynamic("  - Build Dense Shapes")

        testing_points = self.reference_frame.mask.true_indices()
        close_mask = BooleanImage(matpath(
            self.reference_frame.landmarks['sparse'].lms.points[3:]
        ).contains_points(testing_points).reshape(
            self.reference_frame.mask.mask.shape
        ))
        self.reference_frame.mask = close_mask
        # self.reference_frame.constrain_mask_to_landmarks(group='sparse')

        # Get Dense Shape from Masked Image
        dense_reference_shape = PointCloud(
            np.vstack((
                align_t.apply(lms_shapes[self.template]).points,
                align_t.apply(target_shape).points,
                self.reference_frame.mask.true_indices()
            ))
        )

        # Set Dense Shape as Reference Landmarks
        self.reference_frame.landmarks['source'] = dense_reference_shape
        dense_shapes = []
        for i, t in enumerate(transforms):
            warped_points = t.apply(dense_reference_shape)
            dense_shape = warped_points
            dense_shapes.append(dense_shape)

        self._dense_shapes = dense_shapes

        # build dense shape model
        dense_shape_model = super(DeformationFieldBuilder, self). \
            _build_shape_model(dense_shapes, max_components)

        self.n_landmarks = n_landmarks

        # group correlation
        if self._is_mc:
            self.group_corr = sample_groups
        else:
            self.group_corr = [range(self.n_landmarks)]

        return dense_shape_model

    # Helper Functions -------------------------------
    def align_shapes(self, shapes, target_shape):
        if self._has_lms_align:
            lms_target = self._norm_imgs[self.template].landmarks['align'].lms
            lms_shapes = [i.landmarks['align'].lms for i in self._norm_imgs]

            forward_transform = [
                AlignmentSimilarity(ls, lms_target) for ls in lms_shapes
            ]
            aligned_shapes = np.array([
                t.apply(s) for t, s in zip(forward_transform, shapes)
            ])
            removed_transform = [t.pseudoinverse() for t in forward_transform]

            target_shape = aligned_shapes[self.template]
            icp = None

        else:
            # Align Shapes Using ICP
            icp = SICP(shapes, target_shape)
            aligned_shapes = icp.aligned_shapes
            # Store Removed Transform
            removed_transform = []
            forward_transform = []
            for a_s, s in zip(aligned_shapes, shapes):
                ast = AlignmentSimilarity(a_s, s)
                removed_transform.append(ast)
                icpt = AlignmentSimilarity(s, a_s)
                forward_transform.append(icpt)

        return aligned_shapes, target_shape, removed_transform, forward_transform, icp

    def _build_svs(self, svs_path_in, target_shape, aligned_shapes, align_t):
        if not os.path.exists(svs_path_in):
            os.makedirs(svs_path_in)
        # Build Transform Using SVS
        xr, yr = self.reference_frame.shape

        if ((
                not glob.glob(svs_path_in + '/*.gif')
                and self._is_mc)
                or (not glob.glob(svs_path_in + '/svs_*.png')
                    and not self._is_mc)):
            for j, (a_s, tr) in enumerate(
                    zip(
                        [target_shape] + aligned_shapes.tolist(),
                        [AlignmentSimilarity(target_shape, target_shape)] + self._icp_transform
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
                svsLms = self._norm_imgs[self.template if j == 0 else j-1].landmarks['align'].lms
                svsLms = align_t.apply(tr.apply(svsLms))
                if not os.path.exists(
                    svsLmsPath
                ):
                    tempRef = self.reference_frame.copy()
                    tempRef.landmarks['temp'] = svsLms
                    mio.export_landmark_file(tempRef.landmarks['temp'], svsLmsPath)

                # Construct tplt_edge
                tplt_edge = None
                lindex = 0
                # Get Grouped Landmark Indexes
                if j > 0:
                    g_i = self._feature_images[0][j-1].landmarks['groups'].items()
                else:
                    g_i = self._feature_images[0][j].landmarks['groups'].items()
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
                if self._shape_desc == 'SVS':
                    svs = SVS(
                        points, tplt_edge=tplt_edge, tolerance=3, nu=0.8,
                        gamma=0.8, max_f=20
                    )
                    store_image = svs.svs_image(range(xr), range(yr))
                elif self._shape_desc == 'draw':
                    store_image = sample_points(points, xr, yr, edge_ig)
                elif self._shape_desc == 'draw_gaussian':
                    ni = sample_points(points, xr, yr, edge_ig)
                    store_image = Image.init_blank(ni.shape)
                    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                elif self._shape_desc == 'sample_gaussian':
                    ni = Image.init_blank((xr, yr))
                    for pts in points:
                        ni.pixels[0, pts[0], pts[1]] = 1
                    store_image = Image.init_blank(ni.shape)
                    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                elif self._shape_desc == 'sample':
                    store_image = Image.init_blank((xr, yr))
                    for pts in points:
                        store_image.pixels[0, pts[0], pts[1]] = 1
                else:
                    raise Exception('Undefined Shape Descriptor: {}'.format(self._shape_desc))

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
                    if self._shape_desc == 'SVS':
                        svs = SVS(
                            points[g], tplt_edge=edges, tolerance=3, nu=0.8,
                            gamma=0.8, max_f=20
                        )
                        store_image = svs.svs_image(range(xr), range(yr))
                    elif self._shape_desc == 'draw':
                        store_image = sample_points(points[g], xr, yr)
                    elif self._shape_desc == 'draw_gaussian':
                        ni = sample_points(points[g], xr, yr, edge_ig)
                        store_image = Image.init_blank(ni.shape)
                        store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                    elif self._shape_desc == 'sample_gaussian':
                        ni = Image.init_blank((xr, yr))
                        for pts in points[g]:
                            ni.pixels[0, pts[0], pts[1]] = 1
                        store_image = Image.init_blank(ni.shape)
                        store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                    elif self._shape_desc == 'sample':
                        store_image = Image.init_blank((xr, yr))
                        for pts in points[g]:
                            store_image.pixels[0, pts[0], pts[1]] = 1
                    else:
                        raise Exception('Undefined Shape Descriptor: {}'.format(self._shape_desc))

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

    def _build_trajectory_basis(self, sample_groups, target_shape,
                                aligned_shapes, dense_reference_shape, align_t):
        if not isfile(self._db_path + '/bas.mat'):
            # Build basis
            # group correspondence
            align_gcorr = None
            groups = np.array(sample_groups)
            tps_t = []

            if self._is_mc:
                for g in groups:
                    g_align_s = []
                    for aligned_s in aligned_shapes:
                        g_align_s.append(PointCloud(aligned_s.points[g]))
                    # _, point_correspondence = FastNICP(
                    #   g_align_s, PointCloud(icp.target.points[g])
                    # )
                    gnicp = SNICP(g_align_s, PointCloud(target_shape.points[g]))
                    g_align = np.array(gnicp.point_correspondence) + g[0]
                    if align_gcorr is None:
                        align_gcorr = g_align
                    else:
                        align_gcorr = np.hstack((align_gcorr, g_align))
            else:
                print ' single channel basis'
                _, point_correspondence = FastNICP(aligned_shapes, target_shape)
                # gnicp = NICP(icp.aligned_shapes, icp.target)
                align_gcorr = point_correspondence

            # compute non-linear transforms (tps)
            for a_s, a_corr in zip(aligned_shapes, align_gcorr):
                # Align shapes with reference frame
                temp_as = align_t.apply(a_s)
                temp_s = align_t.apply(PointCloud(target_shape.points[a_corr]))

                self._aligned_shapes.append(temp_as)
                tps_t.append(self.transform(temp_s, temp_as))
                # transforms.append(pwa(temp_s, temp_as))

            # build dense shapes
            dense_shapes = []
            for i, t in enumerate(tps_t):
                warped_points = t.apply(dense_reference_shape)
                dense_shape = warped_points
                dense_shapes.append(dense_shape)

            # build dense shape model
            uvs = np.array([ds.points.flatten() - dense_reference_shape.points.flatten()
                            for ds in dense_shapes])
            nPoints = dense_shapes[0].n_points
            h, w = self.reference_frame.shape
            W = np.zeros((2 * self.n_shapes, nPoints))
            v = uvs[:, 0:2*nPoints:2]
            u = uvs[:, 1:2*nPoints:2]

            u = np.transpose(np.reshape(u.T, (w, h, self.n_shapes)), [1, 0, 2])
            v = np.transpose(np.reshape(v.T, (w, h, self.n_shapes)), [1, 0, 2])

            W[0:2*self.n_shapes:2, :] = np.reshape(u, (w*h, self.n_shapes)).T
            W[1:2*self.n_shapes:2, :] = np.reshape(v, (w*h, self.n_shapes)).T

            S = W.dot(W.T)
            U, var, _ = np.linalg.svd(S)
            # csum = np.cumsum(var)
            # csum = 100 * csum / csum[-1]
            # accept_rate = 99.9
            # # rank = np.argmin(np.abs(csum - accept_rate))
            Q = U[:, :]
            basis = np.vstack((Q[1::2, :], Q[0::2, :]))

            # construct basis
            sio.savemat('{}/{}'.format(self._db_path, 'bas.mat'), {
                'bas': basis,
                'tps_u': W[0:2*self.n_shapes:2, :],
                'tps_v': W[1:2*self.n_shapes:2, :],
                'ou': uvs[:, 1:2*nPoints:2],
                'ov': uvs[:, 0:2*nPoints:2],
                'tu': u,
                'tv': v
            })


class OFBodyBuilder(OpticalFieldBuilder):

    def __init__(self, features=igo, transform=DifferentiableThinPlateSplines,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=False,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=10, template=0):
        super(OFBodyBuilder, self).__init__(
            features, transform,
            trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models,
            max_shape_components, max_appearance_components,
            boundary, template
        )

    def _build_svs(self, svs_path_in, target_shape, aligned_shapes, align_t):
        svs_path_in = self._db_path
        svs_path_out = '{}'.format(svs_path_in)
        seg_mask_points = mio.import_pickle(
            svs_path_in+'/seg_mask_points_group.pkl'
        )
        # compute and normalize mask images size
        # single channel ---------------------------------------------
        if not glob.glob(svs_path_in + '/svs_*.png'):
            mask_images = []
            for mii in range(self.n_shapes):
                mi = mio.import_image(
                    '{}/{:04d}_mask.png'.format(svs_path_in, mii + 1)
                )
                mi.landmarks['PTS'] = mio.import_landmark_file(
                    svs_path_in + '/{:04d}.pts'.format(mii+1)
                )
                if mi.n_channels == 3:
                        mi = mi.as_greyscale()
                mask_images.append(mi)

            nshape = mask_images[0].shape
            nlms = mask_images[0].landmarks['PTS'].lms.n_points

            _, normalized_mask_images = \
                normalization_wrt_reference_shape(
                    mask_images, 'PTS', None,
                    self.normalization_diagonal, target_shape, False
                )

            for index, mi in enumerate(
                    normalized_mask_images[:self.n_shapes]
            ):

                nmi = mi.warp_to_shape(
                    mi.shape, self._removed_transform[index],
                    warp_landmarks=True
                ).warp_to_shape(
                    self.reference_frame.shape, self._rf_align,
                    warp_landmarks=True
                )
                if nmi.n_channels == 3:
                    nmi = nmi.as_greyscale()

                mio.export_image(
                    nmi, '{}/svs_{:04d}.png'.format(svs_path_in, index+1),
                    overwrite=True
                )
                mio.export_landmark_file(
                    nmi.landmarks['PTS'],
                    svs_path_in + '/svs_{:04d}.pts'.format(index+1),
                    overwrite=True
                )
        # end single channel ---------------------------------------------

        # normalise multichannel images
        # multichannel -----------------------------------------------------
        if not glob.glob(svs_path_in + '/svs_*.gif'):
            for iindex in range(self.n_shapes): #{[mask, Z]}:
                msi = seg_mask_points[iindex]
                # for every image
                mask_seg_images = []
                for iseg in range(np.max(msi[1])+1):
                    seg_img = Image.init_blank(nshape)
                    seg_img.landmarks['PTS'] = mio.import_landmark_file(
                        svs_path_in + '/{:04d}.pts'.format(iindex+1)
                    )
                    for pt in msi[0][np.where(msi[1] == iseg)]:
                        try:
                            seg_img.pixels[0, pt[0], pt[1]] = 1
                        except IndexError:
                            print 'Index Error'
                    mask_seg_images.append(seg_img)

                _, normalized_seg_images = \
                    normalization_wrt_reference_shape(
                        mask_seg_images, 'PTS', None,
                        self.normalization_diagonal, target_shape, False
                    )

                for index, mi in enumerate(normalized_seg_images[:self.n_shapes]):

                    nmi = mi.warp_to_shape(
                        mi.shape, self._removed_transform[iindex],
                        warp_landmarks=True
                    ).warp_to_shape(
                        self.reference_frame.shape, self._rf_align,
                        warp_landmarks=True
                    )

                    if nmi.n_channels == 3:
                        nmi = nmi.as_greyscale()

                    # print(iindex,index, nmi.shape, self.reference_frame.shape)

                    mio.export_image(
                        nmi, svs_path_in + '/svs_{:04d}_g{:02d}.png'.format(
                            iindex+1, index
                        ), overwrite=True
                    )

                subprocess.Popen([
                    'convert',
                    '-delay', '10', '-loop', '0',
                    '{}/svs_{:04d}_g*.png'.format(svs_path_in, iindex+1),
                    '{}/svs_{:04d}.gif'.format(svs_path_in, iindex+1)])

                print_dynamic(
                    ' - Generating Multi-Channel Images: {}/{}'.format(
                        iindex+1, self.n_shapes
                    )
                )
        # end multi channel ------------------------------------------------

    def _build_trajectory_basis(self, sample_groups, target_shape,
                                aligned_shapes, dense_reference_shape, align_t):
        pass


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


def build_reference_frame(mean_shape):
    reference_shape = mean_shape

    from menpofit.aam.base import build_reference_frame as brf

    return brf(reference_shape)


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