from menpo.transform.base import Transform, VInvertible, VComposable
from menpo.transform import Translation, AlignmentSimilarity
from menpo.transform.icp import SICP as ICP, SNICP as NICP
from menpo.shape import PointCloud
import numpy as np
from scipy.spatial import KDTree

import numpy as np
from copy import deepcopy
from menpo.transform import AlignmentUniformScale
from menpo.image import BooleanImage
from menpofit.fitter import ModelFitter, noisy_shape_from_bounding_box
from menpofit.modelinstance import OrthoPDM
from menpofit.sdm import SupervisedDescentFitter
from menpofit.transform import OrthoMDTransform, LinearOrthoMDTransform
import menpofit.checks as checks

from .transforms import LinearWarp
from .results import DAAMFitterResult

from menpofit.aam import LucasKanadeAAMFitter
from menpofit.aam.algorithm.lk import (
    LucasKanadeStandardInterface, LucasKanadeLinearInterface,
    LucasKanadePatchInterface, WibergInverseCompositional)

#
# class DFFittingResult(ParametricFittingResult):
#
#     @property
#     def n_landmarks(self):
#         return self.fitter.transform.n_landmarks
#
#     # @property
#     # def final_shape(self):
#     #     return PointCloud(self.final_transform.target.points[
#     #                       :self.n_landmarks])
#     #
#     # @property
#     # def initial_shape(self):
#     #     return PointCloud(self.initial_transform.target.points[
#     #                       :self.n_landmarks])
#
#
# class DeformationFieldAICompositional(SIC):
#
#     def _create_fitting_result(self, image, parameters, gt_shape=None):
#         return DFFittingResult(image, self, parameters=[parameters],
#                                        gt_shape=gt_shape)
#
#
# class DFMultilevelFittingResult(AAMMultilevelFittingResult):
#
#     def __init__(self, image, multiple_fitter, fitting_results,
#                  affine_correction, gt_shape=None):
#         super(DFMultilevelFittingResult, self).__init__(
#             image, multiple_fitter, fitting_results, affine_correction,
#             gt_shape
#         )
#         # self._prepare_gt_rf()
#
#     @property
#     def aam(self):
#         return self.fitter.aam
#
#     def appearance_errors(self):
#         t = self.fitter.fitters[-1].transform
#         return [compute_appearance_error(t, s, self.aam.reference_frame,
#                 self.image, ar) for (s, ar) in
#                 zip(self.shapes, self.appearance_reconstructions)]
#
#     def true_errors(self):
#         # weights = self.aam.appearance_models[-1].project(self._warpped_images[0][0])
#         # return img_error(
#         #     self._warpped_images[0][0],
#         #     self.aam.appearance_models[-1].instance(weights)
#         # )
#         t = self.fitter.fitters[-1].transform
#         t.set_target(self._gt_shape)
#         ds = t.target.points
#         return compute_error(
#             PointCloud(ds[self._lms_corr[0]]),
#             self._gt_shape
#         )
#
#     @property
#     def sparse_shapes(self):
#         if hasattr(self, 'sparseshapes'):
#             pass
#         else:
#             self.sparseshapes = [
#                 PointCloud(
#                     s.points[:self.aam.n_landmarks]
#                 ) for s in self.shapes
#             ]
#         return self.sparseshapes
#
#     def _prepare_gt_rf(self):
#         gt_image = self.image
#         # compute reference_shape and normalize images size
#         self.reference_shape, normalized_images = \
#             normalization_wrt_reference_shape(
#                 [gt_image], 'PTS', None, self.aam.normalization_diagonal, True
#             )
#
#         # create pyramid
#         generators = create_pyramid(normalized_images, self.n_levels,
#                                     self.downscale, self.aam.features,
#                                     verbose=True)
#         self._feature_images = []
#         self._warpped_images = []
#         for j in range(self.n_levels):
#
#             # get feature images of current level
#             feature_images = []
#             for c, g in enumerate(generators):
#                 feature_images.append(next(g))
#
#             self._feature_images.append(feature_images)
#
#             # extract potentially rescaled shapes
#             shapes = [i.landmarks['PTS'][None] for i in feature_images]
#
#             # define shapes that will be used for training
#             if j == 0:
#                 original_shapes = shapes
#                 train_shapes = shapes
#             else:
#                 train_shapes = original_shapes
#
#             # Align Shapes Using ICP
#             icp = ICP(train_shapes, self.aam.icp.target)
#             aligned_shapes = icp.aligned_shapes
#
#             # Store Removed Transform
#             self._removed_transform = []
#             for a_s, s in zip(aligned_shapes, train_shapes):
#                 ast = AlignmentSimilarity(a_s, s)
#                 self._removed_transform.append(ast)
#
#             # Get Dense Shape from Masked Image
#             dense_reference_shape = self.aam.reference_frame.landmarks[
#                 'source'].lms
#             self._transforms = transforms = []
#
#             align_centre = icp.target.centre_of_bounds()
#             align_t = Translation(
#                 dense_reference_shape.centre_of_bounds()-align_centre
#             )
#
#             self._rf_align = Translation(
#                 align_centre - dense_reference_shape.centre_of_bounds()
#             )
#
#             # Ground Truth Correspondence
#             # align_gcorr = [range(55)]*len(shapes)
#
#             # Finding Correspondance
#             # self._nicp = nicp = NICP(icp.aligned_shapes, icp.target)
#             # align_gcorr = nicp.point_correspondence
#
#             # Finding Correspondence by Group
#             align_gcorr = None
#             groups = self.aam.group_corr
#
#             for g in groups:
#                 g_align_s = []
#                 for aligned_s in icp.aligned_shapes:
#                     g_align_s.append(PointCloud(aligned_s.points[g]))
#                 gnicp = NICP(g_align_s, PointCloud(icp.target.points[g]))
#                 g_align = np.array(gnicp.point_correspondence) + g[0]
#                 if align_gcorr is None:
#                     align_gcorr = g_align
#                 else:
#                     align_gcorr = np.hstack((align_gcorr, g_align))
#
#             # compute non-linear transforms (tps)
#             for a_s, a_corr in zip(aligned_shapes, align_gcorr):
#                 # Align shapes with reference frame
#                 temp_as = align_t.apply(a_s)
#                 temp_s = align_t.apply(PointCloud(icp.target.points[a_corr]))
#
#                 transforms.append(tps(temp_s, temp_as))
#                 # transforms.append(pwa(temp_s, temp_as))pes
#
#             # build dense shapes
#             lms_corr = []
#             for i, (t, a_s) in enumerate(zip(transforms, aligned_shapes)):
#                 dense_shape = t.apply(dense_reference_shape)
#                 kdOBJ = KDTree(dense_shape.points)
#                 _, match = kdOBJ.query(a_s.points)
#                 lms_corr.append(
#                     match
#                 )
#             self._lms_corr = lms_corr
#
#
# def img_error(img_1, img_2):
#     t_pixels = img_1.pixels
#     w_pixels = img_2.pixels
#     diff = t_pixels - w_pixels
#     error = np.sum(diff[:, :, :]**2)
#
#     return error
#
#
# def compute_appearance_error(t, shape, ref, gt_img, rec_img):
#     t.set_target(shape)
#     test_img = gt_img.warp_to_mask(ref.mask, t)
#     return img_error(test_img, rec_img)
#
#
# class LucasKanadeDeformationFieldAAMFitter(LucasKanadeAAMFitter):
#
#     def __init__(self, aam, algorithm=DeformationFieldAICompositional,
#                  md_transform=LinearWarp, n_shape=None,
#                  n_appearance=None, **kwargs):
#         super(LucasKanadeDeformationFieldAAMFitter, self).__init__(
#             aam, algorithm, md_transform, n_shape, n_appearance, **kwargs)
#
#     @property
#     def algorithm(self):
#         r"""
#         Returns a string containing the name of fitting algorithm.
#
#         :type: `str`
#         """
#         return 'DF-AAM-' + self._fitters[0].algorithm
#
#     def _set_up(self, algorithm=DeformationFieldAICompositional,
#                 md_transform=LinearWarp,
#                 global_transform=DifferentiableAlignmentSimilarity,
#                 n_shape=None, n_appearance=None, **kwargs):
#
#         # check n_shape parameter
#         if n_shape is not None:
#             if type(n_shape) is int or type(n_shape) is float:
#                 for sm in self.aam.shape_models:
#                     sm.n_active_components = n_shape
#             elif len(n_shape) == 1 and self.aam.n_levels > 1:
#                 for sm in self.aam.shape_models:
#                     sm.n_active_components = n_shape[0]
#             elif len(n_shape) == self.aam.n_levels:
#                 for sm, n in zip(self.aam.shape_models, n_shape):
#                     sm.n_active_components = n
#             else:
#                 raise ValueError('n_shape can be an int or a float or None '
#                                  'or a list containing 1 or {} of '
#                                  'those'.format(self.aam.n_levels))
#
#         # check n_appearance parameter
#         if n_appearance is not None:
#             if type(n_appearance) is int or type(n_appearance) is float:
#                 for am in self.aam.appearance_models:
#                     am.n_active_components = n_appearance
#             elif len(n_appearance) == 1 and self.aam.n_levels > 1:
#                 for am in self.aam.appearance_models:
#                     am.n_active_components = n_appearance[0]
#             elif len(n_appearance) == self.aam.n_levels:
#                 for am, n in zip(self.aam.appearance_models, n_appearance):
#                     am.n_active_components = n
#             else:
#                 raise ValueError('n_appearance can be an integer or a float '
#                                  'or None or a list containing 1 or {} of '
#                                  'those'.format(self.aam.n_levels))
#
#         self._fitters = []
#         for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
#                                          self.aam.shape_models)):
#             transform = md_transform(
#                 sm, self.aam.group_corr, self.aam.n_landmarks
#             )
#             self._fitters.append(algorithm(am, transform, **kwargs))
#
#     def _create_fitting_result(self, image, fitting_results, affine_correction,
#                                gt_shape=None):
#         r"""
#         Creates the :class: `menpo.aam.fitting.MultipleFitting` object
#         associated with a particular Fitter object.
#
#         Parameters
#         -----------
#         image: :class:`menpo.image.masked.MaskedImage`
#             The original image to be fitted.
#         fitting_results: :class:`menpo.fit.fittingresult.FittingResultList`
#             A list of basic fitting objects containing the state of the
#             different fitting levels.
#         affine_correction: :class: `menpo.transforms.affine.Affine`
#             An affine transform that maps the result of the top resolution
#             fitting level to the space scale of the original image.
#         gt_shape: class:`menpo.shape.PointCloud`, optional
#             The ground truth shape associated to the image.
#
#             Default: None
#         error_type: 'me_norm', 'me' or 'rmse', optional
#             Specifies the way in which the error between the fitted and
#             ground truth shapes is to be computed.
#
#             Default: 'me_norm'
#
#         Returns
#         -------
#         fitting: :class:`menpo.fitmultilevel.fittingresult.MultilevelFittingResult`
#             The fitting object that will hold the state of the fitter.
#         """
#         return DFMultilevelFittingResult(image, self, fitting_results,
#                             affine_correction, gt_shape=gt_shape)
#
#     @property
#     def fitters(self):
#         return self._fitters
#
#     @property
#     def _str_title(self):
#         r"""
#         Returns a string containing name of the model.
#
#         :type: `string`
#         """
#         return 'AAM Based Deformation Field Fitter'
#
#     def __str__(self):
#         out = super(LucasKanadeDeformationFieldAAMFitter, self).__str__()
#         out_splitted = out.splitlines()
#         out_splitted[0] = self._str_title
#         return '\n'.join(out_splitted)


class LucasKanadeDAAMFitter(LucasKanadeAAMFitter):
    r"""
    """
    def __init__(self, aam, lk_algorithm_cls=WibergInverseCompositional,
                 n_shape=None, n_appearance=None, sampling=None):
        self._model = aam
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)
        self._sampling = checks.check_sampling(sampling, aam.n_scales)
        self._set_up(lk_algorithm_cls)

    def _set_up(self, lk_algorithm_cls):
        self.algorithms = []
        for j, (am, sm, s) in enumerate(zip(self.aam.appearance_models,
                                            self.aam.shape_models,
                                            self._sampling)):

            template = am.mean()
            # build orthonormal model driven transform
            md_transform = LinearWarp(
                sm, self.aam.n_landmarks, self.aam.n_align_lms)
            interface = LucasKanadeStandardInterface(am, md_transform,
                                                     template, sampling=s)
            algorithm = lk_algorithm_cls(interface)

            self.algorithms.append(algorithm)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return DAAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)