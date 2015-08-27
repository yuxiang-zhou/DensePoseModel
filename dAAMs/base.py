from menpofit.aam import AAM
from menpo.shape import PointCloud
from menpofit.aam.builder import build_reference_frame
from menpo.transform.piecewiseaffine.base import CythonPWA as pwa
from menpo.transform.piecewiseaffine import PiecewiseAffine

import numpy as np


class DeformationField(AAM):

    def __init__(self, shape_models, appearance_models, n_training_images,
                 transform, features, reference_shape, downscale,
                 scaled_shape_models, reference_frame, icp,
                 normalization_diagonal, n_landmarks, group_corr):
        super(DeformationField, self).__init__(
            shape_models, appearance_models, n_training_images, transform,
            features, reference_shape, downscale, scaled_shape_models,
            n_landmarks, group_corr)
        self.reference_frame = reference_frame
        self.icp = icp
        self.normalization_diagonal = normalization_diagonal
        self.group_corr = group_corr

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'AAM Based Deformation Field'

    def __str__(self):
        out = super(DeformationField, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        return '\n'.join(out_splitted)

    # def _instance(self, level, shape_instance, appearance_instance):
    #     template = self.appearance_models[level].mean()
    #     landmarks = template.landmarks['sparse'].lms
    #
    #     # use spare shape
    #     sparse_rate = 8
    #     w, h = template.shape
    #     sparse_index = []
    #     for i in range(w / sparse_rate):
    #         for j in range(h / sparse_rate):
    #             index = i*sparse_rate*h+j*sparse_rate
    #             if index < landmarks.n_points:
    #                 sparse_index.append(index)
    #
    #     reference_frame = self._build_reference_frame(
    #         PointCloud(shape_instance.points[sparse_index])
    #     )
    #
    #     source = reference_frame.landmarks['source'].lms
    #
    #     target = PointCloud(landmarks.points[sparse_index])
    #
    #     transform = self.transform(source, target)
    #
    #     return appearance_instance

    # def _instance(self, level, shape_instance, appearance_instance):
    #     template = self.appearance_models[level].mean()
    #     landmarks = template.landmarks['source'].lms
    #
    #     reference_frame = self._build_reference_frame(
    #         shape_instance)
    #
    #     transform = pwa(
    #         reference_frame.landmarks['source'].lms, landmarks)
    #
    #     return appearance_instance.warp_to_mask(reference_frame.mask,
    #                                             transform, warp_landmarks=True)

    # def _instance(self, level, shape_instance, appearance_instance):
    #     template = self.appearance_models[level].mean()
    #     landmarks = PointCloud(
    #         template.landmarks['source'].lms.points[:self.n_landmarks]
    #     )
    #
    #     appearance_instance.landmarks['source'] = landmarks
    #
    #     reference_frame = self._build_reference_frame(
    #         PointCloud(shape_instance.points[:self.n_landmarks])
    #     )
    #
    #     transform = PiecewiseAffine(
    #         reference_frame.landmarks['source'].lms,
    #         landmarks
    #     )
    #
    #     return appearance_instance.warp_to_mask(reference_frame.mask,
    #                                             transform, warp_landmarks=True)

    def _build_reference_frame(self, reference_shape, landmarks=None):

        return build_reference_frame(
            reference_shape, trilist=None, boundary=10)
