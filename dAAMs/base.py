import os
import sys
import glob
import uuid
import subprocess
import menpo.io as mio
import scipy.io as sio
import numpy as np
import warnings

from copy import deepcopy

from menpo.shape import PointCloud
from menpo.transform.piecewiseaffine.base import CythonPWA as pwa
from menpo.transform.piecewiseaffine import PiecewiseAffine

from .MatlabExecuter import MatlabExecuter
from .lineerror import interpolate
from .svs import SVS
from .transforms import OpticalFlowTransform
from .tools import rescale_images_to_reference_shape
from .transforms import LinearWarp

from menpofit.aam import HolisticAAM
from menpofit import checks
from menpofit.transform import (DifferentiableThinPlateSplines,
                                DifferentiablePiecewiseAffine)
from menpo.transform.piecewiseaffine.base import CythonPWA as pwa
from menpo.transform import Scale
from menpofit.builder import normalization_wrt_reference_shape

from menpo.feature import no_op
from menpo.feature import igo

from menpo.image import Image, BooleanImage
from menpo.transform.icp import SICP, SNICP
from menpo.model import PCAModel
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.transform import Translation, AlignmentSimilarity


from skimage import filters
from os.path import isfile
from matplotlib.path import Path as matpath
from menpofit.base import name_of_callable, batch
from menpofit.builder import (
    build_reference_frame, build_patch_reference_frame,
    compute_features, scale_images, build_shape_model,
    align_shapes, densify_shapes,
    extract_patches, MenpoFitBuilderWarning, compute_reference_shape)


class dAAMs(HolisticAAM):
    r"""
    Active Appearance Model class.
    """
    def __init__(self, images, group=None, verbose=False, reference_shape=None,
                 holistic_features=no_op, diagonal=None,
                 scales=(1.0, 1.0), max_shape_components=None,
                 max_appearance_components=None, batch_size=None):

        super(dAAMs, self).__init__(images, group, verbose,
                 reference_shape,
                 holistic_features,
                 LinearWarp, diagonal,
                 scales, max_shape_components,
                 max_appearance_components, batch_size)

    def _train_batch(self, image_batch, increment=False, group=None,
                     verbose=False, shape_forgetting_factor=1.0,
                     appearance_forgetting_factor=1.0):
        r"""
        Builds an Active Appearance Model from a list of landmarked images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the AAM.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        aam : :map:`AAM`
            The AAM object. Shape and appearance models are stored from
            lowest to highest scale
        """
        # Rescale to existing reference shape
        image_batch, self.transforms, self.reference_frame, self.n_landmarks\
            = rescale_images_to_reference_shape(
                image_batch, group, self.reference_shape,
                verbose=verbose
            )

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')

        feature_images = []
        # for each scale (low --> high)
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Handle holistic features
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif j == 0 or self.holistic_features[j] is not self.holistic_features[j - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)

            # Obtain warped images - we use a scaled version of the
            # reference shape, computed here. This is because the mean
            # moves when we are incrementing, and we need a consistent
            # reference frame.

            warped_images = warp_images(feature_images, self.reference_frame, self.transforms, group, scale_prefix, verbose)

            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                warped_images = scale_images(warped_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)

            # Extract potentially rescaled shapes
            scale_shapes = [i.landmarks[group].lms for i in warped_images]

             # Build the shape model
            if verbose:
                print_dynamic('{}Building shape model'.format(scale_prefix))

            if j == 0:
                shape_model = self._build_shape_model(
                    scale_shapes, j)
                self.shape_models.append(shape_model)
            else:
                self.shape_models.append(deepcopy(shape_model))

            # obtain appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(
                    scale_prefix))

            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[j])
            # add appearance model to the list
            self.appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(scale_prefix))

        # Because we just copy the shape model, we need to wait to trim
        # it after building each model. This ensures we can have a different
        # number of components per level
        for j, sm in enumerate(self.shape_models):
            max_sc = self.max_shape_components[j]
            if max_sc is not None:
                sm.trim_components(max_sc)

    def _instance(self, scale_index, shape_instance, appearance_instance):
        template = self.appearance_models[scale_index].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = build_reference_frame(shape_instance)

        transform = pwa(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)


def warp_images(images, reference_frame, transforms, group, prefix='',
                verbose=None):

    warped_images = []
    # Build a dummy transform, use set_target for efficiency

    for i, t in zip(images, transforms):
        # Update Transform Target
        # warp images
        warped_i = i.warp_to_mask(reference_frame.mask, t)
        # attach reference frame landmarks to images
        warped_i.landmarks[group] = reference_frame.landmarks['source']

        warped_images.append(warped_i)

    return warped_images