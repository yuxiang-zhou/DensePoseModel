from copy import deepcopy
from .tools import rescale_images_to_reference_shape
from .tools import svs_shape
from .transforms import LinearWarp

from menpofit.aam import HolisticAAM

from menpo.transform.piecewiseaffine.base import CythonPWA as pwa
from menpo.feature import no_op
from menpo.model import PCAModel
from menpofit.modelinstance import OrthoPDM
from menpo.visualize import print_dynamic

from menpofit.builder import (
    build_reference_frame,
    compute_features, scale_images,)


class dAAMs(HolisticAAM):
    r"""
    Active Appearance Model class.
    """
    def __init__(self, images, group=None, verbose=False, reference_shape=None,
                 holistic_features=no_op, diagonal=None, target_group=None,
                 scales=(0.5, 1.0), max_shape_components=None,
                 max_appearance_components=None, batch_size=None, tight_mask=True,
                 shape_desc=svs_shape):

        self.tight_mask = tight_mask
        self.shape_desc = shape_desc
        self.target_group = target_group
        super(dAAMs, self).__init__(images, group, verbose,
                 reference_shape,
                 holistic_features,
                 LinearWarp, diagonal,
                 scales, OrthoPDM, max_shape_components,
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
        image_batch, self.transforms, self.reference_frame, self.n_landmarks, self.n_align_lms,_,_,_,self.reference_shape,self.debug\
            = rescale_images_to_reference_shape(
                image_batch, group, self.reference_shape,
                tight_mask=self.tight_mask, sd=self.shape_desc, target_group=self.target_group,
                verbose=verbose
            )

        self.normalised_img = image_batch

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
            # handle scales
            scaled_images = feature_images

            # Extract potentially rescaled shapes
            scale_shapes = [i.landmarks[group].lms for i in scaled_images]

            # Build the shape model
            if verbose:
                print_dynamic('{}Building shape model'.format(scale_prefix))

            if j == 0:
                shape_model = self._build_shape_model(
                    scale_shapes, j)
                self.shape_models.append(shape_model)
            else:
                self.shape_models.append(deepcopy(shape_model))

            # Obtain warped images - we use a scaled version of the
            # reference shape, computed here. This is because the mean
            # moves when we are incrementing, and we need a consistent
            # reference frame.
            warped_images = self.warped_images = self._warp_images(
                scaled_images, scale_shapes, self.reference_shape,
                j, scale_prefix, verbose
            )

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

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        scaled_images = scale_images(images, self.scales[scale_index],
                                             prefix=prefix,
                                             verbose=verbose)
        warpped, landmarkmapping, timages = warp_images(scaled_images, self.reference_frame, self.transforms, self.scales[scale_index])
        if self.scales[scale_index] >= 1.0:
            self.mapping = [landmarkmapping, timages]
        return warpped

    def _instance(self, scale_index, shape_instance, appearance_instance):
        template = self.appearance_models[scale_index].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = build_reference_frame(shape_instance)

        transform = pwa(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)


def warp_images(images, reference_frame, transforms, scale):

    warped_images = []
    # Build a dummy transform, use set_target for efficiency

    for i, t in zip(images, transforms):
        # Update Transform Target
        # warp images
        rescale = 1.0 / scale
        scale_i = i.rescale(rescale) if scale != 1.0 else i
        warped_i = scale_i.warp_to_mask(reference_frame.mask, t, warp_landmarks=False)
        # attach reference frame landmarks to images
        warped_i.landmarks['source'] = reference_frame.landmarks['source']
        warped_images.append(warped_i)

    landmark_mapping = []
    for i, t in zip(warped_images, transforms):
        landmark_mapping.append(t.apply(i.landmarks['source'].lms))

    return warped_images, landmark_mapping, images
