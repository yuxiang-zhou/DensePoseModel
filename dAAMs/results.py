import numpy as np

from menpo.shape import PointCloud
from menpofit.aam.result import AAMResult as AAMFitterResult
from menpofit.error import euclidean_bb_normalised_error as compute_point_to_point_error


class DAAMFitterResult(AAMFitterResult):

    def final_error(self, compute_error=None):
        r"""
        Returns the final fitting error.

        Parameters
        -----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting procedure.
        """
        if compute_error is None:
            compute_error = compute_normalise_point_to_point_error
        if self.gt_shape is not None:
            final_shape = PointCloud(
                self.final_shape.points[:self.gt_shape.n_points]
            )
            return compute_error(final_shape, self.gt_shape)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def initial_error(self, compute_error=None):
        r"""
        Returns the initial fitting error.

        Parameters
        -----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        initial_error : `float`
            The initial error at the start of the fitting procedure.
        """
        if compute_error is None:
            compute_error = compute_normalise_point_to_point_error
        if self.gt_shape is not None:
            initial_shape = PointCloud(
                self.initial_shape.points[:self.gt_shape.n_points]
            )
            return compute_error(initial_shape, self.gt_shape)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def errors(self, compute_error=None):
        r"""
        Returns a list containing the error at each fitting iteration.

        Parameters
        -----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        errors : `list` of `float`
            The errors at each iteration of the fitting process.
        """
        if compute_error is None:
            compute_error = compute_normalise_point_to_point_error
        if self.gt_shape is not None:
            return [
                compute_error(
                    PointCloud(t.points[:self.gt_shape.n_points]),
                    self.gt_shape
                ) for t in self.shapes
            ]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')


def compute_normalise_point_to_point_error(shape, gt_shape, norm_shape=None):
    r"""
    """
    if norm_shape is None:
        norm_shape = gt_shape
    normalizer = np.mean(np.max(norm_shape, axis=0) -
                         np.min(norm_shape, axis=0))
    return compute_point_to_point_error(shape, gt_shape) / normalizer