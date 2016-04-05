import numpy as np

from menpo.base import Targetable
from menpo.transform.base import Transform, VInvertible, VComposable
from menpo.shape import PointCloud
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import DifferentiableAlignmentSimilarity


# Optical Flow Transform
class OpticalFlowTransform(Transform):
    def __init__(self, u, v):
        super(OpticalFlowTransform, self).__init__()
        self._u = v
        self._v = u

    def _apply(self, x, **kwargs):
        ret = x.copy().astype(float)
        for p in ret:
            i, j = p[0].astype(int), p[1].astype(int)
            p += np.array([self._u[i, j], self._v[i, j]])
        return ret

    @property
    def n_dims(self):
        r"""
        The dimensionality of the data the transform operates on.

        None if the transform is not dimension specific.

        :type: int or None
        """
        return 2


class LinearWarp(OrthoPDM, Transform, VInvertible, VComposable):

    def __init__(self, model, n_landmarks=0, n_align_lms=0):
        super(LinearWarp, self).__init__(model)
        self.pdm = OrthoPDM(model)
        self.n_landmarks = n_landmarks
        self.n_align_lms = n_align_lms
        self.W = np.vstack((self.similarity_model.components,
                            self.model.components))
        v = self.W[:, :self.n_dims*self.n_landmarks]
        self.pinv_v = np.linalg.pinv(v)

        va = self.W[:, :self.n_dims*self.n_align_lms]
        self.pinv_va = np.linalg.pinv(va)
        # sm_mean_l = self.models[self.model_index-1].mean()
        # sm_mean_h = self.model.mean()
        # icp = ICP([sm_mean_l], sm_mean_h)
        # spare_index = spare_index_base = icp.point_correspondence[0]*2
        #
        # for i in range(self.n_dims-1):
        #     spare_index = np.vstack((spare_index, spare_index_base+i+1))
        #
        # spare_index = spare_index.T.reshape(
        #     spare_index_base.shape[0]*self.n_dims
        # )
        #
        # v = self.W[:, spare_index]
        # self.pinv_v = scipy.linalg.pinv(v)

    @property
    def n_dims(self):
        r"""
        The dimensionality of the data the transform operates on.

        None if the transform is not dimension specific.

        :type: int or None
        """
        return 2

    @property
    def dense_target(self):
        return PointCloud(self.target.points[self.n_landmarks:])

    @property
    def sparse_target(self):
        return PointCloud(self.target.points[:self.n_landmarks])

    def set_target(self, target):
        if target.n_points < self.target.n_points:

            if target.n_points < self.n_landmarks:
                target = PointCloud(target.points[:self.n_align_lms])
                target = np.dot(np.dot(target.as_vector(), self.pinv_va), self.W)
            else:
                target = PointCloud(target.points[:self.n_landmarks])
                target = np.dot(np.dot(target.as_vector(), self.pinv_v), self.W)

            target = PointCloud(np.reshape(target, (-1, self.n_dims)))

        OrthoPDM.set_target(self, target)

    def _apply(self, _, **kwargs):
        return self.target.points[self.n_landmarks:]

    def d_dp(self, _):
        return OrthoPDM.d_dp(self, _)[self.n_landmarks:, ...]

    def has_true_inverse(self):
        return False

    def pseudoinverse_vector(self, vector):
        return -vector

    def compose_after_from_vector_inplace(self, delta):
        self.from_vector_inplace(self.as_vector() + delta)
