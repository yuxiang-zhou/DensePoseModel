from .transforms import LinearWarp
from .results import DAAMFitterResult

from menpofit import checks
from menpofit.aam import LucasKanadeAAMFitter
from menpofit.aam.algorithm.lk import (
    LucasKanadeStandardInterface, WibergInverseCompositional)


class LucasKanadeDAAMFitter(LucasKanadeAAMFitter):
    r"""
    """
    def _set_up(self, lk_algorithm_cls):
        self.algorithms = []
        for j, (am, sm, s) in enumerate(zip(self.aam.appearance_models,
                                            self.aam.shape_models,
                                            self._sampling)):

            template = am.mean()
            # build orthonormal model driven transform
            md_transform = LinearWarp(
                sm.model, self.aam.n_landmarks, self.aam.n_align_lms)
            interface = LucasKanadeStandardInterface(am, md_transform,
                                                     template, sampling=s)
            algorithm = lk_algorithm_cls(interface)

            self.algorithms.append(algorithm)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return DAAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)