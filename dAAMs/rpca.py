from __future__ import division
import numpy as np
from menpo.math import pca, ipca, as_matrix, pcacov
from menpo.model import PCAVectorModel
from menpo.model.vectorizable import VectorizableBackedModel
from menpo.visualize import print_dynamic
from menpo.base import doc_inherit
import scipy.linalg as la


class RobustPCAModel(PCAVectorModel, VectorizableBackedModel):
    r"""
    A :map:`MeanLinearModel` where components are Principal Components
    and the components are vectorized instances.

    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`pca`.

    Parameters
    ----------
    samples : `list` or `iterable` of :map:`Vectorizable`
        List or iterable of samples to build the model from.
    centre : `bool`, optional
        When ``True`` (default) PCA is performed after mean centering the data.
        If ``False`` the data is assumed to be centred, and the mean will be
        ``0``.
    n_samples : `int`, optional
        If provided then ``samples``  must be an iterator that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).
    max_n_components : `int`, optional
        The maximum number of components to keep in the model. Any components
        above and beyond this one are discarded.
    inplace : `bool`, optional
        If ``True`` the data matrix is modified in place. Otherwise, the data
        matrix is copied.
    verbose : `bool`, optional
        Whether to print building information or not.
     """

    def __init__(self, samples, centre=True, n_samples=None,
                 max_n_components=None, inplace=True, verbose=False):

        # build a data matrix from all the samples
        data, template = as_matrix(samples, length=n_samples,
                                   return_template=True, verbose=verbose)
        data, E = rpca_pcp(data)

        print E

        n_samples = data.shape[0]

        PCAVectorModel.__init__(self, data, centre=centre,
                                max_n_components=max_n_components,
                                n_samples=n_samples, inplace=inplace)
        VectorizableBackedModel.__init__(self, template)

    @classmethod
    def init_from_covariance_matrix(cls, C, mean, n_samples, centred=True,
                                    max_n_components=None):
        r"""
        Build the Principal Component Analysis (PCA) by eigenvalue
        decomposition of the provided covariance/scatter matrix. For details
        of the implementation of PCA, see :map:`pcacov`.

        Parameters
        ----------
        C : ``(n_features, n_features)`` `ndarray`
            The Covariance/Scatter matrix, where `N` is the number of features.
        mean : :map:`Vectorizable`
            The mean instance. It must be a :map:`Vectorizable` and *not* an
            `ndarray`.
        n_samples : `int`
            The number of samples used to generate the covariance matrix.
        centred : `bool`, optional
            When ``True`` we assume that the data were centered before
            computing the covariance matrix.
        max_n_components : `int`, optional
            The maximum number of components to keep in the model. Any
            components above and beyond this one are discarded.
        """
        # Create new pca instance
        self_model = PCAVectorModel.__new__(cls)
        self_model.n_samples = n_samples

        # Compute pca on covariance
        e_vectors, e_values = pcacov(C)

        # The call to __init__ of MeanLinearModel is done in here
        self_model._constructor_helper(eigenvalues=e_values,
                                       eigenvectors=e_vectors,
                                       mean=mean.as_vector(),
                                       centred=centred,
                                       max_n_components=max_n_components)
        VectorizableBackedModel.__init__(self_model, mean)
        return self_model

    @classmethod
    def init_from_components(cls, components, eigenvalues, mean, n_samples,
                             centred, max_n_components=None):
        r"""
        Build the Principal Component Analysis (PCA) using the provided
        components (eigenvectors) and eigenvalues.

        Parameters
        ----------
        components : ``(n_components, n_features)`` `ndarray`
            The eigenvectors to be used.
        eigenvalues : ``(n_components, )`` `ndarray`
            The corresponding eigenvalues.
        mean : :map:`Vectorizable`
            The mean instance. It must be a :map:`Vectorizable` and *not* an
            `ndarray`.
        n_samples : `int`
            The number of samples used to generate the eigenvectors.
        centred : `bool`, optional
            When ``True`` we assume that the data were centered before
            computing the eigenvectors.
        max_n_components : `int`, optional
            The maximum number of components to keep in the model. Any
            components above and beyond this one are discarded.
        """
        # Create new pca instance
        self_model = PCAVectorModel.__new__(cls)
        self_model.n_samples = n_samples

        # The call to __init__ of MeanLinearModel is done in here
        self_model._constructor_helper(
            eigenvalues=eigenvalues, eigenvectors=components,
            mean=mean.as_vector(), centred=centred,
            max_n_components=max_n_components)
        VectorizableBackedModel.__init__(self_model, mean)
        return self_model

    def mean(self):
        r"""
        Return the mean of the model.

        :type: :map:`Vectorizable`
        """
        return self.template_instance.from_vector(self._mean)

    @property
    def mean_vector(self):
        r"""
        Return the mean of the model as a 1D vector.

        :type: `ndarray`
        """
        return self._mean

    @doc_inherit(name='project_out')
    def project_out_vector(self, instance_vector):
        return PCAVectorModel.project_out(self, instance_vector)

    @doc_inherit(name='reconstruct')
    def reconstruct_vector(self, instance_vector):
        return PCAVectorModel.reconstruct(self, instance_vector)

    @doc_inherit(name='project')
    def project_vector(self, instance_vector):
        return PCAVectorModel.project(self, instance_vector)

    @doc_inherit(name='instance')
    def instance_vector(self, weights, normalized_weights=False):
        return PCAVectorModel.instance(self, weights,
                                       normalized_weights=normalized_weights)

    @doc_inherit(name='component')
    def component_vector(self, index, with_mean=True, scale=1.0):
        return PCAVectorModel.component(self, index, with_mean=with_mean,
                                        scale=scale)

    @doc_inherit(name='project_whitened')
    def project_whitened_vector(self, vector_instance):
        return PCAVectorModel.project_whitened(self, vector_instance)

    def component(self, index, with_mean=True, scale=1.0):
        r"""
        Return a particular component of the linear model.

        Parameters
        ----------
        index : `int`
            The component that is to be returned
        with_mean: `bool`, optional
            If ``True``, the component will be blended with the mean vector
            before being returned. If not, the component is returned on it's
            own.
        scale : `float`, optional
            A scale factor that should be applied to the component. Only
            valid in the case where ``with_mean == True``. See
            :meth:`component_vector` for how this scale factor is interpreted.

        Returns
        -------
        component : `type(self.template_instance)`
            The requested component instance.
        """
        return self.template_instance.from_vector(self.component_vector(
            index, with_mean=with_mean, scale=scale))

    def instance(self, weights, normalized_weights=False):
        """
        Creates a new instance of the model using the first ``len(weights)``
        components.

        Parameters
        ----------
        weights : ``(n_weights,)`` `ndarray` or `list`
            ``weights[i]`` is the linear contribution of the i'th component
            to the instance vector.
        normalized_weights : `bool`, optional
            If ``True``, the weights are assumed to be normalized w.r.t the
            eigenvalues. This can be easier to create unique instances by
            making the weights more interpretable.
        Raises
        ------
        ValueError
            If n_weights > n_components

        Returns
        -------
        instance : `type(self.template_instance)`
            An instance of the model.
        """
        v = self.instance_vector(weights, normalized_weights=normalized_weights)
        return self.template_instance.from_vector(v)

    def project_whitened(self, instance):
        """
        Projects the `instance` onto the whitened components, retrieving the
        whitened linear weightings.

        Parameters
        ----------
        instance : :map:`Vectorizable`
            A novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of whitened linear weightings
        """
        return self.project_whitened_vector(instance.as_vector())

    def increment(self, samples, n_samples=None, forgetting_factor=1.0,
                  verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        ----------
        samples : `list` of :map:`Vectorizable`
            List of new samples to update the model from.
        n_samples : `int`, optional
            If provided then ``samples``  must be an iterator that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples. If 1.0, all samples are weighted equally
            and, hence, the results is the exact same as performing batch
            PCA on the concatenated list of old and new simples. If <1.0,
            more emphasis is put on the new samples. See [1] for details.

        References
        ----------
        .. [1] David Ross, Jongwoo Lim, Ruei-Sung Lin, Ming-Hsuan Yang.
           "Incremental Learning for Robust Visual Tracking". IJCV, 2007.
        """
        # build a data matrix from the new samples
        data = as_matrix(samples, length=n_samples, verbose=verbose)
        n_new_samples = data.shape[0]
        PCAVectorModel.increment(self, data, n_samples=n_new_samples,
                                 forgetting_factor=forgetting_factor,
                                 verbose=verbose)

    def __str__(self):
        str_out = 'PCA Model \n'                             \
                  ' - instance class:       {}\n'            \
                  ' - centred:              {}\n'            \
                  ' - # features:           {}\n'            \
                  ' - # active components:  {}\n'            \
                  ' - kept variance:        {:.2}  {:.1%}\n' \
                  ' - noise variance:       {:.2}  {:.1%}\n' \
                  ' - total # components:   {}\n'            \
                  ' - components shape:     {}\n'.format(
            type(self.template_instance), self.centred,  self.n_features,
            self.n_active_components, self.variance(), self.variance_ratio(),
            self.noise_variance(), self.noise_variance_ratio(),
            self.n_components, self.components.shape)
        return str_out


def rpca_alm(X, lmbda=None, tol=1e-7, max_iters=1000, verbose=True,
             inexact=True):
    """
    Augmented Lagrange Multiplier
    """
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(X.shape[0])

    Y = np.sign(X)
    norm_two = svd(Y, 1)[1]
    norm_inf = np.abs(Y).max() / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm

    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)

    dnorm = la.norm(X, ord='fro')
    tol_primal = 1e-6 * dnorm
    total_svd = 0
    mu = 0.5 / norm_two
    rho = 6

    sv = 5
    n = Y.shape[0]

    for iter1 in xrange(max_iters):
        primal_converged = False
        sv = sv + np.round(n * 0.1)
        primal_iter = 0

        while not primal_converged:
            Eraw = X - A + (1/mu) * Y
            Eupdate = np.maximum(
                Eraw - lmbda/mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
            U, S, V = svd(X - Eupdate + (1 / mu) * Y, sv)

            svp = (S > 1/mu).sum()
            if svp < sv:
                sv = np.min([svp + 1, n])
            else:
                sv = np.min([svp + round(.05 * n), n])

            Aupdate = np.dot(
                np.dot(U[:, :svp], np.diag(S[:svp] - 1/mu)), V[:svp, :])

            if primal_iter % 10 == 0 and verbose >= 2:
                print(la.norm(A - Aupdate, ord='fro'))

            if ((la.norm(A - Aupdate, ord='fro') < tol_primal and
                la.norm(E - Eupdate, ord='fro') < tol_primal) or
                (inexact and primal_iter > 5)):
                primal_converged = True

            A = Aupdate
            E = Eupdate
            primal_iter += 1
            total_svd += 1

        Z = X - A - E
        Y = Y + mu * Z
        mu *= rho

        if la.norm(Z, ord='fro') / dnorm < tol:
            if verbose:
                print('\nConverged at iteration {}'.format(iter1))
            break

        if verbose:
            _verbose(A, E, X)

    return A, E


def rpca_pcp(X, lamda=None, max_iters=1000, tol=1.0e-7, verbose=True):
    m, n = X.shape
    # Set params
    if lamda is None:
        lamda = 1.0 / np.sqrt(min(m, n))
    # Initialize
    Y = X
    u, s, v = svd(Y, k=1)
    norm_two = s[0]
    norm_inf = la.norm(Y.ravel(), ord=np.inf) / lamda
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    A_hat = np.zeros((m, n))
    mu = 1.25/norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(X, 'fro')

    num_iters = 0
    total_svd = 0
    sv = 10
    while True:
        num_iters += 1

        temp_T = X - A_hat + (1/mu)*Y
        E_hat = np.maximum(temp_T - lamda/mu, 0)
        E_hat = E_hat + np.minimum(temp_T + lamda/mu, 0)

        u, s, v = svd(X - E_hat + (1/mu)*Y, k=sv)
        svp = np.sum(s > 1/mu)

        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05*n), n)

        A_hat = np.dot(
            np.dot(
                u[:, :svp],
                np.diag(s[:svp] - 1 / mu)
            ),
            v[:svp, :]
        )

        total_svd += 1

        Z = X - A_hat - E_hat

        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if verbose:
            _verbose(A_hat, E_hat, X)

        if (la.norm(Z, ord='fro') / d_norm < tol) or num_iters >= max_iters:
            return A_hat, E_hat


def svd(X, k=-1):
    U, S, V = la.svd(X, full_matrices=False)
    if k < 0:
        return U, S, V
    else:
        return U[:, :k], S[:k], V[:k, :]


def _verbose(A, E, D):
    A_rank = np.linalg.matrix_rank(A)
    perc_E = (np.count_nonzero(E) / E.size) * 100
    error = la.norm(D - A - E, ord='fro')
    print_dynamic('rank(A): {}, |E|_1: {:.2f}%, |D-A-E|_F: {:.2e}'.format(
        A_rank, perc_E, error))
