from menpo.shape import PointCloud, TriMesh
from menpo.transform.groupalign.base import MultipleAlignment
from menpo.math import pca
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


# ICP --------------------------------------------
class SICP(MultipleAlignment):
    def __init__(self, sources, target=None):
        self._test_iteration = []
        self.transformations = []
        self.point_correspondence = []

        # sort sources in number of points
        sources = np.array(sources)
        sortindex = np.argsort(np.array([s.n_points for s in sources]))[-1::-1]
        sort_sources = sources[sortindex]

        # Set first source as target (e.g. having most number of points)
        if target is None:
            target = sort_sources[0]

        super(SICP, self).__init__(sources, target)

        # Align Source with Target
        self.aligned_shapes = np.array(
            [self._align_source(s) for s in sources]
        )

    def _align_source(self, source, eps=1e-3, max_iter=100):

        # Initial Alignment using PCA
        # p0, r, sm, tm = self._pca_align(source)
        # transforms.append([r, sm, tm])
        p0 = source.points

        a_p, transforms, iters, point_corr = self._align(p0, eps, max_iter)
        iters = [source.points, p0] + iters

        self._test_iteration.append(iters)
        self.transformations.append(transforms)
        self.point_correspondence.append(point_corr)

        return PointCloud(a_p)

    def _align(self, i_s, eps, max_iter):
        # Align Shapes
        transforms = []
        iters = []
        it = 0
        pf = i_s
        n_p = i_s.shape[0]
        tolerance_old = tolerance = eps + 1
        while tolerance > eps and it < max_iter:
            pk = pf

            # Compute Closest Points
            yk, _ = self._cloest_points(pk)

            # Compute Registration
            pf, rot, smean, tmean = self._compute_registration(pk, yk)
            transforms.append([rot, smean, tmean])

            # Update source
            # pf = self._update_source(pk, np.hstack((qr, qt)))

            # Calculate Mean Square Matching Error
            tolerance_new = np.sum(np.power(pf - yk, 2)) / n_p
            tolerance = abs(tolerance_old - tolerance_new)
            tolerance_old = tolerance_new

            it += 1
            iters.append(pf)

        _, point_corr = self._cloest_points(pf)

        return pf, transforms, iters, point_corr

    def _pca_align(self, source):
        # Apply PCA on both source and target
        svecs, svals, smean = pca(source.points)
        tvecs, tvals, tmean = pca(self.target.points)

        # Compute Rotation
        svec = svecs[np.argmax(svals)]
        tvec = tvecs[np.argmax(tvals)]

        sang = np.arctan2(svec[1], svec[0])
        tang = np.arctan2(tvec[1], tvec[0])

        da = sang - tang

        tr = np.array([[np.cos(da), np.sin(da)],
                       [-1*np.sin(da), np.cos(da)]])

        # Compute Aligned Point
        pt = np.array([tr.dot(s - smean) + tmean for s in source.points])

        return pt, tr, smean, tmean

    def _update_source(self, p, q):
        return _apply_q(p, q)[:, :p.shape[1]]

    def _compute_registration(self, p, x):
        # Calculate Covariance
        up = np.mean(p, axis=0)
        ux = np.mean(x, axis=0)
        u = up[:, None].dot(ux[None, :])
        n_p = p.shape[0]
        cov = sum([pi[:, None].dot(xi[None, :])
                   for (pi, xi) in zip(p, x)]) / n_p - u

        # Apply SVD
        U, W, T = np.linalg.svd(cov)

        # Calculate Rotation Matrix
        qr = T.T.dot(U.T)
        # Calculate Translation Point
        pk = np.array([qr.dot(s - up) + ux for s in p])

        return pk, qr, up, ux

    def _cloest_points(self, source, target=None):
        points = np.array([self._closest_node(s, target) for s in source])

        return np.vstack(points[:, 0]), np.hstack(points[:, 1])

    def _closest_node(self, node, target=None):
        if target is None:
            target = self.target

        nodes = target
        if isinstance(target, PointCloud):
            nodes = np.array(target.points)

        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        index = np.argmin(dist_2)
        return nodes[index], index


class SNICP(SICP):
    def __init__(self, sources, target=None, max_iter=100):
        self.n_dims = sources[0].n_dims
        super(SNICP, self).__init__(sources, target)

    def _align(self, tplt, eps, max_iter):

        # Configuration
        higher = 2001
        lower = 1
        step = 100
        transforms = []
        iters = []

        # Build TriMesh Source
        tplt_tri = TriMesh(tplt).trilist

        # Generate Edge List
        tplt_edge = tplt_tri[:, [0, 1]]
        tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [0, 2]]))
        tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [1, 2]]))
        tplt_edge = np.sort(tplt_edge)

        # Get Unique Edge List
        b = np.ascontiguousarray(tplt_edge).view(
            np.dtype((np.void, tplt_edge.dtype.itemsize * tplt_edge.shape[1]))
        )
        _, idx = np.unique(b, return_index=True)
        tplt_edge = tplt_edge[idx]

        # init
        m = tplt_edge.shape[0]
        n = tplt.shape[0]

        # get node-arc incidence matrix
        M = np.zeros((m, n))
        M[range(m), tplt_edge[:, 0]] = -1
        M[range(m), tplt_edge[:, 1]] = 1

        # weight matrix
        G = np.identity(self.n_dims+1)

        # build the kD-tree
        target_2d = self.target.points
        kdOBJ = KDTree(target_2d)

        # init tranformation
        prev_X = np.zeros((self.n_dims, self.n_dims+1))
        prev_X = np.tile(prev_X, n).T
        tplt_i = tplt

        # start nicp
        # for each stiffness
        sf = np.logspace(lower, higher, num=step, base=1.005)[-1::-1]
        sf = [10 ** i for i in range(5, 0, -1)] + range(9,1,-1) + [1.0/(i+1.0) for i in range(10)]
        print sf
        sf_kron = np.kron(M, G)
        errs = []

        for alpha in sf:
            # get the term for stiffness
            sf_term = alpha*sf_kron
            # iterate until X converge
            niters = 0
            while niters < max_iter:
                # find nearest neighbour
                _, match = kdOBJ.query(tplt_i)

                # formulate target and template data, and distance term
                U = target_2d[match, :]

                point_size = self.n_dims+1
                D = np.zeros((n, n*point_size))
                for k in range(n):
                    D[k, k*point_size:k*point_size+2] = tplt_i[k, :]
                    D[k, k*point_size+2] = 1

                # % correspondence detection for setting weight
                # add distance term
                sA = np.vstack((sf_term, D))
                sB = np.vstack((np.zeros((sf_term.shape[0], self.n_dims)), U))
                sX = np.linalg.pinv(sA).dot(sB)

                # deform template
                tplt_i = D.dot(sX)
                err = np.linalg.norm(prev_X-sX, ord='fro')
                errs.append([alpha, err])
                prev_X = sX

                transforms.append(sX)
                iters.append(tplt_i)

                niters += 1

                if err/np.sqrt(np.size(prev_X)) < eps:
                    break

        # final result
        fit_2d = tplt_i
        _, point_corr = kdOBJ.query(fit_2d)
        return fit_2d, transforms, iters, point_corr


def _compose_r(qr):
    q0, q1, q2, q3 = qr
    r = np.zeros((3, 3))
    r[0, 0] = np.sum(np.power(qr, 2)) * [1, 1, -1, -1]
    r[1, 1] = np.sum(np.power(qr[[0, 2, 1, 3]], 2)) * [1, 1, -1, -1]
    r[2, 2] = np.sum(np.power(qr[[0, 3, 1, 2]], 2)) * [1, 1, -1, -1]
    r[0, 1] = 2 * (q1 * q2 - q0 * q3)
    r[1, 0] = 2 * (q1 * q2 + q0 * q3)
    r[0, 2] = 2 * (q1 * q3 + q0 * q2)
    r[2, 0] = 2 * (q1 * q3 - q0 * q2)
    r[1, 2] = 2 * (q2 * q3 - q0 * q1)
    r[2, 1] = 2 * (q2 * q3 + q0 * q1)

    return r


def _apply_q(source, q):
    if source.shape[1] == 2:
        source = np.hstack((source, np.zeros((source.shape[0], 1))))

    r = _compose_r(q[:4])
    t = q[4:]
    s1 = [r.dot(s) + t for s in source]
    return np.array(s1)

# END ICP ----------------------------------------


def _pca_align(self, source):
    # Apply PCA on both source and target
    svecs, svals, smean = pca(source.points)
    tvecs, tvals, tmean = pca(self.target.points)

    # Compute Rotation
    svec = svecs[np.argmax(svals)]
    tvec = tvecs[np.argmax(tvals)]

    sang = np.arctan2(svec[1], svec[0])
    tang = np.arctan2(tvec[1], tvec[0])

    da = sang - tang

    tr = np.array([[np.cos(da), np.sin(da)],
                   [-1*np.sin(da), np.cos(da)]])

    # Compute Aligned Point
    pt = np.array([tr.dot(s - smean) + tmean for s in source.points])

    return pt, tr, smean, tmean


def _update_source(p, q):
    return _apply_q(p, q)[:, :p.shape[1]]


def _compute_registration(p, x):
    # Calculate Covariance
    up = np.mean(p, axis=0)
    ux = np.mean(x, axis=0)
    u = up[:, None].dot(ux[None, :])
    n_p = p.shape[0]
    cov = sum([pi[:, None].dot(xi[None, :])
               for (pi, xi) in zip(p, x)]) / n_p - u

    # Apply SVD
    U, W, T = np.linalg.svd(cov)

    # Calculate Rotation Matrix
    qr = T.T.dot(U.T)
    # Calculate Translation Point
    pk = np.array([qr.dot(s - up) + ux for s in p])

    return pk, qr, up, ux


def _cloest_points(source, target):
    points = np.array([_closest_node(s, target) for s in source])

    return np.vstack(points[:, 0]), np.hstack(points[:, 1])


def _closest_node(node, target):

    nodes = target
    if isinstance(target, PointCloud):
        nodes = np.array(target.points)

    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    index = np.argmin(dist_2)
    return nodes[index], index


def icp(source, target, eps=1e-3, max_iter=100):
    # Align Shapes
    transforms = []
    iters = []
    it = 0
    pf = source
    n_p = source.shape[0]
    tolerance_old = tolerance = eps + 1
    kdtree = KDTree(target)
    while tolerance > eps and it < max_iter:
        pk = pf

        # Compute Closest Points
        match = kdtree.query(pk)[1]

        # Compute Registration
        pf, rot, smean, tmean = _compute_registration(pk, target[match, :])
        transforms.append([rot, smean, tmean])

        # Update source
        # pf = self._update_source(pk, np.hstack((qr, qt)))

        # Calculate Mean Square Matching Error
        tolerance_new = np.sum(np.power(pf - target[match, :], 2)) / n_p
        tolerance = abs(tolerance_old - tolerance_new)
        tolerance_old = tolerance_new

        it += 1
        iters.append(pf)

    point_corr = kdtree.query(pf)[1]

    return pf, (transforms, iters, point_corr)


def nicp(source, target, eps=1e-3, us=101, ls=1, step=5, max_iter=100):
    r"""
    Deforms the source trimesh to align with to optimally the target.
    """
    n_dims = source.n_dims
    # Homogeneous dimension (1 extra for translation effects)
    h_dims = n_dims + 1
    points = source.points
    trilist = source.trilist

    # Configuration
    upper_stiffness = us
    lower_stiffness = ls
    stiffness_step = step

    # Get a sorted list of edge pairs (note there will be many mirrored pairs
    # e.g. [4, 7] and [7, 4])
    edge_pairs = np.sort(np.vstack((trilist[:, [0, 1]],
                                    trilist[:, [0, 2]],
                                    trilist[:, [1, 2]])))

    # We want to remove duplicates - this is a little hairy, but basically we
    # get a view on the array where each pair is considered by numpy to be
    # one item
    edge_pair_view = np.ascontiguousarray(edge_pairs).view(
        np.dtype((np.void, edge_pairs.dtype.itemsize * edge_pairs.shape[1])))
    # Now we can use this view to ask for only unique edges...
    unique_edge_index = np.unique(edge_pair_view, return_index=True)[1]
    # And use that to filter our original list down
    unique_edge_pairs = edge_pairs[unique_edge_index]

    # record the number of unique edges and the number of points
    n = points.shape[0]
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    row = np.hstack((np.arange(m), np.arange(m)))
    col = unique_edge_pairs.T.ravel()
    data = np.hstack((-1 * np.ones(m), np.ones(m)))
    M_s = sp.coo_matrix((data, (row, col)))

    # weight matrix
    G = np.identity(n_dims + 1)

    M_kron_G_s = sp.kron(M_s, G)

    # build the kD-tree
    # print('building KD-tree for target...')
    kdtree = KDTree(target.points)

    # init transformation
    X_prev = np.zeros((n_dims, n_dims + 1))
    X_prev = np.tile(X_prev, n).T
    v_i = points

    # start nicp
    # for each stiffness
    # stiffness = range(upper_stiffness, lower_stiffness, -stiffness_step)
    stiffness = np.logspace(lower_stiffness, upper_stiffness, num=stiffness_step, base=1.005)[-1::-1]
    stiffness = [10 ** i for i in range(5, 0, -1)] + range(9,1,-1) + [1.0/(i+1.0) for i in range(10)]
    errs = []


    # we need to prepare some indices for efficient construction of the D
    # sparse matrix.
    row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(),
                     np.arange(n)))

    x = np.arange(n * h_dims).reshape((n, h_dims))
    col = np.hstack((x[:, :n_dims].ravel(),
                     x[:, n_dims]))

    o = np.ones(n)
    iterations = [v_i]
    for alpha in stiffness:
        # print(alpha)
        # get the term for stiffness
        alpha_M_kron_G_s = alpha * M_kron_G_s

        # iterate until X converge
        iter = 0
        while iter < max_iter:
            # find nearest neighbour
            match = kdtree.query(v_i)[1]

            # formulate target and template data, and distance term
            U = target.points[match, :]

            data = np.hstack((v_i.ravel(), o))
            D_s = sp.coo_matrix((data, (row, col)))

            # correspondence detection for setting weight
            # add distance term
            A_s = sp.vstack((alpha_M_kron_G_s, D_s)).tocsr()
            B_s = sp.vstack((np.zeros((alpha_M_kron_G_s.shape[0], n_dims)),
                             U)).tocsr()
            X_s = spsolve(A_s.T.dot(A_s), A_s.T.dot(B_s))
            X = X_s.toarray()

            # deform template
            v_i = D_s.dot(X)
            err = np.linalg.norm(X_prev - X, ord='fro')
            errs.append([alpha, err])
            X_prev = X

            iter += 1

            if err / np.sqrt(np.size(X_prev)) < eps:
                iterations.append(v_i)
                break

    # final result
    point_corr = kdtree.query(v_i)[1]
    return (v_i, iterations), point_corr
