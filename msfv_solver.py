from __future__ import division
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from .utils import compute_trans, interpolate, pu_scale
from .bases_solver import compute_bases
from .corr_solver import compute_corr
from . import predict_inner
from .predict_inner import predict
from .recon_flux import recon_flux


def solveMSFV(G, CG, DG, K, q, dirichlet=None, verbose=False,
              flux=False, model=None, scale=True, ret_all=False):
    """
    Return: result [dict]
    result:
        pressure [3Darray]
        coarsepressure [3Darray]
        bases [list of csr_matrix]
        correction [csr_matrix]
        coarsesystem [dict]
            A [csr_matrix]
            C [array]
            q [array]
    """

    # if verbose:
    #     print 'Solving basis and correction functions...'

    if model is None:
        bases = compute_bases(G, CG, DG, K, verbose)
    else:
        # do the naive thing for now
        bases = compute_bases(G, CG, DG, K, verbose)
        bases = predict(bases, G, CG, DG, K, model, verbose=verbose)
        if scale:
            pu_scale(bases, DG)  # in place modification

    corr = compute_corr(G, CG, DG, K, q, verbose)

    # Compute transmissibilities
    TX, TY, TZ = compute_trans(G, K)
    T = np.concatenate((TX.ravel(), TY.ravel(), TZ.ravel()))

    # Number of coarse cells
    N_cg = len(CG['centers'])

    # Coarse flux matrix
    A = spa.lil_matrix((N_cg, N_cg))

    # Correction vector
    C = np.zeros(N_cg)

    # Coarse source/sink vector
    q_cg = np.zeros(N_cg)

    # if verbose:
    #     print 'Building coarse system of equations...'

    for i in range(N_cg):
        neighbors = CG['neighbors'][i] + [i]
        edges, border_in, border_out = CG['borders'][i]

        # Flux by basis functions
        for k in neighbors:
            basis_in = bases[k][border_in].toarray().ravel()
            basis_out = bases[k][border_out].toarray().ravel()
            A[i, k] = A[i, k] + np.dot(basis_in - basis_out, T[edges])

        # Flux by correction function
        corr_in = corr[border_in].toarray().ravel()
        corr_out = corr[border_out].toarray().ravel()
        C[i] = np.dot(corr_in - corr_out, T[edges])

    # Compute q_cg
    for i in range(N_cg):
        q_cg[i] = np.sum(q[CG['cells'][i]])

    # Impose boundary conditions
    if dirichlet is not None:
        large_number = 1e200
        idxs = dirichlet[:, 0].astype('int')
        vals = dirichlet[:, 1]
        A[idxs, idxs] = large_number
        q_cg[idxs] = large_number * vals
    else:
        A[0, 0] = A[0, 0] + sum(K[0, 0, 0, :])

    A = A.tocsr()

    # if verbose:
    #     print 'Solving coarse system...'
    P_cg = spla.spsolve(A, q_cg - C)

    # if verbose:
    #     print 'Interpolating solution to fine grid...'
    P = interpolate(P_cg, G, CG, bases, corr)

    P = np.reshape(P, (G['nz'], G['ny'], G['nx']))
    P_cg = np.reshape(P_cg, (CG['nz'], CG['ny'], CG['nx']))

    if flux:
        # if verbose:
        #     print 'Computing flux...'
        V = recon_flux(P, G, CG, K, q)
    else:
        # if verbose:
        #     print 'Skipping flux computation...'
        V = {}

    if ret_all:
        result = {}
        result['pressure'] = P
        result['flux'] = V
        result['pressurecoarse'] = P_cg
        result['bases'] = bases
        result['correction'] = corr
        result['coarsesystem'] = {}
        result['coarsesystem']['A'] = A
        result['coarsesystem']['C'] = C
        result['coarsesystem']['q'] = q_cg
        return result
    else:
        return P
