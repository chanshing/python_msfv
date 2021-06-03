from __future__ import division
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from .utils import compute_trans

# from scikits import umfpack

# import time

__all__ = ['tpfa']


def tpfa(Grid, K, q0, dirichlet=None, flux=False, timer=None):  # Return P, V

    # Compute transmissibilities by harmonic averaging.
    TX,TY,TZ = compute_trans(Grid, K)

    nx = Grid['nx']
    ny = Grid['ny']
    nz = Grid['nz']
    N = nx*ny*nz

    # Assemble TPFA discretization matrix.
    x1 = np.reshape(TX[:,:,0:nx],N); x2 = np.reshape(TX[:,:,1:nx+1],N)
    y1 = np.reshape(TY[:,0:ny,:],N); y2 = np.reshape(TY[:,1:ny+1,:],N)
    z1 = np.reshape(TZ[0:nz,:,:],N); z2 = np.reshape(TZ[1:nz+1,:,:],N)
    DiagVecs = np.array([-z2,-y2,-x2,x1+x2+y1+y2+z1+z2,-x1,-y1,-z1])
    DiagIndx = np.array([-nx*ny,-nx,-1,0,1,nx,nx*ny])

    # Eliminate any zero vectors present
    nonzero_diags_indices = ~np.all(DiagVecs==0, axis=1)
    DiagVecs = DiagVecs[nonzero_diags_indices]
    DiagIndx = DiagIndx[nonzero_diags_indices]
    A = spa.spdiags(DiagVecs, DiagIndx, N, N).tocsr()

    q=np.copy(q0)
    # Impose boundary conditions.
    if dirichlet is not None:
        large_number = 1e16
        idxs = dirichlet[:,0].astype('int')
        vals = dirichlet[:,1]
        A[idxs,idxs] = large_number
        q[idxs] = large_number*vals
    else:
        A[0,0] = A[0,0]+sum(K[0,0,0,:])

    P = spla.spsolve(A,q)
    P = np.reshape(P, (nz, ny, nx))

    # q=np.copy(q0)
    # # # Impose boundary conditions.
    # # large_number = 1e12*A.sum().sum()
    # # large_number = sum(K[0,0,0,:])
    # large_number = 1e12
    # # print('large_number %f' % large_number)
    # if dirichlet is not None:
    #     # large_number = 1e16
    #     idxs = dirichlet[:, 0].astype('int')
    #     vals = dirichlet[:, 1]
    #     A[idxs, idxs] =  large_number
    #     # print 'pre', A[idxs, idxs]
    #     # A[idxs, idxs] = A[idxs, idxs] + large_number
    #     # print 'post', A[idxs, idxs]
    #     # print vals
    #     q[idxs] = large_number*vals
    #     # for i, v in zip(idxs, vals):
    #     #     q[i] = A[i,i]*v
    #     # print 'q', q
    # else:
    #     A[0,0] = A[0,0]+sum(K[0,0,0,:])

    # # # Solve linear system
    # # start_t = time.time()
    # # P = spla.spsolve(A,q)
    # # end_t = time.time()
    # # if timer: timer['mytime'] += end_t - start_t
    # # P = np.reshape(P,(nz,ny,nx))

    # # Solve linear system
    # start_t = time.time()
    # # P = spla.spsolve(A, q)
    # # P0 = spla.spsolve(A, q)
    # # P, _ = spla.gmres(A, q, tol=1e-5)
    # P = umfpack.spsolve(A, q)
    # # err = np.linalg.norm(P0-P)
    # # if err > 1e-2:
    # #     print err
    # #     time.sleep(0.1)
    # end_t = time.time()
    # if timer:
    #     timer['mytime'] += end_t - start_t
    # P = np.reshape(P, (nz, ny, nx))

    if flux:
        # Extract interface fluxes.
        V = {}
        V['x'] = np.zeros((nz,ny,nx+1))
        V['y'] = np.zeros((nz,ny+1,nx))
        V['z'] = np.zeros((nz+1,ny,nx))
        V['x'][:,:,1:nx] = (P[:,:,0:nx-1]-P[:,:,1:nx])*TX[:,:,1:nx]
        V['y'][:,1:ny,:] = (P[:,0:ny-1,:]-P[:,1:ny,:])*TY[:,1:ny,:]
        V['z'][1:nz,:,:] = (P[0:nz-1,:,:]-P[1:nz,:,:])*TZ[1:nz,:,:]
        return P, V
    else:
        return P
