import os
import tempfile
import shutil
import multiprocessing as mp
from joblib import Parallel, delayed, dump, load
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from . import utils as ut

__all__ = ['gen_perm']


def gen_perm(nz, ny, nx, length, sigma, nperm, ret=False,
             verbose=True):
    """
    Generate nperm perm fields of size (nz, ny,nx).
    Results are stored in perms/ directory.
    Return: output directory name [string] or K_samples
    """

    assert nz == 1, "3D not supported yet"

    length = float(length)
    sigma = float(sigma)

    # construct the coordinates of each cell
    Lx, Ly = 1.0, 1.0  # unit square only
    dx, dy = Lx/nx, Ly/ny
    x, y = np.meshgrid((np.arange(nx)+.5)*dx,
                       (np.arange(ny)+.5)*dy)
    coords = zip(x.ravel(), y.ravel())
    dists = cdist(coords, coords)
    # cov_mtx = gauss_rbf(dists, eps, sigma)
    cov_mtx = mycov(dists, length, sigma)
    # A, Q = sp.linalg.eigh(cov_mtx)
    # A[np.where(A<np.finfo(float).eps)] = np.finfo(float).eps
    # corr_mtx = Q.dot(np.diag(np.sqrt(A))).dot(Q.T)
    corr_mtx = sp.linalg.cholesky(cov_mtx, lower=True)

    # dump corr_mtx to disk to free memory and reference to mmap array
    folder = tempfile.mkdtemp()
    tmp = os.path.join(folder, 'corr_mtx')
    dump(corr_mtx, tmp)
    corr_mtx = load(tmp, mmap_mode='r')

    # create output directory
    rootdir = 'perm'
    permdir = ut.unparse_perm(nz, ny, nx, length, sigma, nperm)
    odir = os.path.join(rootdir, permdir)
    ut.mkdir_p(odir)

    # create memmap for output
    K_samples = np.memmap(os.path.join(odir,'K_samples'),
                          dtype=float,
                          shape=(nperm,nz,ny,nx,3),
                          mode='w+')

    # if verbose:
    #     print '\nGenerating %d samples of size %dx%dx%dx%d' %(nperm,
    #                                                           nz,ny,nx,3)
    #     print 'Correlation length:', length
    #     print 'Sigma:', sigma
    #     print 'Storing in', odir

    # fire off workers
    batch_size = max(10, nperm/mp.cpu_count())
    Parallel(n_jobs=-1, batch_size=batch_size)\
        (delayed(_task)(i, ny, nx, corr_mtx, K_samples)
         for i in xrange(nperm))

    # if verbose:
    #     print 'Done!!!'
    #     print 'Output filename:', odir

    # try:
    #     shutil.rmtree(folder)
    # except:
    #     print("Failed to delete: " + folder)

    if ret:
        return K_samples


def _task(i, ny, nx, corr_matrix, K_samples):
    np.random.seed()
    R = np.random.randn(ny,nx)
    # R = np.random.uniform(-1,1,(ny,nx))
    z = np.reshape(corr_matrix.dot(R.ravel()), (ny,nx))
    # for now, assume isoparametric
    K = np.rollaxis(np.array([z]*3), axis=0, start=3)
    K = np.expand_dims(K, axis=0)
    K_samples[i] = K


# custom RBF
def gauss_rbf(d, eps, sigma):
    return sigma**2*np.exp(-0.5*(d/np.float64(eps))**2)


# spatial covariance
def mycov(d, length, sigma):
    return sigma**2*np.exp(-d/np.float(length))
