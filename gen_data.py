import os
import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
from .geometry import cart_grid, coarse_grid, dual_grid
from .geometry import csi
from .bases_solver import compute_bases
from . import utils as ut

__all__ = ['gen_data']


def gen_data(permdir, ncg, dim=1, ret=False, nperm=None):

    args = ut.parse_perm(permdir)
    nz = args['nz']
    ny = args['ny']
    nx = args['nx']
    length = args['length']
    sigma = args['sigma']
    (Ny, Nx) = ncg

    # if not specified, use all samples
    if nperm is None:
        nperm = args['nperm']

    # check everything is fine
    assert nperm <= args['nperm']
    assert nz == 1, "3D not supported yet"
    assert ny == nx, "Square grid only"
    assert Ny == Nx, "Square grid only"
    assert (nx % Nx == 0) and (nx/Nx) % 2 != 0,\
        "Invalid coarse grid size"

    # create output directory
    rootdir = 'data'
    datadir = ut.unparse_dataset(Ny, Nx, nz, ny, nx, length, sigma, nperm)
    odir = os.path.join(rootdir, datadir)
    ut.mkdir_p(odir)

    # generate grids
    G = cart_grid([ny,nx])
    CG = coarse_grid(G, [Ny,Nx])
    DG = dual_grid(G, CG)

    # load permeability samples
    K_samples = np.memmap(os.path.join(permdir, 'K_samples'),
                          dtype=float,
                          shape=(nperm,nz,ny,nx,3),
                          mode='r')

    # create memmaps for the patches
    shp = ut.get_mm_shapes(G, CG, nperm, dim)

    X_corner = np.memmap(os.path.join(odir,'X_corner'),
                         dtype=float,
                         shape=shp['corX'],
                         mode='w+')
    Y_corner = np.memmap(os.path.join(odir,'Y_corner'),
                         dtype=float,
                         shape=shp['corY'],
                         mode='w+')
    X_side = np.memmap(os.path.join(odir,'X_side'),
                       dtype=float,
                       shape=shp['sidX'],
                       mode='w+')
    Y_side = np.memmap(os.path.join(odir,'Y_side'),
                       dtype=float,
                       shape=shp['sidY'],
                       mode='w+')
    X_inner = np.memmap(os.path.join(odir,'X_inner'),
                        dtype=float,
                        shape=shp['innX'],
                        mode='w+')
    Y_inner = np.memmap(os.path.join(odir,'Y_inner'),
                        dtype=float,
                        shape=shp['innY'],
                        mode='w+')

    # print '\nReading from:', permdir
    # print 'Size of perm field: %dx%dx%dx%d' %(nz,ny,nx,3)
    # print 'Generating patches by coarsening into: %dx%d' %(Ny,Nx)

    # fire off workers
    batch_size = max(10, nperm/mp.cpu_count())
    Parallel(n_jobs=-1, batch_size=batch_size)\
        (delayed(_task)
         (i, G, CG, DG, K_samples, dim,
          X_corner, Y_corner,
          X_side, Y_side,
          X_inner, Y_inner)
         for i in xrange(nperm))

    # print 'Done!!!'
    # print 'Output directory:', odir

    if ret:
        # return as (n_samples, my, mx, dim)
        shp = ut.get_mm_shapes(G, CG, nperm, dim, per_perm_sample=False)
        X_inner = X_inner.reshape(shp['innX'])
        Y_inner = Y_inner.reshape(shp['innY'])
        X_side = X_side.reshape(shp['sidX'])
        Y_side = Y_side.reshape(shp['sidY'])
        X_corner = X_corner.reshape(shp['corX'])
        Y_corner = Y_corner.reshape(shp['corY'])
        return {'X_inner': X_inner, 'Y_inner': Y_inner,
                'X_side': X_side, 'Y_side': Y_side,
                'X_corner': X_corner, 'Y_corner': Y_corner}


def _task(i, G, CG, DG, K_samples, dim,
          X_corner, Y_corner,
          X_side, Y_side,
          X_inner, Y_inner):
    """ task for our slaves """
    K = np.exp(K_samples[i])
    bs = ut.crop_bases(CG, DG, compute_bases(G, CG, DG, K))
    # Ks = get_mini_Ks(K, DG)
    Ks = ut.get_mini_Ks(K_samples[i], DG)

    cor, sid, inn = csi(CG)

    # corners
    X_corner[i] = [Ks[cor['bottom_left']][...,:dim],
                   ut.rot90(Ks[cor['bottom_right']][...,:dim], 1),
                   ut.rot90(Ks[cor['top_right']][...,:dim], 2),
                   ut.rot90(Ks[cor['top_left']][...,:dim], 3)]
    Y_corner[i] = [bs[cor['bottom_left']],
                   np.rot90(bs[cor['bottom_right']], 1),
                   np.rot90(bs[cor['top_right']], 2),
                   np.rot90(bs[cor['top_left']], 3)]

    # sides
    X_side[i] = [Ks[n][...,:dim] for n in sid['bottom']] + \
                [ut.rot90(Ks[n][...,:dim], 1) for n in sid['right']] + \
                [ut.rot90(Ks[n][...,:dim], 2) for n in sid['top']] + \
                [ut.rot90(Ks[n][...,:dim], 3) for n in sid['left']]
    Y_side[i] = [bs[n] for n in sid['bottom']] + \
                [np.rot90(bs[n], 1) for n in sid['right']] + \
                [np.rot90(bs[n], 2) for n in sid['top']] + \
                [np.rot90(bs[n], 3) for n in sid['left']]

    # inners
    X_inner[i] = [Ks[n][...,:dim] for n in inn]
    Y_inner[i] = [bs[n] for n in inn]
