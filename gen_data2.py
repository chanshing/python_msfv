import os
import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
from .geometry import cart_grid, coarse_grid, dual_grid
from .geometry import csi
from .bases_solver import compute_bases
from . import utils as ut

__all__ = ['gen_data2']


def gen_data2(K_samples, nz, ny, nx, Ny, Nx, outputdir,
              dim=1, ret=False, nperm=None):

    # generate grids
    G = cart_grid([ny,nx])
    CG = coarse_grid(G, [Ny,Nx])
    DG = dual_grid(G, CG)

    # create memmaps for the patches
    shp = ut.get_mm_shapes(G, CG, K_samples.shape[0], dim)

    X_corner = np.memmap(os.path.join(outputdir,'X_corner'),
                         dtype=float,
                         shape=shp['corX'],
                         mode='w+')
    Y_corner = np.memmap(os.path.join(outputdir,'Y_corner'),
                         dtype=float,
                         shape=shp['corY'],
                         mode='w+')
    X_side = np.memmap(os.path.join(outputdir,'X_side'),
                       dtype=float,
                       shape=shp['sidX'],
                       mode='w+')
    Y_side = np.memmap(os.path.join(outputdir,'Y_side'),
                       dtype=float,
                       shape=shp['sidY'],
                       mode='w+')
    X_inner = np.memmap(os.path.join(outputdir,'X_inner'),
                        dtype=float,
                        shape=shp['innX'],
                        mode='w+')
    Y_inner = np.memmap(os.path.join(outputdir,'Y_inner'),
                        dtype=float,
                        shape=shp['innY'],
                        mode='w+')

    # print '\nReading from:', permdir
    # print 'Size of perm field: %dx%dx%dx%d' %(nz,ny,nx,3)
    # print 'Generating patches by coarsening into: %dx%d' %(Ny,Nx)

    # fire off workers
    Parallel(n_jobs=-1)(delayed(_task)(i, G, CG, DG, K_samples, dim,
                                       X_corner, Y_corner,
                                       X_side, Y_side,
                                       X_inner, Y_inner)
                        for i in xrange(nperm))

    # print 'Done!!!'
    # print 'Output directory:', outputdir

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
