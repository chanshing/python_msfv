import os
import errno
import numpy as np
import re
from . import geometry as ge
# from tempfile import mkdtemp
# from joblib import Memory

# cachedir = mkdtemp()
# memory = Memory(cachedir=None, verbose=False)


# @memory.cache
def compute_trans(G, K):
    nx = G['nx']; hx = G['hx']
    ny = G['ny']; hy = G['hy']
    nz = G['nz']; hz = G['hz']
    L = 1.0/K

    tx = 2*hy*hz/hx; TX = np.zeros((nz,ny,nx+1))
    ty = 2*hx*hz/hy; TY = np.zeros((nz,ny+1,nx))
    tz = 2*hx*hy/hz; TZ = np.zeros((nz+1,ny,nx))
    TX[:,:,1:nx] = tx/(L[:,:,0:nx-1,0]+L[:,:,1:nx,0])
    TY[:,1:ny,:] = ty/(L[:,0:ny-1,:,1]+L[:,1:ny,:,1])
    TZ[1:nz,:,:] = tz/(L[0:nz-1,:,:,2]+L[1:nz,:,:,2])

    return TX,TY,TZ


def interpolate(P_cg, G, CG, bases, corr):
    N_cg = len(CG['cells'])
    P = np.zeros((G['nz']*G['ny']*G['nx'],1))

    for k in range(N_cg):
        P = P + P_cg[k]*bases[k]

    P = P + corr

    return np.array(P)


def crop_bases(CG, DG, bases):
    """ Crop the set of bases """
    N_cg = CG['nz']*CG['ny']*CG['nx']
    geo = DG['bases_geo']
    cropped_bases = np.zeros_like(bases)

    for k in range(N_cg):
        basis = bases[k]
        shape = np.shape(geo[k]['cells'])
        cells = geo[k]['cells'].ravel()
        cropped_bases[k] = np.reshape(basis[cells].toarray(), shape)

    return cropped_bases


def get_mini_Ks(K, DG):
    """ Generate the permeability patches """
    Ks = []
    for geo in DG['bases_geo']:
        dy,dx = geo['cells_idxs']
        Ks.append(K[:,dy,dx,:])

    return Ks


def mkdir_p(path):
    """ Create a directory if doesn't exist """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rot90(x, n):
    """ n times rotation ignoring last axis """
    rot_x = []

    for i in range(x.shape[-1]):
        rot_x.append(np.rot90(np.squeeze(x[...,i], axis=0), n))

    rot_x = np.array(rot_x)
    return np.rollaxis(rot_x, 0, rot_x.ndim)[None,...]


def pu_scale(bases, DG):
    """
    Scale the bases in order to ensure partition of unity.
    We assume a_i' = a_i*(1 + c).
    Note: 'bases' is modified in place.
    """

    bases_sum = sum(bases).toarray()
    N_cg = len(bases)

    for k in range(N_cg):
        cells = DG['bases_geo'][k]['cells'].ravel()
        bases[k][cells] = bases[k][cells]/bases_sum[cells]


def pu_correct(bases, p, DG):
    """
    Add a correction term to ensure partition of unity.
    We assume a_i' = a_i + c*a_i^p.
    When p = 1 this is the same as pu_scale(),
    i.e. linear scaling a_i' = a_i*(1 + c).
    Note: 'bases' is modified in place.
    """

    bases_sum = sum(bases).toarray()
    bases_psum = np.zeros_like(bases_sum)
    N_cg = len(bases)

    # compute bases_psum
    for k in range(N_cg):
        cells = DG['bases_geo'][k]['inner_cells'].ravel()
        bases_psum[cells] += bases[k][cells].toarray()**p

    # correct bases
    for k in range(N_cg):
        cells = DG['bases_geo'][k]['inner_cells'].ravel()
        bases[k][cells] += \
            bases[k][cells].toarray()**p*(1-bases_sum[cells])/bases_psum[cells]


def parse_perm(dirname):
    """
    Extract arguments from perm dirname.
    Returns dictionary with extracted arguments.
    """
    [nz, ny, nx, length, sigma, nperm] = re.findall(r'\d+\.\d+|\d+',
                                                    dirname)

    args = {}
    args['nz'] = int(nz)
    args['ny'] = int(ny)
    args['nx'] = int(nx)
    args['length'] = float(length)
    args['sigma'] = float(sigma)
    args['nperm'] = int(nperm)

    return args


def unparse_perm(nz, ny, nx, length, sigma, nperm):
    """
    Create directory name based on arguments.
    """
    dirname = str(nz) + 'x' + str(ny) + 'x' + str(nx) \
        + '_' + 'L' +str(length) + '_' + 's' + str(sigma) \
        + '_' + 'n' + str(nperm)

    return dirname


def parse_dataset(dirname):
    """
    Extract arguments from dataset dirname.
    Returns dictionary with extracted arguments.
    """
    [Ny, Nx, nz, ny, nx, length, sigma, nperm] = \
        re.findall(r'\d+\.\d+|\d+', dirname)

    args = {}
    args['Ny'] = int(Ny)
    args['Nx'] = int(Nx)
    args['nz'] = int(nz)
    args['ny'] = int(ny)
    args['nx'] = int(nx)
    args['length'] = float(length)
    args['sigma'] = float(sigma)
    args['nperm'] = int(nperm)

    return args


def unparse_dataset(Ny, Nx, nz, ny, nx, length, sigma, nperm):
    """
    Create directory name based on arguments.
    """
    dirname = str(Nx) + 'x' + str(Nx) + '_' + \
        str(nz) + 'x' + str(ny) + 'x' + str(nx) + '_' + \
        'L' +str(length) + '_' + 's' + str(sigma) + '_' + \
        'n' + str(nperm)

    return dirname


def read_dataset(datadir, dim=1, ravel=True):
    args = parse_dataset(datadir)
    # read data as (n_samples, my, mx, dim)
    shp = get_mm_shapes({'ny':args['ny'], 'nx':args['nx']},
                        {'ny':args['Ny'], 'nx':args['Nx']},
                        args['nperm'], dim,
                        per_perm_sample=False, ravel=ravel)

    X_corner = np.memmap(os.path.join(datadir,'X_corner'),
                         dtype=float,
                         shape=shp['corX'],
                         mode='r')
    Y_corner = np.memmap(os.path.join(datadir,'Y_corner'),
                         dtype=float,
                         shape=shp['corY'],
                         mode='r')
    X_side = np.memmap(os.path.join(datadir,'X_side'),
                       dtype=float,
                       shape=shp['sidX'],
                       mode='r')
    Y_side = np.memmap(os.path.join(datadir,'Y_side'),
                       dtype=float,
                       shape=shp['sidY'],
                       mode='r')
    X_inner = np.memmap(os.path.join(datadir,'X_inner'),
                        dtype=float,
                        shape=shp['innX'],
                        mode='r')
    Y_inner = np.memmap(os.path.join(datadir,'Y_inner'),
                        dtype=float,
                        shape=shp['innY'],
                        mode='r')

    return {'X_inner': X_inner, 'Y_inner': Y_inner,
            'X_side': X_side, 'Y_side': Y_side,
            'X_corner': X_corner, 'Y_corner': Y_corner}


def get_mm_shapes(G, CG, nperm, dim=1, per_perm_sample=True,
                  ravel=False):

    inn_shape, sid_shape, cor_shape = ge.get_basis_shapes(G, CG)
    n_inn, n_sid, n_cor = ge.get_basis_amounts(CG)

    if per_perm_sample:
        mm_innX = (nperm, n_inn)
        mm_sidX = (nperm, n_sid)
        mm_corX = (nperm, n_cor)
        mm_innY = (nperm, n_inn)
        mm_sidY = (nperm, n_sid)
        mm_corY = (nperm, n_cor)
    else:
        mm_innX = (nperm*n_inn,)
        mm_sidX = (nperm*n_sid,)
        mm_corX = (nperm*n_cor,)
        mm_innY = (nperm*n_inn,)
        mm_sidY = (nperm*n_sid,)
        mm_corY = (nperm*n_cor,)

    if ravel:
        mm_innX += (inn_shape[0]*inn_shape[1]*dim,)
        mm_sidX += (sid_shape[0]*sid_shape[1]*dim,)
        mm_corX += (cor_shape[0]*cor_shape[1]*dim,)
        mm_innY += (inn_shape[0]*inn_shape[1],)
        mm_sidY += (sid_shape[0]*sid_shape[1],)
        mm_corY += (cor_shape[0]*cor_shape[1],)
    else:
        mm_innX += inn_shape + (dim,)
        mm_sidX += sid_shape + (dim,)
        mm_corX += cor_shape + (dim,)
        mm_innY += inn_shape
        mm_sidY += sid_shape
        mm_corY += cor_shape

    return {'innX':mm_innX, 'innY':mm_innY,
            'sidX':mm_sidX, 'sidY':mm_sidY,
            'corX':mm_corX, 'corY':mm_corY}
