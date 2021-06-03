# from __future__ import division
from tempfile import mkdtemp
from joblib import Memory
import numpy as np
import scipy.sparse as sp

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir)


# @memory.cache
def compute_lagrange_bases(G,CG,DG):
    N=G['nz']*G['ny']*G['nx']
    N_cg=CG['nz']*CG['ny']*CG['nx']
    L=G['nx']; l=CG['nx']
    l_dc=L/l+1
    l_dc_half=l_dc/2
    l_basis=2*L/l+1

    l_full = l_basis
    l_three_quarter = l_dc+l_dc_half-1

    full_side=[0]*l_full
    three_quarter_side=[0]*l_three_quarter

    full_side[0:l_dc]=np.arange(0,l_dc).astype('float')
    full_side[l_dc-1:]=np.arange(l_dc-1,-1,-1).astype('float')

    three_quarter_side[0:l_dc_half]=\
        np.array([l_dc-1]*l_dc_half).astype('float')
    three_quarter_side[l_dc_half-1:]=\
        np.arange(l_dc-1,-1,-1).astype('float')

    w = float(np.square(l_dc-1))

    B_inner=np.tile(full_side,(l_full,1))*\
        np.tile(np.reshape(full_side,(l_full,1)),(1,l_full))/w
    B_boundary=np.tile(full_side,(l_three_quarter,1)) * np.tile(
            np.reshape(three_quarter_side,(l_three_quarter,1)),(1,l_full))/w
    B_corner= np.tile(three_quarter_side,(l_three_quarter,1)) *\
        np.tile(
            np.reshape(three_quarter_side,(l_three_quarter,1)),
            (1,l_three_quarter))/w

    bases = [[] for i in range(N_cg)]

    # Full basis functions
    for (j,i) in [(jj,ii) for jj in range(1,l-1) for ii in range(1,l-1)]:
        k=j*l+i
        dy,dx=DG['bases_geo'][k]['cells_idxs']
        basis=sp.lil_matrix((G['ny'],G['nx']))
        basis[dy,dx]=B_inner
        bases[k]=basis.reshape((N,1)).tocsr()

    # Half basis functions (Bottom)
    j=0
    for i in range(1,l-1):
        k=j*l+i
        dy,dx=DG['bases_geo'][k]['cells_idxs']
        basis=sp.lil_matrix((G['ny'],G['nx']))
        basis[dy,dx]=B_boundary
        bases[k]=basis.reshape((N,1)).tocsr()

    # Half basis functions (Right)
    B_boundary=np.rot90(B_boundary,-1)
    i=l-1
    for j in range(1,l-1):
        k=j*l+i
        dy,dx=DG['bases_geo'][k]['cells_idxs']
        basis=sp.lil_matrix((G['ny'],G['nx']))
        basis[dy,dx]=B_boundary
        bases[k]=basis.reshape((N,1)).tocsr()

    # Half basis functions (Top)
    B_boundary=np.rot90(B_boundary,-1)
    j=l-1
    for i in range(1,l-1):
        k=j*l+i
        dy,dx=DG['bases_geo'][k]['cells_idxs']
        basis=sp.lil_matrix((G['ny'],G['nx']))
        basis[dy,dx]=B_boundary
        bases[k]=basis.reshape((N,1)).tocsr()

    # Half basis functions (Left)
    B_boundary=np.rot90(B_boundary,-1)
    i=0
    for j in range(1,l-1):
        k=j*l+i
        dy,dx=DG['bases_geo'][k]['cells_idxs']
        basis=sp.lil_matrix((G['ny'],G['nx']))
        basis[dy,dx]=B_boundary
        bases[k]=basis.reshape((N,1)).tocsr()

    # Corner (Bottom Left)
    i=0; j=0
    k=j*l+i
    dy,dx=DG['bases_geo'][k]['cells_idxs']
    basis=sp.lil_matrix((G['ny'],G['nx']))
    basis[dy,dx]=B_corner
    bases[k]=basis.reshape((N,1)).tocsr()

    # Corner (Bottom Right)
    i=l-1; j=0
    k=j*l+i
    dy,dx=DG['bases_geo'][k]['cells_idxs']
    basis=sp.lil_matrix((G['ny'],G['nx']))
    basis[dy,dx]=np.rot90(B_corner,-1)
    bases[k]=basis.reshape((N,1)).tocsr()

    # Corner (Top Right)
    i=l-1; j=l-1
    k=j*l+i
    dy,dx=DG['bases_geo'][k]['cells_idxs']
    basis=sp.lil_matrix((G['ny'],G['nx']))
    basis[dy,dx]=np.rot90(B_corner,-2)
    bases[k]=basis.reshape((N,1)).tocsr()

    # Corner (Top Left)
    i=0; j=l-1
    k=j*l+i
    dy,dx=DG['bases_geo'][k]['cells_idxs']
    basis=sp.lil_matrix((G['ny'],G['nx']))
    basis[dy,dx]=np.rot90(B_corner,-3)
    bases[k]=basis.reshape((N,1)).tocsr()

    return bases
