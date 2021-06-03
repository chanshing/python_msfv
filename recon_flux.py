import numpy as np
from .utils import compute_trans
from .geometry import cart_grid
from .tpfa import tpfa


def recon_flux(P,G,CG,K,q):
    """
    Return: V [dict]
    V:
        x [(nz,ny,nx+1) array]
        y [(nz,ny+1,nx) array]
        z [(nz+1,ny,nx) array]
    """

    TX, TY, TZ = compute_trans(G,K)
    T = np.concatenate((TX.ravel(), TY.ravel(), TZ.ravel()))

    nz,ny,nx = G['cartDims']

    V = {}
    V['x'] = np.zeros((nz,ny,nx+1))
    V['y'] = np.zeros((nz,ny+1,nx))
    V['z'] = np.zeros((nz+1,ny,nx))
    V['x'][:,:,1:nx] = (P[:,:,0:nx-1]-P[:,:,1:nx])*TX[:,:,1:nx]
    V['y'][:,1:ny,:] = (P[:,0:ny-1,:]-P[:,1:ny,:])*TY[:,1:ny,:]
    V['z'][1:nz,:,:] = (P[0:nz-1,:,:]-P[1:nz,:,:])*TZ[1:nz,:,:]

    Ny,Nx=CG['cartDims']
    N_cg=Ny*Nx
    cny,cnx=CG['cellDims']
    P = P.ravel();
    K = np.reshape(K,(nz*ny*nx,3))

    for k in range(N_cg):
        cells = CG['cells'][k]
        G_local = cart_grid([cny,cnx])
        K_local = np.reshape(K[cells,:],(1,cny,cnx,3))
        q_local = q[cells]

        edges, border_in, border_out = CG['borders'][k]
        bc_flux = (P[border_in]-P[border_out])*T[edges]

        # map to local indices
        pivot = min(cells)
        border_in_local = [bi-pivot for bi in border_in]
        border_in_local = np.unravel_index(border_in_local,(ny,nx))
        border_in_local = np.ravel_multi_index(border_in_local,(cny,cnx))

        for i,bi in enumerate(border_in_local):
            q_local[bi] = q_local[bi] - bc_flux[i]

        _,V_local=tpfa(G_local,K_local,q_local,flux=True)

        cmin=min(cells); cmax=max(cells)
        cmin_z,cmin_y,cmin_x = np.unravel_index(cmin,(nz,ny,nx))
        cmax_z,cmax_y,cmax_x = np.unravel_index(cmax,(nz,ny,nx))
        V['x'][cmin_z:cmax_z+1,cmin_y:cmax_y+1,cmin_x+1:cmax_x+1]\
            =V_local['x'][:,:,1:-1]
        V['y'][cmin_z:cmax_z+1,cmin_y+1:cmax_y+1,cmin_x:cmax_x+1]\
            =V_local['y'][:,1:-1,:]
        V['z'][cmin_z+1:cmax_z+1,cmin_y:cmax_y+1,cmin_x:cmax_x+1]\
            =V_local['z'][1:-1,:,:]

    # restore shape
    P = np.reshape(P,(nz,ny,nx))
    K = np.reshape(K,(nz,ny,nx,3))

    return V
