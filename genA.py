from __future__ import division
import numpy as np
from scipy.sparse import spdiags


def genA(Grid, V, q):  # Return A

    nx=Grid['nx']; ny=Grid['ny']; nz=Grid['nz']; N=nx*ny*nz;
    fp = np.minimum(q,0)

    XN=np.minimum(V['x'],0); x1=np.reshape(XN[:,:,0:nx], N)
    YN=np.minimum(V['y'],0); y1=np.reshape(YN[:,0:ny,:], N)
    ZN=np.minimum(V['z'],0); z1=np.reshape(ZN[0:nz,:,:], N)
    XP=np.maximum(V['x'],0); x2=np.reshape(XP[:,:,1:nx+1], N)
    YP=np.maximum(V['y'],0); y2=np.reshape(YP[:,1:ny+1,:], N)
    ZP=np.maximum(V['z'],0); z2=np.reshape(ZP[1:nz+1,:,:], N)

    DiagVecs=[z2,y2,x2,fp+x1-x2+y1-y2+z1-z2,-x1,-y1,-z1]
    DiagIndx=[-nx*ny,-nx,-1,0,1,nx,nx*ny]
    A=spdiags(DiagVecs,DiagIndx,N,N, format='csr')

    return A
