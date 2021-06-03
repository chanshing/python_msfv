from __future__ import division
import numpy as np
from genA import genA
from relPerm import relPerm
from scipy.sparse import spdiags


def upstream(Grid, S, Fluid, V, q, T):  # Return S

    nx=Grid['nx']; ny=Grid['ny']; nz=Grid['nz']; N=nx*ny*nz;

    # Volume*Porosity
    pv=Grid['V']*np.reshape(Grid['por'], N)

    fi=np.maximum(q,0)
    XP=np.maximum(V['x'],0); XN=np.minimum(V['x'],0);
    YP=np.maximum(V['y'],0); YN=np.minimum(V['y'],0);
    ZP=np.maximum(V['z'],0); ZN=np.minimum(V['z'],0);

    # Total flux into each block
    Vi=XP[:,:,0:nx]+YP[:,0:ny,:]+ZP[0:nz,:,:]\
        -XN[:,:,1:nx+1]-YN[:,1:ny+1,:]-ZN[1:nz+1,:,:]
    Vi = np.reshape(Vi, N)

    pm = np.min(pv/(Vi+fi))
    cfl = ((1.0-Fluid['swc']-Fluid['sor'])/3)*pm
    Nts = np.ceil(T/cfl)
    dtx = (T/Nts)/pv

    A = genA(Grid,V,q)
    # Compute A*dt/|Omega_i|
    A = spdiags(dtx,0,N,N).dot(A)
    # Compute Q_in*dt/|Omega_i|
    fi = fi*dtx

    for t in np.arange(0,Nts):
        mw, mo, dmw, dmo = relPerm(S,Fluid)
        fw = mw/(mw+mo)
        S = S+(A.dot(fw)+fi)

    return S
