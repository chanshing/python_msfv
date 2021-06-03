from __future__ import division
from .genA import genA
# from .relPerm import relPerm
from scipy.sparse import spdiags
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
import numpy as np


def newtRaph(Grid,S,Fluid,V,q,T,relperm):  # Return S

    N = Grid['nx']*Grid['ny']*Grid['nz']
    A = genA(Grid,V,q)

    conv=0; IT=0; S00=S;
    while conv==0:
        dt = T/np.power(2,IT)
        # Volume*Porosity
        pv = Grid['V']*np.reshape(Grid['por'], N)
        dtx = dt/pv
        # Compute A*dt/|Omega_i|
        B = spdiags(dtx,0,N,N).dot(A)
        # Compute Q_in*dt/|Omega_i|
        fi = np.maximum(q,0)*dtx

        I=0; tol=1e-3
        while I<np.power(2,IT):
            S0=S; dsn=1; it=0; I=I+1

            while dsn>tol and it<10:
                Mw, Mo, dMw, dMo = relperm(S,Fluid)
                df = dMw/(Mw+Mo)-Mw/np.power((Mw+Mo),2)*(dMw+dMo)
                dG = identity(N)-B.dot(spdiags(df,0,N,N))
                fw = Mw/(Mw+Mo)
                G = S-S0-(B.dot(fw)+fi)
                dS = spsolve(-dG,G)
                S = S+dS
                dsn = norm(dS)
                it = it+1

            if dsn>tol:
                I=np.power(2,IT); S=S00

        if dsn<tol:
            conv=1
        else:
            IT=IT+1

    return S
