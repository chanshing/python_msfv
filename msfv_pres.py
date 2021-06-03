from __future__ import division
import numpy as np
from .relPerm import relPerm
from .msfv_solver import solveMSFV


def pres(G,CG,DG,K,S,Fluid,q, dirichlet=None, model=None):

    # Compute K*M(S)
    Mw,Mo,dMw,dMo = relPerm(S,Fluid)
    Mt = Mw + Mo
    KM = np.reshape(np.array([Mt,Mt,Mt]).T,
                    (G['nz'],G['ny'],G['nx'],3))*K

    # Compute and return pressure and fluxes
    solMSFV = solveMSFV(G, CG, DG, KM, q, dirichlet=dirichlet,
                        model=model, flux=True, ret_all=True)
    return solMSFV['pressure'], solMSFV['flux']
