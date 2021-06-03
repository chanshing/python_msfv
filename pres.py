from __future__ import division
import numpy as np
from .tpfa import tpfa
# from .relPerm import relPerm
from .msfv_solver import solveMSFV


def pres(G, K, S, Fluid, q, relperm, dirichlet=None,
         msfv=False, CG=None, DG=None, model=None):

    # Compute K*M(S)
    Mw, Mo, dMw, dMo = relperm(S, Fluid)
    Mt = Mw + Mo
    KM = np.reshape(np.array([Mt,Mt,Mt]).T,
                    (G['nz'],G['ny'],G['nx'],3))*K
    # Compute and return pressure and fluxes
    if msfv:
        assert CG is not None
        assert DG is not None
        solMSFV = solveMSFV(G, CG, DG, KM, q, dirichlet=dirichlet,
                            model=model, flux=True, ret_all=True)
        return solMSFV['pressure'], solMSFV['flux']
    else:
        return tpfa(G, KM, q, dirichlet=dirichlet, flux=True)
