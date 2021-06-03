from __future__ import division
import numpy as np


def relperm(s, Fluid):  # Return Mw, Mo, dMw, dMo

    # S = (s-Fluid['swc'])/(1.0-Fluid['swc']-Fluid['sor'])
    # Mw = np.square(S)/Fluid['vw']
    # Mo = np.square(1.0-S)/Fluid['vo']
    # dMw = 2.0*S/Fluid['vw']/(1.0-Fluid['swc']-Fluid['sor'])
    # dMo = -2.0*(1.0-S)/Fluid['vo']/(1.0-Fluid['swc']-Fluid['sor'])

    S = (s-Fluid['swc'])/(1.0-Fluid['swc']-Fluid['sor'])
    Mw = S/Fluid['vw']
    Mo = 1.0-S/Fluid['vo']
    dMw = 1.0/(Fluid['vw']*(1.0-Fluid['swc']-Fluid['sor']))
    dMo = -1.0/(Fluid['vo']*(1.0-Fluid['swc']-Fluid['sor']))

    return Mw, Mo, dMw, dMo
