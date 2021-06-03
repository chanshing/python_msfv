import numpy as np
import scipy.sparse as spa
from .geometry import csi
from .import utils as ut


def predict_bases(G,CG,DG,K,model,verbose='False'):
    """ Prediction using model passed as argument """
    Ks = ut.get_mini_Ks(K, DG)
    corners, sides, inners = csi(CG)
    N_cg = CG['ny']*CG['nx']
    bases = [0]*N_cg

    # if verbose:
    #     print 'Predicting bases...'

    # In the following, each basis (sides and corners) has to be
    # rotated accordingly to match the trained model input. After
    # prediction, it has to be rotated back before using it in MSFV.
    for i in xrange(N_cg):
        if i in inners:
            bases[i] = model['inner'].predict(Ks[i][...,:2].reshape(1,-1))
        elif i in sides['bottom']:
            bases[i] = model['side'].predict(Ks[i][...,:2].reshape(1,-1))
        elif i in sides['right']:
            KK = ut.rot90(Ks[i][...,:2],1)
            shp = np.squeeze(KK[...,0]).shape
            b = model['side'].predict(KK.reshape(1,-1))
            bases[i]= np.rot90(b.reshape(shp),-1).ravel()
        elif i in sides['top']:
            KK = ut.rot90(Ks[i][...,:2],2)
            shp = np.squeeze(KK[...,0]).shape
            b = model['side'].predict(KK.reshape(1,-1))
            bases[i]= np.rot90(b.reshape(shp),-2).ravel()
        elif i in sides['left']:
            KK = ut.rot90(Ks[i][...,:2],3)
            shp = np.squeeze(KK[...,0]).shape
            b = model['side'].predict(KK.reshape(1,-1))
            bases[i]= np.rot90(b.reshape(shp),-3).ravel()
        elif i == corners['bottom_left']:
            bases[i] = model['corner'].predict(Ks[i][...,:2].reshape(1,-1))
        elif i == corners['bottom_right']:
            KK = ut.rot90(Ks[i][...,:2],1)
            shp = np.squeeze(KK[...,0]).shape
            b = model['corner'].predict(KK.reshape(1,-1))
            bases[i] = np.rot90(b.reshape(shp),-1).ravel()
        elif i == corners['top_right']:
            KK = ut.rot90(Ks[i][...,:2],2)
            shp = np.squeeze(KK[...,0]).shape
            b = model['corner'].predict(KK.reshape(1,-1))
            bases[i] = np.rot90(b.reshape(shp),-2).ravel()
        elif i == corners['top_left']:
            KK = ut.rot90(Ks[i][...,:2],3)
            shp = np.squeeze(KK[...,0]).shape
            b = model['corner'].predict(KK.reshape(1,-1))
            bases[i] = np.rot90(b.reshape(shp),-3).ravel()

    # if verbose:
    #     print 'Expanding bases...'

    return expand_bases(G,CG,DG,bases)


def expand_bases(G,CG,DG,bases):
    N = G['nz']*G['ny']*G['nx']
    N_cg = CG['nz']*CG['ny']*CG['nx']
    expanded_bases = [[] for i in range(N_cg)]
    bases_geo = DG['bases_geo']

    for k in range(N_cg):
        cells = bases_geo[k]['cells'].ravel()
        ij = (cells,np.zeros_like(cells))
        expanded_bases[k] = spa.csr_matrix((bases[k].ravel(),ij),
                                           shape=(N,1))

    return expanded_bases
