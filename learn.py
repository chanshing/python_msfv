import os
import numpy as np
import scipy.sparse as spa
from geometry import csi
from joblib import dump
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import IncrementalPCA
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.utils as skut
import utils as ut

__all__ = ['train', 'predict']


def train(data, modelargs,
          N_trains={'inner': None, 'side': None, 'corner': None},
          ret=False):
    """
    data = {'X_inner': X_inner, 'Y_inner': Y_inner,
            'X_side': X_side, 'Y_side': Y_side,
            'X_corner': X_corner, 'Y_corner': Y_corner}
    modelparams = {'name': name, 'gsargs': gsargs}
    """

    X_inner, Y_inner = skut.shuffle(data['X_inner'], data['Y_inner'],
                                    n_samples=N_trains['inner'])
    X_side, Y_side = skut.shuffle(data['X_side'], data['Y_side'],
                                  n_samples=N_trains['side'])
    X_corner, Y_corner = skut.shuffle(data['X_corner'], data['Y_corner'],
                                      n_samples=N_trains['corner'])

    modelname = modelargs['name']
    demography = modelargs['demography']
    pipe, params = modelargs['gsargs']

    modelfolder = modelname+'_'+\
        'inn'+str(X_inner.shape[0])+'_'+\
        'sid'+str(X_side.shape[0])+'_'+\
        'cor'+str(X_corner.shape[0])

    odir = os.path.join('models', demography, modelfolder)
    ut.mkdir_p(odir)
    print 'Training model:', modelname
    print 'Model will be stored at', os.path.join(odir, 'model.pkl')

    # train corners
    print 'Training corners with %d samples' %(X_corner.shape[0])
    corner = GridSearchCV(pipe,
                          params,
                          cv=5,
                          n_jobs=-1,
                          pre_dispatch='2*n_jobs').fit(X_corner,
                                                       Y_corner)
    print 'Best params for corners:', corner.best_params_
    print 'Best score:', corner.best_score_

    # train sides
    print 'Training sides with %d samples' %(X_side.shape[0])
    side = GridSearchCV(pipe,
                        params,
                        cv=5,
                        n_jobs=-1,
                        pre_dispatch='2*n_jobs').fit(X_side,
                                                     Y_side)
    print 'Best params for sides:', side.best_params_
    print 'Best score:', side.best_score_

    # train inners
    print 'Training inners with %d samples' %(X_inner.shape[0])
    inner = GridSearchCV(pipe,
                         params,
                         cv=5,
                         n_jobs=-1,
                         pre_dispatch='2*n_jobs').fit(X_inner,
                                                      Y_inner)
    print 'Best params for inners:', inner.best_params_
    print 'Best score:', inner.best_score_

    model = {'inner':inner, 'side':side, 'corner':corner}
    dump(model, os.path.join(odir, 'model.pkl'))
    print 'Finished and saved at', os.path.join(odir, 'model.pkl')

    if ret:
        return model


def predict(G, CG, DG, K, model, dim=1, verbose='False'):
    """ Prediction using model passed as argument """
    K = np.log(K)  # use log-perm for prediction
    Ks = ut.get_mini_Ks(K, DG)
    corners, sides, inners = csi(CG)
    N_cg = CG['ny']*CG['nx']
    bases = [0]*N_cg

    if verbose:
        print 'Predicting bases...'

    # In the following, each basis (sides and corners) has to be
    # rotated accordingly to match the trained model input. After
    # prediction, it has to be rotated back before usage in MSFV.
    for i in xrange(N_cg):
        if i in inners:
            bases[i] = model['inner'].predict(Ks[i][...,:dim].reshape(1,-1))
        elif i in sides['bottom']:
            bases[i] = model['side'].predict(Ks[i][...,:dim].reshape(1,-1))
        elif i in sides['right']:
            KK = ut.rot90(Ks[i][...,:dim],1)
            shp = np.squeeze(KK[...,0]).shape
            b = model['side'].predict(KK.reshape(1,-1))
            bases[i]= np.rot90(b.reshape(shp),-1).ravel()
        elif i in sides['top']:
            KK = ut.rot90(Ks[i][...,:dim],2)
            shp = np.squeeze(KK[...,0]).shape
            b = model['side'].predict(KK.reshape(1,-1))
            bases[i]= np.rot90(b.reshape(shp),-2).ravel()
        elif i in sides['left']:
            KK = ut.rot90(Ks[i][...,:dim],3)
            shp = np.squeeze(KK[...,0]).shape
            b = model['side'].predict(KK.reshape(1,-1))
            bases[i]= np.rot90(b.reshape(shp),-3).ravel()
        elif i == corners['bottom_left']:
            bases[i] = model['corner'].predict(Ks[i][...,:dim].reshape(1,-1))
        elif i == corners['bottom_right']:
            KK = ut.rot90(Ks[i][...,:dim],1)
            shp = np.squeeze(KK[...,0]).shape
            b = model['corner'].predict(KK.reshape(1,-1))
            bases[i] = np.rot90(b.reshape(shp),-1).ravel()
        elif i == corners['top_right']:
            KK = ut.rot90(Ks[i][...,:dim],2)
            shp = np.squeeze(KK[...,0]).shape
            b = model['corner'].predict(KK.reshape(1,-1))
            bases[i] = np.rot90(b.reshape(shp),-2).ravel()
        elif i == corners['top_left']:
            KK = ut.rot90(Ks[i][...,:dim],3)
            shp = np.squeeze(KK[...,0]).shape
            b = model['corner'].predict(KK.reshape(1,-1))
            bases[i] = np.rot90(b.reshape(shp),-3).ravel()

    if verbose:
        print 'Expanding bases...'

    return _expand_bases(G,CG,DG,bases)


def _expand_bases(G,CG,DG,bases):
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
