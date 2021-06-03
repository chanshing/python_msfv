from __future__ import division
import numpy as np
import scipy.sparse as spa
from .tpfa import tpfa


def _corr_solve1D(G,DG,K,q_fg):
    K = np.reshape(K,(G['nz']*G['ny']*G['nx'],3))
    N_edge = len(DG['edge_cells'])
    corr1D = [list() for _ in range(N_edge)]

    for edge in range(N_edge):
        cells=DG['edge_cells'][edge]
        nodes=DG['connectivity']['edge_nodes'][edge]
        centers=[DG['nodes'][node] for node in nodes]
        Grid = dict()
        Grid['nx']=np.size(cells); Grid['hx']=1
        Grid['ny']=1; Grid['hy']=1
        Grid['nz']=1; Grid['hz']=1
        Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
        Grid['K'][0,0,:,:]=K[cells,:]
        idxs = [i for i,j in enumerate(cells) if j in centers]
        vals = np.zeros(len(idxs))
        dirichlet = np.column_stack([idxs, vals])
        q=q_fg[cells]
        P = tpfa(Grid,Grid['K'],q,dirichlet)
        corr1D[edge]=P

    return corr1D


def _corr_solve2D(G,CG,DG,K,q_fg,corr1D):
    K = np.reshape(K,(G['nz']*G['ny']*G['nx'],3))
    N_dg = len(DG['cells'])
    corr2D = [list() for _ in range(N_dg)]
    L=G['nx']; l=CG['nx']
    lp=int(L/l)

    Grid=dict()
    Grid['nx']=lp+1; Grid['hx']=1
    Grid['ny']=lp+1; Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))

    for j in range(1,l):
        for i in range(1,l):
            cell=j*(l+1)+i
            cells=DG['cells'][cell]
            edges=DG['connectivity']['cell_edges'][cell]
            Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
            idxs=[]; vals=[]
            for edge in edges:
                corr = corr1D[edge].ravel()
                idxs.extend([ii for ii,jj in
                             enumerate(cells) if
                             jj in DG['edge_cells'][edge]])
                vals.extend(corr)

            # Dirichlet conditions
            dirichlet = np.column_stack([idxs,vals])
            q=q_fg[cells]
            P = tpfa(Grid, Grid['K'], q, dirichlet)
            corr2D[cell]=P

    # Bottom half cells
    Grid=dict()
    Grid['nx']=lp+1; Grid['hx']=1
    Grid['ny']=int((lp+1)/2); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    j = 0
    for i in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        idxs=[]; vals=[]
        for edge in edges:
            corr = corr1D[edge].ravel()
            idxs.extend([ii for ii,jj in
                         enumerate(cells) if
                         jj in DG['edge_cells'][edge]])
            vals.extend(corr)

        # Dirichlet conditions
        dirichlet = np.column_stack([idxs,vals])
        q=q_fg[cells]
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        corr2D[cell]=P

    # Top half cells
    Grid=dict()
    Grid['nx']=int(lp+1); Grid['hx']=1
    Grid['ny']=int((lp+1)/2); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    j = l
    for i in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        idxs=[]; vals=[]
        for edge in edges:
            corr = corr1D[edge].ravel()
            idxs.extend([ii for ii,jj in
                         enumerate(cells) if
                         jj in DG['edge_cells'][edge]])
            vals.extend(corr)

        # Dirichlet conditions
        dirichlet = np.column_stack([idxs,vals])
        q=q_fg[cells]
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        corr2D[cell]=P

    # Left half cells
    Grid=dict()
    Grid['nx']=int((lp+1)/2); Grid['hx']=1
    Grid['ny']=int(lp+1); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    i = 0
    for j in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        idxs=[]; vals=[]
        for edge in edges:
            corr = corr1D[edge].ravel()
            idxs.extend([ii for ii,jj in
                         enumerate(cells) if
                         jj in DG['edge_cells'][edge]])
            vals.extend(corr)

        # Dirichlet conditions
        dirichlet = np.column_stack([idxs,vals])
        q=q_fg[cells]
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        corr2D[cell]=P

    # Right half cells
    Grid=dict()
    Grid['nx']=int((lp+1)/2); Grid['hx']=1
    Grid['ny']=int(lp+1); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    i = l
    for j in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        idxs=[]; vals=[]
        for edge in edges:
            corr = corr1D[edge].ravel()
            idxs.extend([ii for ii,jj in
                         enumerate(cells) if
                         jj in DG['edge_cells'][edge]])
            vals.extend(corr)

        # Dirichlet conditions
        dirichlet = np.column_stack([idxs,vals])
        q=q_fg[cells]
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        corr2D[cell]=P

    # Corner quarter cells
    Grid=dict()
    Grid['nx']=int((lp+1)/2); Grid['hx']=1
    Grid['ny']=int((lp+1)/2); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1/Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    i=0; j=0
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    idxs=[]; vals=[]
    for edge in edges:
        corr = corr1D[edge].ravel()
        idxs.extend([ii for ii,jj in
                     enumerate(cells) if
                     jj in DG['edge_cells'][edge]])
        vals.extend(corr)

    # Dirichlet conditions
    dirichlet = np.column_stack([idxs,vals])
    q=q_fg[cells]
    P = tpfa(Grid, Grid['K'], q, dirichlet)
    corr2D[cell]=P

    i=l; j=0
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    idxs=[]; vals=[]
    for edge in edges:
        corr = corr1D[edge].ravel()
        idxs.extend([ii for ii,jj in
                     enumerate(cells) if
                     jj in DG['edge_cells'][edge]])
        vals.extend(corr)

    # Dirichlet conditions
    dirichlet = np.column_stack([idxs,vals])
    q=q_fg[cells]
    P = tpfa(Grid, Grid['K'], q, dirichlet)
    corr2D[cell]=P

    i=0; j=l
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    idxs=[]; vals=[]
    for edge in edges:
        corr = corr1D[edge].ravel()
        idxs.extend([ii for ii,jj in
                     enumerate(cells) if
                     jj in DG['edge_cells'][edge]])
        vals.extend(corr)

    # Dirichlet conditions
    dirichlet = np.column_stack([idxs,vals])
    q=q_fg[cells]
    P = tpfa(Grid, Grid['K'], q, dirichlet)
    corr2D[cell]=P

    i=l; j=l
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    idxs=[]; vals=[]
    for edge in edges:
        corr = corr1D[edge].ravel()
        idxs.extend([ii for ii,jj in
                     enumerate(cells) if
                     jj in DG['edge_cells'][edge]])
        vals.extend(corr)

    # Dirichlet conditions
    dirichlet = np.column_stack([idxs,vals])
    q=q_fg[cells]
    P = tpfa(Grid, Grid['K'], q, dirichlet)
    corr2D[cell]=P

    return corr2D


def _patch_corrs(G,DG,corr2D):
    N_dg = len(DG['cells'])
    N = G['nz']*G['ny']*G['nx']
    corr = spa.lil_matrix((N,1))

    for i in range(N_dg):
        cells = DG['cells'][i]
        pc = corr2D[i].ravel()
        corr[cells] = np.reshape(pc,(len(pc),1))

    # Convert to csr for performance
    corr = corr.tocsr()

    return corr


def compute_corr(G,CG,DG,K,q,verbose=False):
    # if verbose:
    #     print '-Computing correction functions...'
    corr1D = _corr_solve1D(G,DG,K,q)
    corr2D = _corr_solve2D(G,CG,DG,K,q,corr1D)

    # if verbose:
    #     print '-Patching correction functions...'
    return _patch_corrs(G,DG,corr2D)
