from __future__ import division
import numpy as np
import scipy.sparse as spa
from .tpfa import tpfa


def _solve1D(G,DG,K):
    # Solve the lower dimensional problem on the edges

    K = np.reshape(K,(G['nz']*G['ny']*G['nx'],3))
    N_edge = len(DG['edge_cells'])
    bases1D = [list() for _ in range(N_edge)]

    for edge in range(N_edge):
        cells=DG['edge_cells'][edge]
        nodes=DG['connectivity']['edge_nodes'][edge]
        centers=[DG['nodes'][node] for node in nodes]
        Grid = dict()
        Grid['nx']=np.size(cells); Grid['hx']=1
        Grid['ny']=1; Grid['hy']=1
        Grid['nz']=1; Grid['hz']=1
        N=Grid['nx']*Grid['ny']*Grid['nz']
        Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
        Grid['K'][0,0,:,:]=K[cells,:]
        idxs = [i for i,j in enumerate(cells) if j in centers]

        for idx in idxs:
            vals = np.zeros(len(idxs))
            vals[idxs.index(idx)] = 1.0
            dirichlet = np.column_stack([idxs, vals])
            q=np.zeros(N)
            P = tpfa(Grid,Grid['K'],q,dirichlet)
            bases1D[edge].append((nodes[idxs.index(idx)],P.ravel()))

    return bases1D


def _solve2D(G,CG,DG,K,bases1D,timer=None):
    # Solve for the bases for each dual cell
    K = np.reshape(K,(G['nz']*G['ny']*G['nx'],3))
    N_dg = len(DG['cells'])
    bases2D = [list() for _ in range(N_dg)]
    L=G['nx']; l=CG['nx']
    lp=int(L/l)

    Grid=dict()
    Grid['nx']=lp+1; Grid['hx']=1
    Grid['ny']=lp+1; Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    N=Grid['nx']*Grid['ny']*Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))

    for j in range(1,l):
        for i in range(1,l):
            cell=j*(l+1)+i
            cells=DG['cells'][cell]
            nodes=DG['connectivity']['cell_nodes'][cell]
            edges=DG['connectivity']['cell_edges'][cell]
            Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
            for node in nodes:
                bottom = range(0,Grid['nx'],1)
                top = range(N-Grid['nx'],N,1)
                left = range(0,N-Grid['nx']+1,Grid['nx'])
                right = range(Grid['nx']-1,N,Grid['nx'])
                idxs = list(set(bottom+top+left+right))
                vals = np.zeros(len(idxs))
                basis_idxs = list()
                basis_vals = list()
                for edge in edges:
                    bases = bases1D[edge]
                    for basis in bases:
                        if node == basis[0]:
                            basis_idxs.extend([ii for ii,jj in
                                               enumerate(cells) if
                                               jj in DG['edge_cells'][edge]])
                            basis_vals.extend(basis[1])

                # Dirichlet conditions
                for idx in basis_idxs:
                    if idx in idxs:
                        val_idx = idxs.index(idx)
                        vals[val_idx] = basis_vals[basis_idxs.index(idx)]
                dirichlet = np.column_stack([idxs,vals])
                q=np.zeros(N)
                P = tpfa(Grid, Grid['K'], q, dirichlet,timer=timer)
                bases2D[cell].append((node,P.ravel()))

    # Bottom half cells
    Grid=dict()
    Grid['nx']=lp+1; Grid['hx']=1
    Grid['ny']=int((lp+1)/2); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    N=Grid['nx']*Grid['ny']*Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    j = 0
    for i in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        nodes=DG['connectivity']['cell_nodes'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        for node in nodes:
            # bottom = range(0,Grid['nx'],1)
            top = range(N-Grid['nx'],N,1)
            left = range(0,N-Grid['nx']+1,Grid['nx'])
            right = range(Grid['nx']-1,N,Grid['nx'])
            idxs = list(set(top+left+right))
            vals = np.zeros(len(idxs))
            basis_idxs = list()
            basis_vals = list()
            for edge in edges:
                bases = bases1D[edge]
                for basis in bases:
                    if node == basis[0]:
                        basis_idxs.extend([ii for ii,jj in
                                           enumerate(cells) if
                                           jj in DG['edge_cells'][edge]])
                        basis_vals.extend(basis[1])

            for idx in basis_idxs:
                if idx in idxs:
                    val_idx = idxs.index(idx)
                    vals[val_idx] = basis_vals[basis_idxs.index(idx)]

            dirichlet = np.column_stack([idxs,vals])
            q=np.zeros(N)
            P = tpfa(Grid, Grid['K'], q, dirichlet)
            bases2D[cell].append((node,P.ravel()))

    # Top half cells
    Grid=dict()
    Grid['nx']=int(lp+1); Grid['hx']=1
    Grid['ny']=int((lp+1)/2); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    N=Grid['nx']*Grid['ny']*Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    j = l
    for i in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        nodes=DG['connectivity']['cell_nodes'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        for node in nodes:
            bottom = range(0,Grid['nx'],1)
            # top = range(N-Grid['nx'],N,1)
            left = range(0,N-Grid['nx']+1,Grid['nx'])
            right = range(Grid['nx']-1,N,Grid['nx'])
            idxs = list(set(bottom+left+right))
            vals = np.zeros(len(idxs))
            basis_idxs = list()
            basis_vals = list()
            for edge in edges:
                bases = bases1D[edge]
                for basis in bases:
                    if node == basis[0]:
                        basis_idxs.extend([ii for ii,jj in
                                           enumerate(cells) if
                                           jj in DG['edge_cells'][edge]])
                        basis_vals.extend(basis[1])

            for idx in basis_idxs:
                if idx in idxs:
                    val_idx = idxs.index(idx)
                    vals[val_idx] = basis_vals[basis_idxs.index(idx)]

            dirichlet = np.column_stack([idxs,vals])
            q=np.zeros(N)
            P = tpfa(Grid, Grid['K'], q, dirichlet)
            bases2D[cell].append((node,P.ravel()))

    # Left half cells
    Grid=dict()
    Grid['nx']=int((lp+1)/2); Grid['hx']=1
    Grid['ny']=int(lp+1); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    N=Grid['nx']*Grid['ny']*Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    i = 0
    for j in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        nodes=DG['connectivity']['cell_nodes'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        for node in nodes:
            bottom = range(0,Grid['nx'],1)
            top = range(N-Grid['nx'],N,1)
            # left = range(0,N-Grid['nx']+1,Grid['nx'])
            right = range(Grid['nx']-1,N,Grid['nx'])
            idxs = list(set(bottom+top+right))
            vals = np.zeros(len(idxs))
            basis_idxs = list()
            basis_vals = list()
            for edge in edges:
                bases = bases1D[edge]
                for basis in bases:
                    if node == basis[0]:
                        basis_idxs.extend([ii for ii,jj in
                                           enumerate(cells) if
                                           jj in DG['edge_cells'][edge]])
                        basis_vals.extend(basis[1])

            for idx in basis_idxs:
                if idx in idxs:
                    val_idx = idxs.index(idx)
                    vals[val_idx] = basis_vals[basis_idxs.index(idx)]

            dirichlet = np.column_stack([idxs,vals])
            q=np.zeros(N)
            P = tpfa(Grid, Grid['K'], q, dirichlet)
            bases2D[cell].append((node,P.ravel()))

    # Right half cells
    Grid=dict()
    Grid['nx']=int((lp+1)/2); Grid['hx']=1
    Grid['ny']=int(lp+1); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1
    N=Grid['nx']*Grid['ny']*Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    i = l
    for j in range(1,l):
        cell=j*(l+1)+i
        cells=DG['cells'][cell]
        nodes=DG['connectivity']['cell_nodes'][cell]
        edges=DG['connectivity']['cell_edges'][cell]
        Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
        for node in nodes:
            bottom = range(0,Grid['nx'],1)
            top = range(N-Grid['nx'],N,1)
            left = range(0,N-Grid['nx']+1,Grid['nx'])
            # right = range(Grid['nx']-1,N,Grid['nx'])
            idxs = list(set(bottom+top+left))
            vals = np.zeros(len(idxs))
            basis_idxs = list()
            basis_vals = list()
            for edge in edges:
                bases = bases1D[edge]
                for basis in bases:
                    if node == basis[0]:
                        basis_idxs.extend([ii for ii,jj in
                                           enumerate(cells) if
                                           jj in DG['edge_cells'][edge]])
                        basis_vals.extend(basis[1])

            for idx in basis_idxs:
                if idx in idxs:
                    val_idx = idxs.index(idx)
                    vals[val_idx] = basis_vals[basis_idxs.index(idx)]

            dirichlet = np.column_stack([idxs,vals])
            q=np.zeros(N)
            P = tpfa(Grid, Grid['K'], q, dirichlet)
            bases2D[cell].append((node,P.ravel()))

    # Corner quarter cells
    Grid=dict()
    Grid['nx']=int((lp+1)/2); Grid['hx']=1
    Grid['ny']=int((lp+1)/2); Grid['hy']=1
    Grid['nz']=1; Grid['hz']=1/Grid['nz']
    N=Grid['nx']*Grid['ny']*Grid['nz']
    Grid['K']=np.zeros((Grid['nz'],Grid['ny'],Grid['nx'],3))
    i=0; j=0
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    nodes=DG['connectivity']['cell_nodes'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    for node in nodes:
        basis_idxs = list()
        basis_vals = list()
        for edge in edges:
            bases = bases1D[edge]
            for basis in bases:
                if node == basis[0]:
                    basis_idxs.extend([ii for ii,jj in
                                       enumerate(cells) if
                                       jj in DG['edge_cells'][edge]])
                    basis_vals.extend(basis[1])

        dirichlet = np.column_stack([basis_idxs,basis_vals])
        q=np.zeros(N)
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        bases2D[cell].append((node,P.ravel()))

    i=l; j=0
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    nodes=DG['connectivity']['cell_nodes'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    for node in nodes:
        basis_idxs = list()
        basis_vals = list()
        for edge in edges:
            bases = bases1D[edge]
            for basis in bases:
                if node == basis[0]:
                    basis_idxs.extend([ii for ii,jj in
                                       enumerate(cells) if
                                       jj in DG['edge_cells'][edge]])
                    basis_vals.extend(basis[1])

        q=np.zeros(N)
        dirichlet = np.column_stack([basis_idxs,basis_vals])
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        bases2D[cell].append((node,P.ravel()))

    i=0; j=l
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    nodes=DG['connectivity']['cell_nodes'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    for node in nodes:
        basis_idxs = list()
        basis_vals = list()
        for edge in edges:
            bases = bases1D[edge]
            for basis in bases:
                if node == basis[0]:
                    basis_idxs.extend([ii for ii,jj in
                                       enumerate(cells) if
                                       jj in DG['edge_cells'][edge]])
                    basis_vals.extend(basis[1])

        dirichlet = np.column_stack([basis_idxs,basis_vals])
        q=np.zeros(N)
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        bases2D[cell].append((node,P.ravel()))

    i=l; j=l
    cell=j*(l+1)+i
    cells=DG['cells'][cell]
    nodes=DG['connectivity']['cell_nodes'][cell]
    edges=DG['connectivity']['cell_edges'][cell]
    Grid['K'][0,:,:,:]=np.reshape(K[cells,:],(Grid['ny'],Grid['nx'],3))
    for node in nodes:
        basis_idxs = list()
        basis_vals = list()
        for edge in edges:
            bases = bases1D[edge]
            for basis in bases:
                if node == basis[0]:
                    basis_idxs.extend([ii for ii,jj in
                                       enumerate(cells) if
                                       jj in DG['edge_cells'][edge]])
                    basis_vals.extend(basis[1])

        dirichlet = np.column_stack([basis_idxs,basis_vals])
        q=np.zeros(N)
        P = tpfa(Grid, Grid['K'], q, dirichlet)
        bases2D[cell].append((node,P.ravel()))

    return bases2D


def _patch_bases(G,CG,DG,bases2D):
    N = G['nz']*G['ny']*G['nx']
    N_cg = CG['nz']*CG['ny']*CG['nx']
    N_dg = len(DG['cells'])

    bases = [[] for k in range(N_cg)]
    basis_in_pieces = [[] for k in range(N_cg)]
    cells_list = [[] for k in range(N_cg)]

    # gather dual cells
    for i in range(N_dg):
        for k,b in bases2D[i]:
            basis_in_pieces[k].append(b)
            cells_list[k].append(DG['cells'][i])

    # patch them
    for k in range(N_cg):
        cells = np.concatenate(cells_list[k])
        basis = np.concatenate(basis_in_pieces[k])
        # eliminate boundary duplication
        u = np.unique(cells, return_index=True)[1]
        ij = (cells[u],np.zeros_like(cells[u]))
        bases[k] = spa.csr_matrix((basis[u],ij), shape=(N,1))

    return bases


def compute_bases(G,CG,DG,K,verbose=False,timer=None):
    # if verbose:
    #     print '-Computing basis functions...'
    bases1D = _solve1D(G,DG,K)
    bases2D = _solve2D(G,CG,DG,K,bases1D,timer=timer)

    # if verbose:
    #     print '-Patching basis functions...'
    return _patch_bases(G,CG,DG,bases2D)
