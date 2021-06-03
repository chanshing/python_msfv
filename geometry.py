import numpy as np
import itertools


def cart_grid(cartdim,physdim=[1.0,1.0,1.0]):
    """ Create a catersian grid.
    Return: G [dict]
    G:
        cells [dict]
            num [int]
            faces [list of dict]
        faces [dict]
            num [int]
            neighbors [list of set]
        cartDims [tuple]
        physDims [tuple]
        cellSize [tuple]
        hx, hy, hz [float]
        nx, ny, nz [int]
    """

    physdim = [float(s) for s in physdim]
    cartdim = [int(i) for i in cartdim]
    assert isinstance(cartdim,(tuple,list))
    assert isinstance(physdim,(tuple,list))
    assert len(cartdim) > 1
    assert len(physdim) > 1
    if len(cartdim) == 2:
        cartdim=[1]+cartdim
    if len(physdim) == 2:
        physdim=[1.0]+physdim
    assert len(cartdim) == len(physdim)

    nz,ny,nx=cartdim
    lz,ly,lx=physdim
    hx=float(lx)/nx
    hy=float(ly)/ny
    hz=float(lz)/nz

    # Build faces and neighbors
    num_edgeX=nz*ny*(nx+1)
    num_edgeY=nz*(ny+1)*nx
    num_edgeZ=(nz+1)*ny*nx
    faces=[{} for i in range(nz*ny*nx)]
    neighbors=[set() for i in range(num_edgeX+num_edgeY+num_edgeZ)]
    y_shift=num_edgeX
    z_shift=num_edgeX+num_edgeY
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c=np.ravel_multi_index((k,j,i),(nz,ny,nx))
                e1=np.ravel_multi_index((k,j,i),(nz,ny,nx+1))
                e2=np.ravel_multi_index((k,j,i+1),(nz,ny,nx+1))
                e3=np.ravel_multi_index((k,j,i),(nz,ny+1,nx))+y_shift
                e4=np.ravel_multi_index((k,j+1,i),(nz,ny+1,nx))+y_shift
                e5=np.ravel_multi_index((k,j,i),(nz+1,ny,nx))+z_shift
                e6=np.ravel_multi_index((k+1,j,i),(nz+1,ny,nx))+z_shift
                faces[c]={'w':e1,'e':e2,'s':e3,'n':e4,'t':e5,'b':e6}
                for e in [e1,e2,e3,e4,e5,e6]:
                    neighbors[e].add(c)

    G={}
    G['nx']=nx; G['hx']=hx
    G['ny']=ny; G['hy']=hy
    G['nz']=nz; G['hz']=hz
    G['cartDims']=(nz,ny,nx)
    G['physDims']=(lz,ly,lx)
    G['cellSize']=(hz,hy,hx)
    G['cells']={}
    G['cells']['num']=nx*ny*nz
    G['cells']['faces']=faces
    G['faces']={}
    G['faces']['num']=num_edgeX+num_edgeY+num_edgeZ
    G['faces']['neighbors']=neighbors

    return G


def coarse_grid(G,cgdim):
    """Creates a coarse grid from fine grid G.
    Return: CG [dict]
    CG:
        cells [list of array]
        centers [list of int]
        borders [list of tuple of list]
        neighbors [list of list]
        cartDims [tuple]
        cellDims [tuple]
        nx, ny, nz [int]
    """

    assert isinstance(cgdim,(tuple,list))
    # only 2D square grid so far
    nz,ny,nx = G['cartDims']
    assert nz == 1
    assert ny == nx
    assert len(cgdim) == 2
    Ny,Nx = cgdim
    assert Ny == Nx
    L=nx; l=Nx
    # check L/l is odd
    assert (L/l) % 2

    N = L*L; N_cg = l*l
    grid = np.reshape(np.arange(N),(L,L))
    cells = [[] for _ in range(N_cg)]
    centers = [[] for _ in range(N_cg)]
    borders = [() for _ in range(N_cg)]
    neighbors = [[] for _ in range(N_cg)]

    lp = L/l; r = (lp-1)/2
    num_edgeX = L*(L+1)  # In general, nz*ny*(nx+1)

    # Build cells[k] and centers[k]
    for j in range(l):
        for i in range(l):
            cell = j*l+i
            dx = slice(i*lp,(i+1)*lp)
            dy = slice(j*lp,(j+1)*lp)
            cells[cell] = grid[dy,dx].ravel()
            centers[cell] = (j*lp+r)*L+i*lp+r

    # Cells with 4 borders
    for j in range(1,l-1):
        for i in range(1,l-1):
            cell = j*l+i
            c = centers[cell]
            cj,ci=np.unravel_index(c,(L,L))
            right_in = range(c+r-r*L,c+r+r*L+1,L)
            right_out = range(c+r+1-r*L,c+r+1+r*L+1,L)
            right_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                          [(y,ci+r+1) for y in range(cj-r,cj+r+1)]]
            top_in = range(c+r*L-r,c+r*L+r+1,1)
            top_out = range(c+(r+1)*L-r,c+(r+1)*L+r+1,1)
            top_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                        (jj,ii) in [(cj+r+1,x) for x in range(ci-r,ci+r+1)]]
            left_in = range(c-r-r*L,c-r+r*L+1,L)
            left_out = range(c-r-1-r*L,c-r-1+r*L+1,L)
            left_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                         [(y,ci-r) for y in range(cj-r,cj+r+1)]]
            bottom_in = range(c-r*L-r,c-r*L+r+1,1)
            bottom_out = range(c-(r+1)*L-r,c-(r+1)*L+r+1,1)
            bottom_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                           (jj,ii) in [(cj-r,x) for x in range(ci-r,ci+r+1)]]
            borders[cell] = (left_edge+right_edge+bottom_edge+top_edge,
                             left_in+right_in+bottom_in+top_in,
                             left_out+right_out+bottom_out+top_out)

    # Cells with 3 borders
    # # Bottom cells
    j = 0
    for i in range(1,l-1):
        cell = j*l+i
        c = centers[cell]
        cj,ci=np.unravel_index(c,(L,L))
        right_in = range(c+r-r*L,c+r+r*L+1,L)
        right_out = range(c+r+1-r*L,c+r+1+r*L+1,L)
        right_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                      [(y,ci+r+1) for y in range(cj-r,cj+r+1)]]
        top_in = range(c+r*L-r,c+r*L+r+1,1)
        top_out = range(c+(r+1)*L-r,c+(r+1)*L+r+1,1)
        top_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                    (jj,ii) in [(cj+r+1,x) for x in range(ci-r,ci+r+1)]]
        left_in = range(c-r-r*L,c-r+r*L+1,L)
        left_out = range(c-r-1-r*L,c-r-1+r*L+1,L)
        left_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                     [(y,ci-r) for y in range(cj-r,cj+r+1)]]
        borders[cell] = (left_edge+right_edge+top_edge,
                         left_in+right_in+top_in,
                         left_out+right_out+top_out)

    # # Top cells
    j = l-1
    for i in range(1,l-1):
        cell = j*l+i
        c = centers[cell]
        cj,ci=np.unravel_index(c,(L,L))
        right_in = range(c+r-r*L,c+r+r*L+1,L)
        right_out = range(c+r+1-r*L,c+r+1+r*L+1,L)
        right_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                      [(y,ci+r+1) for y in range(cj-r,cj+r+1)]]
        left_in = range(c-r-r*L,c-r+r*L+1,L)
        left_out = range(c-r-1-r*L,c-r-1+r*L+1,L)
        left_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                     [(y,ci-r) for y in range(cj-r,cj+r+1)]]
        bottom_in = range(c-r*L-r,c-r*L+r+1,1)
        bottom_out = range(c-(r+1)*L-r,c-(r+1)*L+r+1,1)
        bottom_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                       (jj,ii) in [(cj-r,x) for x in range(ci-r,ci+r+1)]]
        borders[cell] = (left_edge+right_edge+bottom_edge,
                         left_in+right_in+bottom_in,
                         left_out+right_out+bottom_out)

    # # Right cells
    i = l-1
    for j in range(1,l-1):
        cell = j*l+i
        c = centers[cell]
        cj,ci=np.unravel_index(c,(L,L))
        top_in = range(c+r*L-r,c+r*L+r+1,1)
        top_out = range(c+(r+1)*L-r,c+(r+1)*L+r+1,1)
        top_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                    (jj,ii) in [(cj+r+1,x) for x in range(ci-r,ci+r+1)]]
        left_in = range(c-r-r*L,c-r+r*L+1,L)
        left_out = range(c-r-1-r*L,c-r-1+r*L+1,L)
        left_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                     [(y,ci-r) for y in range(cj-r,cj+r+1)]]
        bottom_in = range(c-r*L-r,c-r*L+r+1,1)
        bottom_out = range(c-(r+1)*L-r,c-(r+1)*L+r+1,1)
        bottom_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                       (jj,ii) in [(cj-r,x) for x in range(ci-r,ci+r+1)]]
        borders[cell] = (left_edge+bottom_edge+top_edge,
                         left_in+bottom_in+top_in,
                         left_out+bottom_out+top_out)

    # # Left cells
    i = 0
    for j in range(1,l-1):
        cell = j*l+i
        c = centers[cell]
        cj,ci=np.unravel_index(c,(L,L))
        top_in = range(c+r*L-r,c+r*L+r+1,1)
        top_out = range(c+(r+1)*L-r,c+(r+1)*L+r+1,1)
        top_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                    (jj,ii) in [(cj+r+1,x) for x in range(ci-r,ci+r+1)]]
        bottom_in = range(c-r*L-r,c-r*L+r+1,1)
        bottom_out = range(c-(r+1)*L-r,c-(r+1)*L+r+1,1)
        bottom_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                       (jj,ii) in [(cj-r,x) for x in range(ci-r,ci+r+1)]]
        right_in = range(c+r-r*L,c+r+r*L+1,L)
        right_out = range(c+r+1-r*L,c+r+1+r*L+1,L)
        right_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                      [(y,ci+r+1) for y in range(cj-r,cj+r+1)]]
        borders[cell] = (right_edge+bottom_edge+top_edge,
                         right_in+bottom_in+top_in,
                         right_out+bottom_out+top_out)

    # Cells with 2 borders (corner cells)
    # # Bottom left
    j=0; i=0; cell=j*l+i
    c = centers[cell]
    cj,ci=np.unravel_index(c,(L,L))
    top_in = range(c+r*L-r,c+r*L+r+1,1)
    top_out = range(c+(r+1)*L-r,c+(r+1)*L+r+1,1)
    top_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                (jj,ii) in [(cj+r+1,x) for x in range(ci-r,ci+r+1)]]
    right_in = range(c+r-r*L,c+r+r*L+1,L)
    right_out = range(c+r+1-r*L,c+r+1+r*L+1,L)
    right_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                  [(y,ci+r+1) for y in range(cj-r,cj+r+1)]]
    borders[cell] = (right_edge+top_edge,
                     right_in+top_in,
                     right_out+top_out)

    # # Bottom right
    j=0; i=l-1; cell=j*l+i
    c = centers[cell]
    cj,ci=np.unravel_index(c,(L,L))
    top_in = range(c+r*L-r,c+r*L+r+1,1)
    top_out = range(c+(r+1)*L-r,c+(r+1)*L+r+1,1)
    top_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                (jj,ii) in [(cj+r+1,x) for x in range(ci-r,ci+r+1)]]
    left_in = range(c-r-r*L,c-r+r*L+1,L)
    left_out = range(c-r-1-r*L,c-r-1+r*L+1,L)
    left_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                 [(y,ci-r) for y in range(cj-r,cj+r+1)]]
    borders[cell] = (left_edge+top_edge,
                     left_in+top_in,
                     left_out+top_out)

    # # Top left
    j=l-1; i=0; cell=j*l+i
    c = centers[cell]
    cj,ci=np.unravel_index(c,(L,L))
    bottom_in = range(c-r*L-r,c-r*L+r+1,1)
    bottom_out = range(c-(r+1)*L-r,c-(r+1)*L+r+1,1)
    bottom_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                   (jj,ii) in [(cj-r,x) for x in range(ci-r,ci+r+1)]]
    right_in = range(c+r-r*L,c+r+r*L+1,L)
    right_out = range(c+r+1-r*L,c+r+1+r*L+1,L)
    right_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                  [(y,ci+r+1) for y in range(cj-r,cj+r+1)]]
    borders[cell] = (right_edge+bottom_edge,
                     right_in+bottom_in,
                     right_out+bottom_out)

    # # Top right
    j=l-1; i=l-1; cell=j*l+i
    c = centers[cell]
    cj,ci=np.unravel_index(c,(L,L))
    bottom_in = range(c-r*L-r,c-r*L+r+1,1)
    bottom_out = range(c-(r+1)*L-r,c-(r+1)*L+r+1,1)
    bottom_edge = [num_edgeX+np.ravel_multi_index((jj,ii),(L+1,L)) for
                   (jj,ii) in [(cj-r,x) for x in range(ci-r,ci+r+1)]]
    left_in = range(c-r-r*L,c-r+r*L+1,L)
    left_out = range(c-r-1-r*L,c-r-1+r*L+1,L)
    left_edge = [np.ravel_multi_index((jj,ii),(L,L+1)) for (jj,ii) in
                 [(y,ci-r) for y in range(cj-r,cj+r+1)]]
    borders[cell] = (left_edge+bottom_edge,
                     left_in+bottom_in,
                     left_out+bottom_out)

    # Neighbors
    # Inner
    for j in range(1,l-1):
        for i in range(1,l-1):
            cell=j*l+i
            neighbors[cell] = [j*l+(i+1), j*l+(i-1), (j+1)*l+i,
                               (j-1)*l+i, (j-1)*l+(i-1),
                               (j-1)*l+(i+1), (j+1)*l+(i-1),
                               (j+1)*l+(i+1)]

    # Bottom
    j=0
    for i in range(1,l-1):
        cell=j*l+i
        neighbors[cell] = [j*l+(i+1), j*l+(i-1), (j+1)*l+i,
                           (j+1)*l+(i-1), (j+1)*l+(i+1)]

    # Top
    j=l-1
    for i in range(1,l-1):
        cell=j*l+i
        neighbors[cell] = [j*l+(i+1), j*l+(i-1), (j-1)*l+i,
                           (j-1)*l+(i-1), (j-1)*l+(i+1)]

    # Left
    i=0
    for j in range(1,l-1):
        cell=j*l+i
        neighbors[cell] = [j*l+(i+1), (j+1)*l+i, (j-1)*l+i,
                           (j-1)*l+(i+1), (j+1)*l+(i+1)]

    # Right
    i=l-1
    for j in range(1,l-1):
        cell=j*l+i
        neighbors[cell] = [j*l+(i-1), (j+1)*l+i, (j-1)*l+i,
                           (j-1)*l+(i-1), (j+1)*l+(i-1)]

    # Corner
    # Bottom left
    i=0; j=0; cell=j*l+i
    neighbors[cell] = [j*l+(i+1), (j+1)*l+i, (j+1)*l+(i+1)]
    # Bottom right
    i=l-1; j=0; cell=j*l+i
    neighbors[cell] = [j*l+(i-1), (j+1)*l+i, (j+1)*l+(i-1)]
    # Top left
    i=0; j=l-1; cell=j*l+i
    neighbors[cell] = [j*l+(i+1), (j-1)*l+i, (j-1)*l+(i+1)]
    # Top right
    i=l-1; j=l-1; cell=j*l+i
    neighbors[cell] = [j*l+(i-1), (j-1)*l+i, (j-1)*l+(i-1)]

    CG = {}
    CG['cells']=cells; CG['centers']=centers; CG['borders']=borders
    CG['neighbors']=neighbors
    CG['nx']=l; CG['ny']=l; CG['nz']=1
    CG['cartDims']=cgdim
    CG['cellDims']=(ny/Ny,nx/Nx)

    return CG


def dual_grid(G,CG):
    """Create dual grid.
    Return: DG [dict]
    DG:
        cells [list of array]
        inner_cells [list of array]
        edge_cells [list of array]
        edge_inner_cells [list of array]
        nodes [list of int]
        connectivity [dict]
            node_cells [list of list]
            node_edges [list of list]
            cell_nodes [list of list]
            cell_edges [list of list]
            edge_nodes [list of list]
        bases_geo [list of dict]
            cells_idxs [tuple]
            cells [2D array]
            inner_cells [2D array]
            local [dict]
            Grid [dict]
    """

    nz,ny,nx = G['cartDims']
    Ny,Nx = CG['cartDims']
    # only 2D square grid so far
    assert nz == 1
    assert ny == nx

    L=nx; l=Nx
    # check L/l is odd
    assert (L/l) % 2

    cells = get_dg_cells(L,l)
    inner_cells = get_dg_inner_cells(L,l)
    edge_cells = get_dg_edge_cells(L,l)
    edge_inner_cells = get_dg_edge_inner_cells(L,l)
    nodes = get_centers(L,l)
    connectivity = get_dg_connectivity(l)
    bases_geo = get_bases_geometry(L,l)

    DG = {}
    DG['cells']=cells
    DG['inner_cells']=inner_cells
    DG['edge_cells']=edge_cells
    DG['edge_inner_cells']=edge_inner_cells
    DG['nodes']=nodes
    DG['connectivity']=connectivity
    DG['bases_geo']=bases_geo
    return DG


def get_dg_connectivity(l):
    N_nodes=l*l; N_dg=(l+1)*(l+1); N_edge=2*l*(l+1)
    nex=l; ney=l; N_shift=l*(l+1)
    node_cells = [[] for _ in range(N_nodes)]
    node_edges = [[] for _ in range(N_nodes)]
    cell_edges = [[] for _ in range(N_dg)]
    cell_nodes = [[] for _ in range(N_dg)]
    edge_nodes = [[] for _ in range(N_edge)]

    for j in range(l):
        for i in range(l):
            node = j*l+i
            cells = [j*(l+1)+i, j*(l+1)+i+1, (j+1)*(l+1)+i, (j+1)*(l+1)+i+1]
            edges = [edgeX(j,i,nex), edgeX(j+1,i,nex),
                     edgeY(j,i,ney,N_shift), edgeY(j,i+1,ney,N_shift)]
            node_cells[node] = cells
            node_edges[node] = edges
            for k in cells:
                cell_nodes[k].append(node)
            for k in edges:
                edge_nodes[k].append(node)

    # Full dual cells
    for j in range(1,l):
        for i in range(1,l):
            cell = j*(l+1)+i
            cell_edges[cell] = [edgeX(j,i,nex), edgeX(j,i-1,nex),
                                edgeY(j,i,ney,N_shift),
                                edgeY(j-1,i,ney,N_shift)]

    # Half dual cells
    # # Bottom
    j = 0
    for i in range(1,l):
        cell = j*(l+1)+i
        cell_edges[cell] = [edgeX(j,i,nex), edgeX(j,i-1,nex),
                            edgeY(j,i,ney,N_shift)]

    # # Top
    j = l
    for i in range(1,l):
        cell = j*(l+1)+i
        cell_edges[cell] = [edgeX(j,i,nex), edgeX(j,i-1,nex),
                            edgeY(j-1,i,ney,N_shift)]

    # # Left
    i = 0
    for j in range(1,l):
        cell = j*(l+1)+i
        cell_edges[cell] = [edgeX(j,i,nex), edgeY(j,i,ney,N_shift),
                            edgeY(j-1,i,ney,N_shift)]

    # # Right
    i = l
    for j in range(1,l):
        cell = j*(l+1)+i
        cell_edges[cell] = [edgeX(j,i-1,nex), edgeY(j,i,ney,N_shift),
                            edgeY(j-1,i,ney,N_shift)]

    # Corners
    i=0; j=0
    cell = j*(l+1)+i
    cell_edges[cell] = [edgeX(j,i,nex), edgeY(j,i,ney,N_shift)]
    i=l; j=0
    cell = j*(l+1)+i
    cell_edges[cell] = [edgeX(j,i-1,nex), edgeY(j,i,ney,N_shift)]
    i=0; j=l
    cell = j*(l+1)+i
    cell_edges[cell] = [edgeX(j,i,nex), edgeY(j-1,i,ney,N_shift)]
    i=l; j=l
    cell = j*(l+1)+i
    cell_edges[cell] = [edgeX(j,i-1,nex), edgeY(j-1,i,ney,N_shift)]

    connectivity = {}
    connectivity['node_cells'] = node_cells
    connectivity['node_edges'] = node_edges
    connectivity['cell_nodes'] = cell_nodes
    connectivity['cell_edges'] = cell_edges
    connectivity['edge_nodes'] = edge_nodes
    return connectivity


def edgeX(j,i,nex):
    return j*nex+i


def edgeY(j,i,ney,N_shift):
    return N_shift + i*ney+j


def get_dg_cells(L,l):
    N = L*L
    N_dg = (l+1)*(l+1)
    grid = np.reshape(np.arange(N),(L,L))
    cells = [-1]*N_dg
    lp = L/l

    # Full cells
    for j in range(1,l):
        for i in range(1,l):
            cell = j*(l+1)+i
            dx = slice((i-1)*lp+(lp+1)/2-1,(i-1)*lp+(lp+1)/2+lp)
            dy = slice((j-1)*lp+(lp+1)/2-1,(j-1)*lp+(lp+1)/2+lp)
            cells[cell] = grid[dy,dx].ravel()

    # Side cells (half cells)
    # # Bottom
    j = 0
    for i in range(1,l):
        cell = j*(l+1)+i
        dx = slice((i-1)*lp+(lp+1)/2-1,(i-1)*lp+(lp+1)/2+lp)
        dy = slice(0,(lp+1)/2)
        cells[cell] = grid[dy,dx].ravel()

    # # Top
    j = l
    for i in range(1,l):
        cell = j*(l+1)+i
        dx = slice((i-1)*lp+(lp+1)/2-1,(i-1)*lp+(lp+1)/2+lp)
        dy = slice(L-(lp+1)/2,L)
        cells[cell] = grid[dy,dx].ravel()

    # # Left
    i = 0
    for j in range(1,l):
        cell = j*(l+1)+i
        dx = slice(0,(lp+1)/2)
        dy = slice((j-1)*lp+(lp+1)/2-1,(j-1)*lp+(lp+1)/2+lp)
        cells[cell] = grid[dy,dx].ravel()

    # # Right
    i = l
    for j in range(1,l):
        cell = j*(l+1)+i
        dx = slice(L-(lp+1)/2,L)
        dy = slice((j-1)*lp+(lp+1)/2-1,(j-1)*lp+(lp+1)/2+lp)
        cells[cell] = grid[dy,dx].ravel()

    # Corner cells (quarter cells)
    # # Bottom left
    j=0; i=0; cell=j*(l+1)+i
    dx = slice(0,(lp+1)/2)
    dy = slice(0,(lp+1)/2)
    cells[cell] = grid[dy,dx].ravel()

    # # Bottom right
    j=0; i=l; cell=j*(l+1)+i
    dx = slice(L-(lp+1)/2,L)
    dy = slice(0,(lp+1)/2)
    cells[cell] = grid[dy,dx].ravel()

    # # Top left
    j=l; i=0; cell=j*(l+1)+i
    dx = slice(0,(lp+1)/2)
    dy = slice(L-(lp+1)/2,L)
    cells[cell] = grid[dy,dx].ravel()

    # # Top right
    j=l; i=l; cell=j*(l+1)+i
    dx = slice(L-(lp+1)/2,L)
    dy = slice(L-(lp+1)/2,L)
    cells[cell] = grid[dy,dx].ravel()

    return cells


def get_dg_inner_cells(L,l):
    N = L*L
    N_dg = (l+1)*(l+1)
    grid = np.reshape(np.arange(N),(L,L))
    inner_cells = [-1]*N_dg
    lp = L/l

    # Build inner_cells[]
    # Full cells
    for j in range(1,l):
        for i in range(1,l):
            cell = j*(l+1)+i
            dx = slice((i-1)*lp+(lp+1)/2,(i-1)*lp+(lp+1)/2+lp-1)
            dy = slice((j-1)*lp+(lp+1)/2,(j-1)*lp+(lp+1)/2+lp-1)
            inner_cells[cell] = grid[dy,dx].ravel()

    # Side cells (half cells)
    # # Bottom
    j = 0
    for i in range(1,l):
        cell = j*(l+1)+i
        dx = slice((i-1)*lp+(lp+1)/2,(i-1)*lp+(lp+1)/2+lp-1)
        dy = slice(0,(lp-1)/2)
        inner_cells[cell] = grid[dy,dx].ravel()

    # # Top
    j = l
    for i in range(1,l):
        cell = j*(l+1)+i
        dx = slice((i-1)*lp+(lp+1)/2,(i-1)*lp+(lp+1)/2+lp-1)
        dy = slice(L-(lp-1)/2,L)
        inner_cells[cell] = grid[dy,dx].ravel()

    # # Left
    i = 0
    for j in range(1,l):
        cell = j*(l+1)+i
        dx = slice(0,(lp-1)/2)
        dy = slice((j-1)*lp+(lp+1)/2,(j-1)*lp+(lp+1)/2+lp-1)
        inner_cells[cell] = grid[dy,dx].ravel()

    # # Right
    i = l
    for j in range(1,l):
        cell = j*(l+1)+i
        dx = slice(L-(lp-1)/2,L)
        dy = slice((j-1)*lp+(lp+1)/2,(j-1)*lp+(lp+1)/2+lp-1)
        inner_cells[cell] = grid[dy,dx].ravel()

    # Corner cells (quarter cells)
    # # Bottom left
    j=0; i=0; cell=j*(l+1)+i
    dx = slice(0,(lp-1)/2)
    dy = slice(0,(lp-1)/2)
    inner_cells[cell] = grid[dy,dx].ravel()

    # # Bottom right
    j=0; i=l; cell=j*(l+1)+i
    dx = slice(L-(lp-1)/2,L)
    dy = slice(0,(lp-1)/2)
    inner_cells[cell] = grid[dy,dx].ravel()

    # # Top left
    j=l; i=0; cell=j*(l+1)+i
    dx = slice(0,(lp-1)/2)
    dy = slice(L-(lp-1)/2,L)
    inner_cells[cell] = grid[dy,dx].ravel()

    # # Top right
    j=l; i=l; cell=j*(l+1)+i
    dx = slice(L-(lp-1)/2,L)
    dy = slice(L-(lp-1)/2,L)
    inner_cells[cell] = grid[dy,dx].ravel()

    return inner_cells


def get_dg_edge_cells(L,l):
    N = L*L
    grid = np.reshape(np.arange(N),(L,L))
    N_edge=2*l*(l+1);
    edge_cells = [-1]*N_edge
    lp=L/l; nex=l; ney=l;
    r=(lp-1)/2; N_shift=l*(l+1)

    # Build edge_cells[]
    # Full vertical edges
    for j in range(1,l):
        for i in range(nex):
            edge = j*nex+i
            dx = i*lp+r
            dy = slice((j-1)*lp+r,(j-1)*lp+r+lp+1)
            edge_cells[edge] = grid[dy,dx].ravel()

    # Full horizontal edges
    for i in range(1,l):
        for j in range(ney):
            edge = i*ney+j + N_shift
            dy = j*lp+r
            dx = slice((i-1)*lp+r,(i-1)*lp+r+lp+1)
            edge_cells[edge] = grid[dy,dx].ravel()

    # Half vertical edges
    # # Bottom
    j = 0
    for i in range(nex):
        edge = j*nex+i
        dx = i*lp+r
        dy = slice(0,r+1)
        edge_cells[edge] = grid[dy,dx].ravel()

    # # Top
    j = l
    for i in range(nex):
        edge = j*nex+i
        dx = i*lp+r
        dy = slice(L-r-1,L)
        edge_cells[edge] = grid[dy,dx].ravel()

    # Half horizontal edges
    # # Left
    i = 0
    for j in range(ney):
        edge = i*ney+j + N_shift
        dy = j*lp+r
        dx = slice(0,r+1)
        edge_cells[edge] = grid[dy,dx].ravel()

    # # Right
    i = l
    for j in range(ney):
        edge = i*ney+j + N_shift
        dy = j*lp+r
        dx = slice(L-r-1,L)
        edge_cells[edge] = grid[dy,dx].ravel()

    return edge_cells


def get_dg_edge_inner_cells(L,l):
    N = L*L
    grid = np.reshape(np.arange(N),(L,L))
    N_edge=2*l*(l+1);
    edge_inner_cells = [-1]*N_edge
    lp=L/l; nex=l; ney=l;
    r=(lp-1)/2; N_shift=l*(l+1)

    # Build edge_inner_cells[]
    # Full vertical edges
    for j in range(1,l):
        for i in range(nex):
            edge = j*nex+i
            dx = i*lp+r
            dy = slice((j-1)*lp+r+1,(j-1)*lp+r+lp)
            edge_inner_cells[edge] = grid[dy,dx].ravel()

    # Full horizontal edges
    for i in range(1,l):
        for j in range(ney):
            edge = i*ney+j + N_shift
            dy = j*lp+r
            dx = slice((i-1)*lp+r+1,(i-1)*lp+r+lp)
            edge_inner_cells[edge] = grid[dy,dx].ravel()

    # Half vertical edges
    # # Bottom
    j = 0
    for i in range(nex):
        edge = j*nex+i
        dx = i*lp+r
        dy = slice(0,r)
        edge_inner_cells[edge] = grid[dy,dx].ravel()

    # # Top
    j = l
    for i in range(nex):
        edge = j*nex+i
        dx = i*lp+r
        dy = slice(L-r,L)
        edge_inner_cells[edge] = grid[dy,dx].ravel()

    # Half horizontal edges
    # # Left
    i = 0
    for j in range(ney):
        edge = i*ney+j + N_shift
        dy = j*lp+r
        dx = slice(0,r)
        edge_inner_cells[edge] = grid[dy,dx].ravel()

    # # Right
    i = l
    for j in range(ney):
        edge = i*ney+j + N_shift
        dy = j*lp+r
        dx = slice(L-r,L)
        edge_inner_cells[edge] = grid[dy,dx].ravel()

    return edge_inner_cells


def get_bases_geometry(L,l):
    N_fg = L*L; N_cg = l*l;
    lp=L/l; r=(lp-1)/2
    grid = np.reshape(np.arange(N_fg),(L,L))
    bases_geo=[{} for _ in range(N_cg)]

    # Inner
    nx=2*lp+1; ny=2*lp+1; nz=1
    N=nx*ny*nz
    for j in range(1,l-1):
        for i in range(1,l-1):
            idx = j*l+i
            dy = slice(r+lp*(j-1),r+lp*(j+1)+1)
            dx = slice(r+lp*(i-1),r+lp*(i+1)+1)
            dyi = slice(r+lp*(j-1)+1,r+lp*(j+1))
            dxi = slice(r+lp*(i-1)+1,r+lp*(i+1))
            bases_geo[idx]['cells_idxs']=(dy,dx)
            bases_geo[idx]['cells']=grid[dy,dx]
            bases_geo[idx]['inner_cells']=grid[dyi,dxi]
            bottom = range(0,nx,1)
            top = range(N-nx,N,1)
            left = range(0,N,nx)
            right = range(nx-1,N,nx)
            border = np.unique(bottom+top+left+right)
            bases_geo[idx]['local']={}
            bases_geo[idx]['local']['border']=border
            bases_geo[idx]['local']['center']=lp*nx+lp
            Grid={}
            Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
            Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
            bases_geo[idx]['Grid']=Grid

    # Bottom & Top
    nx=2*lp+1; ny=(lp+1)+r; nz=1
    N=nx*ny*nz
    j=0
    for i in range(1,l-1):
        idx = j*l+i
        dy = slice(0,r+lp*(j+1)+1)
        dx = slice(r+lp*(i-1),r+lp*(i+1)+1)
        dyi = slice(0,r+lp*(j+1))
        dxi = slice(r+lp*(i-1)+1,r+lp*(i+1))
        bases_geo[idx]['cells_idxs']=(dy,dx)
        bases_geo[idx]['cells']=grid[dy,dx]
        bases_geo[idx]['inner_cells']=grid[dyi,dxi]
        top = range(N-nx,N,1)
        left = range(0,N,nx)
        right = range(nx-1,N,nx)
        border = np.unique(top+left+right)
        bases_geo[idx]['local']={}
        bases_geo[idx]['local']['border']=border
        bases_geo[idx]['local']['center']=r*nx+lp
        Grid={}
        Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
        Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
        bases_geo[idx]['Grid']=Grid
    j=l-1
    for i in range(1,l-1):
        idx = j*l+i
        dy = slice(r+lp*(j-1),L)
        dx = slice(r+lp*(i-1),r+lp*(i+1)+1)
        dyi = slice(r+lp*(j-1)+1,L)
        dxi = slice(r+lp*(i-1)+1,r+lp*(i+1))
        bases_geo[idx]['cells_idxs']=(dy,dx)
        bases_geo[idx]['cells']=grid[dy,dx]
        bases_geo[idx]['inner_cells']=grid[dyi,dxi]
        bottom = range(0,nx,1)
        left = range(0,N,nx)
        right = range(nx-1,N,nx)
        border = np.unique(bottom+left+right)
        bases_geo[idx]['local']={}
        bases_geo[idx]['local']['border']=border
        bases_geo[idx]['local']['center']=lp*nx+lp
        Grid={}
        Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
        Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
        bases_geo[idx]['Grid']=Grid

    # Left & Right
    nx=(lp+1)+r; ny=2*lp+1; nz=1
    N=nx*ny*nz
    i=0
    for j in range(1,l-1):
        idx = j*l+i
        dy = slice(r+lp*(j-1),r+lp*(j+1)+1)
        dx = slice(0,r+lp*(i+1)+1)
        dyi = slice(r+lp*(j-1)+1,r+lp*(j+1))
        dxi = slice(0,r+lp*(i+1))
        bases_geo[idx]['cells_idxs']=(dy,dx)
        bases_geo[idx]['cells']=grid[dy,dx]
        bases_geo[idx]['inner_cells']=grid[dyi,dxi]
        bottom = range(0,nx,1)
        top = range(N-nx,N,1)
        right = range(nx-1,N,nx)
        border = np.unique(bottom+top+right)
        bases_geo[idx]['local']={}
        bases_geo[idx]['local']['border']=border
        bases_geo[idx]['local']['center']=lp*nx+r
        Grid={}
        Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
        Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
        bases_geo[idx]['Grid']=Grid
    i=l-1
    for j in range(1,l-1):
        idx = j*l+i
        dy = slice(r+lp*(j-1),r+lp*(j+1)+1)
        dx = slice(r+lp*(i-1),L)
        dyi = slice(r+lp*(j-1)+1,r+lp*(j+1))
        dxi = slice(r+lp*(i-1)+1,L)
        bases_geo[idx]['cells_idxs']=(dy,dx)
        bases_geo[idx]['cells']=grid[dy,dx]
        bases_geo[idx]['inner_cells']=grid[dyi,dxi]
        bottom = range(0,nx,1)
        top = range(N-nx,N,1)
        left = range(0,N,nx)
        border = np.unique(bottom+top+left)
        bases_geo[idx]['local']={}
        bases_geo[idx]['local']['border']=border
        bases_geo[idx]['local']['center']=lp*nx+lp
        Grid={}
        Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
        Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
        bases_geo[idx]['Grid']=Grid

    # Corners
    nx=(lp+1)+r; ny=(lp+1)+r; nz=1
    N=nx*ny*nz
    i=0; j=0
    idx = j*l+i
    dy = slice(0,r+lp*(j+1)+1)
    dx = slice(0,r+lp*(i+1)+1)
    dyi = slice(0,r+lp*(j+1))
    dxi = slice(0,r+lp*(i+1))
    bases_geo[idx]['cells_idxs']=(dy,dx)
    bases_geo[idx]['cells']=grid[dy,dx]
    bases_geo[idx]['inner_cells']=grid[dyi,dxi]
    top = range(N-nx,N,1)
    right = range(nx-1,N,nx)
    border = np.unique(top+right)
    bases_geo[idx]['local']={}
    bases_geo[idx]['local']['border']=border
    bases_geo[idx]['local']['center']=r*nx+r
    Grid={}
    Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
    Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
    bases_geo[idx]['Grid']=Grid
    i=l-1; j=0
    idx = j*l+i
    dy = slice(0,r+lp*(j+1)+1)
    dx = slice(r+lp*(i-1),L)
    dyi = slice(0,r+lp*(j+1))
    dxi = slice(r+lp*(i-1)+1,L)
    bases_geo[idx]['cells_idxs']=(dy,dx)
    bases_geo[idx]['cells']=grid[dy,dx]
    bases_geo[idx]['inner_cells']=grid[dyi,dxi]
    top = range(N-nx,N,1)
    left = range(0,N,nx)
    border = np.unique(top+left)
    bases_geo[idx]['local']={}
    bases_geo[idx]['local']['border']=border
    bases_geo[idx]['local']['center']=r*nx+lp
    Grid={}
    Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
    Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
    bases_geo[idx]['Grid']=Grid
    i=0; j=l-1
    idx = j*l+i
    dy = slice(r+lp*(j-1),L)
    dx = slice(0,r+lp*(i+1)+1)
    dyi = slice(r+lp*(j-1)+1,L)
    dxi = slice(0,r+lp*(i+1))
    bases_geo[idx]['cells_idxs']=(dy,dx)
    bases_geo[idx]['cells']=grid[dy,dx]
    bases_geo[idx]['inner_cells']=grid[dyi,dxi]
    bottom = range(0,nx,1)
    right = range(nx-1,N,nx)
    border = np.unique(bottom+right)
    bases_geo[idx]['local']={}
    bases_geo[idx]['local']['border']=border
    bases_geo[idx]['local']['center']=lp*nx+r
    Grid={}
    Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
    Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
    bases_geo[idx]['Grid']=Grid
    i=l-1; j=l-1
    idx = j*l+i
    dy = slice(r+lp*(j-1),L)
    dx = slice(r+lp*(i-1),L)
    dyi = slice(r+lp*(j-1)+1,L)
    dxi = slice(r+lp*(i-1)+1,L)
    bases_geo[idx]['cells_idxs']=(dy,dx)
    bases_geo[idx]['cells']=grid[dy,dx]
    bases_geo[idx]['inner_cells']=grid[dyi,dxi]
    bottom = range(0,nx,1)
    left = range(0,N,nx)
    border = np.unique(bottom+left)
    bases_geo[idx]['local']={}
    bases_geo[idx]['local']['border']=border
    bases_geo[idx]['local']['center']=lp*nx+lp
    Grid={}
    Grid['nz'],Grid['ny'],Grid['nx']=nz,ny,nx
    Grid['hz'],Grid['hy'],Grid['hx']=1,1,1
    bases_geo[idx]['Grid']=Grid

    return bases_geo


def get_centers(L,l):
    N_centers = l*l
    centers = [-1]*N_centers
    lp=L/l; r = (lp-1)/2
    for j in range(l):
        for i in range(l):
            cell = j*l+i
            centers[cell] = (j*lp+r)*L+i*lp+r
    return centers


def csi(CG):
    """
    Get the indices of corner, side and inner bases of a ny x nx grid
        Return: corners [dict], sides [dict], inners [list]
    """

    ny = CG['ny']
    nx = CG['nx']

    bottom_left = 0
    bottom_right = nx - 1
    top_left = (ny - 1) * nx
    top_right = (ny - 1) * nx + nx - 1
    corners = {}
    corners['bottom_left'] = bottom_left
    corners['bottom_right'] = bottom_right
    corners['top_left'] = top_left
    corners['top_right'] = top_right

    bottom = [i for i in range(1, nx-1)]
    top = [i for i in range((ny-1)*nx + 1, (ny-1)*nx + nx-1)]
    left = [j*nx for j in range(1, ny-1)]
    right = [j*nx + ny-1 for j in range(1, ny-1)]
    sides = {}
    sides['bottom'] = bottom
    sides['top'] = top
    sides['left'] = left
    sides['right'] = right

    inners = [j*nx + i for (j,i) in itertools.product(range(1, ny-1),
                                                      range(1, nx-1))]

    return corners, sides, inners


def get_basis_shapes(G, CG):
    """
    Return: inner, side, and corner shapes.
    """
    ny, nx = G['ny'], G['nx']
    Ny, Nx = CG['ny'], CG['nx']
    m_short = nx/Nx + (nx/Nx)/2 + 1
    m_long = 2*nx/Nx + 1
    return [(m_long, m_long), (m_short, m_long), (m_short, m_short)]


def get_basis_amounts(CG):
    """
    Return: amount of inner, side, and corner bases.
    """
    Ny, Nx = CG['ny'], CG['nx']
    n_inn = (Ny-2)*(Nx-2)
    n_sid = 2*(Ny-2) + 2*(Nx-2)
    n_cor = 4

    return [n_inn, n_sid, n_cor]
