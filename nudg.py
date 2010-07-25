# PyNudg - the python Nodal DG Environment
# (C) 2009, 2010 Tim Warburton, Xueyu Zhu, Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




from __future__ import division
from __future__ import with_statement

from math import sqrt
import numpy as np
import numpy.linalg as la

try:
    import enthought.mayavi.mlab as mv
except ImportError:
    do_vis = False
else:
    do_vis = True


NODETOL = 1e-12
eps = np.finfo(float).eps




# {{{ Low-storage Runge-Kutta coefficients
rk4a = np.array([0 ,
        -567301805773/1357537059087,
        -2404267990393/2016746695238,
        -3550918686646/2091501179385,
        -1275806237668/842570457699])
rk4b = [ 1432997174477/9575080441755,
         5161836677717/13612068292357,
         1720146321549/2090206949498,
         3134564353537/4481467310338,
         2277821191437/14882151754819]
rk4c = [0,
         1432997174477/9575080441755,
         2526269341429/6820363962896,
         2006345519317/3224310063776,
         2802321613138/2924317926251]

# }}}




def fact(z):
    g = 1
    for i in range(1, np.int32(z)):
        g = g*i

    return g

def MeshReaderGambit2D(file_name):
    """Read in basic grid information to build grid
    Note: Gambit(Fluent, Inc) *.neu format is assumed.

    Returns (Nv, VX, VY, K, EToV).
    """

    with open(file_name, 'r') as inf:
        # read after intro
        for i in range(6):
            line = inf.readline()

        # Find number of nodes and number of elements
        dims = inf.readline().split()
        Nv = np.int(dims[0]); K = np.int32(dims[1])

        for i in range(2):
            line = inf.readline()

        # read node coordinates
        VX = np.zeros(Nv); VY = np.zeros(Nv)
        for i  in range(Nv):
            tmpx = inf.readline().split()
            VX[i] = float(tmpx[1]); VY[i] = float(tmpx[2])

        for i in range(2):
            line = inf.readline()

        # read element to node connectivity
        EToV = np.zeros((K, 3))
        for k in range(K):
            tmpcon= inf.readline().split()
            EToV[k,0] = np.int32(tmpcon[3])-1
            EToV[k,1] = np.int32(tmpcon[4])-1
            EToV[k,2] = np.int32(tmpcon[5])-1

        return Nv, VX, VY, K, EToV


def JacobiP(x, alpha, beta, N):
    """ function P = JacobiP(x, alpha, beta, N)
         Purpose: Evaluate Jacobi Polynomial of type (alpha, beta) > -1
                  (alpha+beta <> -1) at points x for order N and
                  returns P[1:length(xp))]
         Note   : They are normalized to be orthonormal."""
    N = np.int32(N)
    Nx = len(x)
    if x.shape[0]>1:
        x = x.T
    # Storage for recursive construction
    PL = np.zeros((np.int32(Nx), np.int32(N+1)))

    # Initial values P_0(x) and P_1(x)
    gamma0 = np.power(2., alpha+beta+1)/(alpha+beta+1.)*fact(alpha+1)*fact(beta+1)/fact(alpha+beta+1)

    #
    PL[:,0] = 1.0/sqrt(gamma0)
    if N==0:
        return PL[:,0]

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[:,1] = ((alpha+beta+2)*x/2 + (alpha-beta)/2)/sqrt(gamma1)
    if N==1:
        return PL[:,1]

    # Repeat value in recurrence.
    aold = 2./(2.+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
            h1 = 2.*i+alpha+beta

            foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
            anew = 2./(h1+2.)*sqrt(foo)

            bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
            PL[:, i+1] = ( -aold*PL[:, i-1] + np.multiply(x-bnew, PL[:, i]) )/anew
            aold =anew

    return PL[:, N]

def Vandermonde1D(N, xp):
    """Initialize the 1D Vandermonde Matrix.
    V_{ij} = phi_j(xp_i)
    """

    Nx = np.int32(xp.shape[0])
    N  = np.int32(N)
    V1D = np.zeros((Nx, N+1))

    for j in range(N+1):
            V1D[:, j] = JacobiP(xp, 0, 0, j).T # give the tranpose of Jacobi.p

    return V1D

def JacobiGQ(alpha, beta, N):
    """Compute the N'th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi
    polynomial, of type (alpha, beta) > -1 ( <> -0.5).
    """

    if N==0:
        x[0]=(alpha-beta)/(alpha+beta+2)
        w[0] = 2
        return x, w

    # Form symmetric matrix from recurrence.
    J    = np.zeros(N+1)
    h1   = 2*np.arange(N+1) + alpha + beta
    temp = np.arange(N) + 1.0
    J    = np.diag(-1.0/2.0*(alpha**2-beta**2)/(h1+2.0)/h1) + np.diag(2.0/(h1[0:N]+2.0)*np.sqrt(temp*(temp+alpha+beta)*(temp+alpha)*(temp+beta)/(h1[0:N]+1.0)/(h1[0:N]+3.0)),1)

    if alpha+beta < 10*np.finfo(float).eps :
        J[0,0] = 0.0
    J = J + J.T

    # Compute quadrature by eigenvalue solve
    D, V = la.eig(J)
    ind = np.argsort(D)
    D = D[ind]
    V = V[:, ind]
    x = D
    w = (V[0,:].T)**2*2**(alpha+beta+1)/(alpha+beta+1)*fact(alpha+1)*fact(beta+1)/fact(alpha+beta+1)

    return x, w

def  JacobiGL(alpha, beta, N):
    """Compute the Nth order Gauss Lobatto quadrature points, x,
    associated with the Jacobi polynomial, of type (alpha, beta) > -1 ( <> -0.5).
    """

    x = np.zeros((N+1,1))
    if N==1:
        x[0]=-1.0
        x[1]=1.0
        return x

    xint, w = JacobiGQ(alpha+1, beta+1, N-2)

    x = np.hstack((-1.0, xint,1.0))

    return x.T



def Warpfactor(N, rout):
    """Compute scaled warp function at order N based on
    rout interpolation nodes.
    """

    # Compute LGL and equidistant node distribution
    LGLr = JacobiGL(0,0, N); req  = np.linspace(-1,1, N+1)
    # Compute V based on req
    Veq = Vandermonde1D(N, req)
    # Evaluate Lagrange polynomial at rout
    Nr = len(rout); Pmat = np.zeros((N+1, Nr))
    for i in range(N+1):
        Pmat[i,:] = JacobiP(rout.T[0,:], 0, 0, i)

    Lmat = la.solve(Veq.T, Pmat)

    # Compute warp factor
    warp = np.dot(Lmat.T, LGLr - req)
    warp = warp.reshape(Lmat.shape[1],1)
    zerof = (abs(rout)<1.0-1.0e-10)
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp*(zerof-1)
    return warp

def Nodes2D(N):
    """Compute (x, y) nodes in equilateral triangle for polynomial
    of order N.
    """

    alpopt = np.array([0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,\
            1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223,1.6258])

    # Set optimized parameter, alpha, depending on order N
    if N< 16:
        alpha = alpopt[N-1]
    else:
        alpha = 5.0/3.0


    # total number of nodes
    Np = (N+1)*(N+2)/2.0

    # Create equidistributed nodes on equilateral triangle
    L1 = np.zeros((Np,1)); L2 = np.zeros((Np,1)); L3 = np.zeros((Np,1))
    sk = 0
    for n in range(N+1):
        for m in range(N+1-n):
            L1[sk] = n/N
            L3[sk] = m/N
            sk = sk+1

    L2 = 1.0-L1-L3
    x = -L2+L3; y = (-L2-L3+2*L1)/sqrt(3.0)

    # Compute blending function at each node for each edge
    blend1 = 4*L2*L3; blend2 = 4*L1*L3; blend3 = 4*L1*L2

    # Amount of warp for each node, for each edge
    warpf1 = Warpfactor(N, L3-L2)
    warpf2 = Warpfactor(N, L1-L3)
    warpf3 = Warpfactor(N, L2-L1)

    # Combine blend & warp
    warp1 = blend1*warpf1*(1 + (alpha*L1)**2)
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2)
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2)

    # Accumulate deformations associated with each edge
    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3
    return x, y

def xytors(x, y):
    """From (x, y) in equilateral triangle to (r, s) coordinates in standard triangle."""

    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0

    r = -L2 + L3 - L1; s = -L2 - L3 + L1
    return r, s

def rstoab(r, s):
    """Transfer from (r, s) -> (a, b) coordinates in triangle.
    """

    Np = len(r); a = np.zeros((Np,1))
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n])-1
        else:
            a[n] = -1

    b = s
    return a, b

def Simplex2DP(a, b, i, j):
    """Evaluate 2D orthonormal polynomial
    on simplex at (a, b) of order (i, j).
    """

    h1 = JacobiP(a,0,0, i).reshape(len(a),1)
    h2 = JacobiP(b,2*i+1,0, j).reshape(len(a),1)
    P  = np.sqrt(2.0)*h1*h2*(1-b)**i
    return P[:,0]

def Vandermonde2D(N, r, s):
    """Initialize the 2D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i)
    """

    V2D = np.zeros((len(r),(N+1)*(N+2)/2))

    # Transfer to (a, b) coordinates
    a, b = rstoab(r, s)

    # build the Vandermonde matrix
    sk = 0

    for i in range(N+1):
        for j in range(N-i+1):
            V2D[:, sk] = Simplex2DP(a, b, i, j)
            sk = sk+1
    return V2D

def GradJacobiP(z, alpha, beta, N):
    """Evaluate the derivative of the orthonormal Jacobi polynomial 
    of type (alpha, beta)>-1, at points x for order N and 
    returns dP[1:len(xp))].
    """
    Nx = np.int32(z.shape[0])
    dP = np.zeros((Nx, 1))
    if N==0:
        dP[:,0] = 0.0
    else:
        dP[:,0]= sqrt(N*(N+alpha+beta+1.))*JacobiP(z, alpha+1, beta+1, N-1)

    return dP

def GradSimplex2DP(a, b, id, jd):
    """Return the derivatives of the modal basis (id, jd) on the 
    2D simplex at (a, b).
    """

    fa  = JacobiP(a, 0, 0, id).reshape(len(a),1)
    dfa = GradJacobiP(a, 0, 0, id)
    gb  = JacobiP(b, 2*id+1,0, jd).reshape(len(b),1)
    dgb = GradJacobiP(b, 2*id+1,0, jd)

    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
    dmodedr = dfa*gb
    if(id>0):
        dmodedr = dmodedr*((0.5*(1-b))**(id-1))

    # s-derivative
    # d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa*(gb*(0.5*(1+a)))
    if(id>0):
        dmodeds = dmodeds*((0.5*(1-b))**(id-1))
    tmp = dgb*((0.5*(1-b))**id)
    if(id>0):
        tmp = tmp-0.5*id*gb*((0.5*(1-b))**(id-1))
    dmodeds = dmodeds+fa*tmp
    # Normalize
    dmodedr = 2**(id+0.5)*dmodedr
    dmodeds = 2**(id+0.5)*dmodeds

    return dmodedr[:,0], dmodeds[:,0]


def GradVandermonde2D(N, r, s):
    """Initialize the gradient of the modal basis 
    (i, j) at (r, s) at order N.
    """

    V2Dr = np.zeros((len(r),(N+1)*(N+2)/2))
    V2Ds = np.zeros((len(r),(N+1)*(N+2)/2))

    # find tensor-product coordinates
    a, b = rstoab(r, s)
    # Initialize matrices
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk = sk+1
    return V2Dr, V2Ds

def Dmatrices2D(N, r, s, V):
    """Initialize the (r, s) differentiation matriceon the simplex, 
    evaluated at (r, s) at order N.
    """

    Vr, Vs = GradVandermonde2D(N, r, s)
    invV   = la.inv(V)
    Dr     = np.dot(Vr, invV)
    Ds     = np.dot(Vs, invV)
    return Dr, Ds

def Lift2D(ldis, r, s, V, Fmask):
    """Compute surface to volume lift term for DG formulation."""
    l = ldis

    Emat = np.zeros((l.Np, l.Nfaces*l.Nfp))

    # face 1
    faceR = r[Fmask[:,0]]
    V1D = Vandermonde1D(l.N, faceR)
    massEdge1 = la.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[:,0],0:l.Nfp] = massEdge1

    # face 2
    faceR = r[Fmask[:,1]]
    V1D = Vandermonde1D(l.N, faceR)
    massEdge2 = la.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[:,1], l.Nfp:2*l.Nfp] = massEdge2

    # face 3
    faceS = s[Fmask[:,2]]
    V1D = Vandermonde1D(l.N, faceS)
    massEdge3 = la.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[:,2],2*l.Nfp:3*l.Nfp] = massEdge3

    # inv(mass matrix)*\I_n (L_i, L_j)_{edge_n}
    LIFT = np.dot(V, np.dot(V.T, Emat))
    return LIFT

def GeometricFactors2D(x, y, Dr, Ds):
    """Compute the metric elements for the local mappings of the elements
    Returns [rx, sx, ry, sy, J].
    """
    # Calculate geometric factors
    xr = np.dot(Dr, x); xs = np.dot(Ds, x)
    yr = np.dot(Dr, y); ys = np.dot(Ds, y); J = -xs*yr + xr*ys
    rx = ys/J; sx =-yr/J; ry =-xs/J; sy = xr/J
    return rx, sx, ry, sy, J

def Normals2D(ldis, x, y, K):
    """Compute outward pointing normals at elements faces 
    and surface Jacobians.
    """

    l = ldis
    xr = np.dot(l.Dr, x)
    yr = np.dot(l.Dr, y)
    xs = np.dot(l.Ds, x)
    ys = np.dot(l.Ds, y)
    J = xr*ys-xs*yr

    # interpolate geometric factors to face nodes
    fxr = xr[l.FmaskF, :]; fxs = xs[l.FmaskF, :]
    fyr = yr[l.FmaskF, :]; fys = ys[l.FmaskF, :]

    # build normals
    nx = np.zeros((3*l.Nfp, K))
    ny = np.zeros((3*l.Nfp, K))
    fid1 = np.arange(l.Nfp).reshape(l.Nfp,1)
    fid2 = fid1+l.Nfp
    fid3 = fid2+l.Nfp

    # face 1

    nx[fid1, :] =  fyr[fid1, :]
    ny[fid1, :] = -fxr[fid1, :]

    # face 2
    nx[fid2, :] =  fys[fid2, :]-fyr[fid2, :]
    ny[fid2, :] = -fxs[fid2, :]+fxr[fid2, :]

    # face 3
    nx[fid3, :] = -fys[fid3, :]
    ny[fid3, :] =  fxs[fid3, :]

    # normalise
    sJ = np.sqrt(nx*nx+ny*ny)
    nx = nx/sJ; ny = ny/sJ
    return nx, ny, sJ

def Connect2D(EToV):
    """Build global connectivity arrays for grid based on 
    standard EToV input array from grid generator.
    """

    EToV = EToV.astype(np.intp)
    Nfaces = 3
    # Find number of elements and vertices
    K = EToV.shape[0]
    Nv = EToV.max()+1

    # Create face to node connectivity matrix
    TotalFaces = Nfaces*K

    # List of local face to local vertex connections
    vn = np.int32([[0,1],[1,2],[0,2]])

    # Build global face to node sparse array
    g_face_no = 0
    vert_indices_to_face_numbers = {}
    face_numbers = xrange(Nfaces)
    for k in xrange(K):
        for face in face_numbers:
            vert_indices_to_face_numbers.setdefault(
                    frozenset(EToV[k,vn[face]]), []).append(g_face_no)
            g_face_no += 1

    faces1 = []
    faces2 = []

    for i in vert_indices_to_face_numbers.itervalues():
        if len(i) == 2:
            faces1.append(i[0])
            faces2.append(i[1])
            faces2.append(i[0])
            faces1.append(i[1])

    faces1 = np.intp(faces1)
    faces2 = np.intp(faces2)

    # Convert faceglobal number to element and face numbers
    element1, face1 = divmod(faces1, Nfaces)
    element2, face2 = divmod(faces2, Nfaces)

    # Rearrange into Nelements x Nfaces sized arrays
    size = np.array([K, Nfaces])
    ind = sub2ind([K, Nfaces], element1, face1)

    EToE = np.outer(np.arange(K), np.ones((1, Nfaces)))
    EToF = np.outer(np.ones((K,1)), np.arange(Nfaces))
    EToE = EToE.reshape(K*Nfaces)
    EToF = EToF.reshape(K*Nfaces)

    EToE[np.int32(ind)] = element2.copy()
    EToF[np.int32(ind)] = face2.copy()

    EToE = EToE.reshape(K, Nfaces)
    EToF = EToF.reshape(K, Nfaces)

    return  EToE, EToF

def sub2ind(size, I, J):
    """Return the linear index equivalent to the row and column subscripts 
    I and J for a matrix of size siz. siz is a vector with ndim(A) elements 
    (in this case, 2), where siz(1) is the number of rows and siz(2) is the 
    number of columns.
    """
    ind = I*size[1]+J
    return ind

def BuildMaps2D(ldis, Fmask, VX, VY, EToV, EToE, EToF, K, N, x, y):
    """Connectivity and boundary tables in the K # of Np elements
    Returns [mapM, mapP, vmapM, vmapP, vmapB, mapB].
    """

    l = ldis

    # number volume nodes consecutively
    temp    = np.arange(K*l.Np)
    nodeids = temp.reshape(l.Np, K, order='F').copy()

    vmapM   = np.zeros((l.Nfp, l.Nfaces, K))
    vmapP   = np.zeros((l.Nfp, l.Nfaces, K))
    mapM    = np.arange(np.int32(K)*l.Nfp*l.Nfaces)
    mapP    = mapM.reshape(l.Nfp, l.Nfaces, K).copy()
    # find index of face nodes with respect to volume node ordering
    for k1 in range(K):
        for f1 in range(l.Nfaces):
            vmapM[:, f1, k1] = nodeids[Fmask[:, f1], k1]

    # need to figure it out
    xtemp = x.reshape(K*l.Np,1, order='F').copy()
    ytemp = y.reshape(K*l.Np,1, order='F').copy()

    one = np.ones((1, l.Nfp))
    for k1 in range(K):
        for f1 in range(l.Nfaces):
            # find neighbor
            k2 = EToE[k1, f1]; f2 = EToF[k1, f1]

            # reference length of edge
            v1 = EToV[k1, f1]
            v2 = EToV[k1, 1+np.mod(f1, l.Nfaces-1)]

            refd = np.sqrt((VX[v1]-VX[v2])**2 \
                    + (VY[v1]-VY[v2])**2 )
            # find find volume node numbers of left and right nodes
            vidM = vmapM[:, f1, k1]; vidP = vmapM[:, f2, k2]
            x1 = xtemp[np.int32(vidM)]
            y1 = ytemp[np.int32(vidM)]
            x2 = xtemp[np.int32(vidP)]
            y2 = ytemp[np.int32(vidP)]
            x1 = np.dot(x1, one);  y1 = np.dot(y1, one)
            x2 = np.dot(x2, one);  y2 = np.dot(y2, one)
            # Compute distance matrix
            D = (x1 -x2.T)**2 + (y1-y2.T)**2
            # need to figure it out
            idM, idP = np.nonzero(np.sqrt(abs(D))<NODETOL*refd)
            vmapP[idM, f1, k1] = vidP[idP]
            mapP[idM, f1, k1] = idP + f2*l.Nfp+k2*l.Nfaces*l.Nfp

    # reshape vmapM and vmapP to be vectors and create boundary node list

    vmapP = vmapP.reshape(l.Nfp*l.Nfaces*K,1, order='F')
    vmapM = vmapM.reshape(l.Nfp*l.Nfaces*K,1, order='F')
    mapP  = mapP.reshape(l.Nfp*l.Nfaces*K,1, order='F')
    mapB  = np.array((vmapP==vmapM).nonzero())[0,:]
    mapB  = mapB.reshape(len(mapB),1)
    vmapB = vmapM[mapB].reshape(len(mapB),1)
    return np.int32(mapM), np.int32(mapP), np.int32(vmapM), np.int32(vmapP), np.int32(vmapB), np.int32(mapB)


def ind2sub(matr, row_size):
    """convert linear index to 2D index"""
    I = np.int32(np.mod(matr, row_size))
    J = np.int32((matr - I)/row_size)
    return I, J




# {{{ discretization data

class LocalDiscretization2D:
    def __init__(self, N):
        self.Np = (N+1)*(N+2)/2
        self.N = N
        self.Nfp = N+1
        self.Nfaces = 3

        # compute nodal set
        x, y = self.x, self.y = Nodes2D(N)
        r, s = self.r, self.s = xytors(self.x, self.y)

        # face masks
        fmask1   = (np.abs(s+1) < NODETOL).nonzero()[0];
        fmask2   = (np.abs(r+s) < NODETOL).nonzero()[0]
        fmask3   = (np.abs(r+1) < NODETOL).nonzero()[0]
        Fmask = self.Fmask = np.vstack((fmask1, fmask2, fmask3)).T
        FmaskF = self.FmaskF = Fmask.T.flatten()

        self.Fx = x[FmaskF[:], :]
        self.Fy = y[FmaskF[:], :]

        # Build reference element matrices
        V = self.V  = Vandermonde2D(N, r, s)
        invV = la.inv(self.V)
        MassMatrix = invV.T*invV
        self.Dr, self.Ds = Dmatrices2D(N, r, s, self.V)

        self.LIFT = Lift2D(self, r, s, V, Fmask)

        # weak operators
        Vr, Vs = GradVandermonde2D(N, r, s)
        invVV = la.inv(np.dot(V, V.T))
        self.Drw = np.dot(np.dot(V, Vr.T), invVV);
        self.Dsw = np.dot(np.dot(V, Vs.T), invVV)

    def gen_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element."""

        node_tuples = [
                (i,j) 
                for i in range(self.N+1)
                for j in range(self.N+1-i)
                ]

        node_dict = dict(
                (ituple, idx)
                for idx, ituple in enumerate(node_tuples))

        for i, j in node_tuples:
            if i + j < self.N:
                yield (node_dict[i, j], node_dict[i + 1, j],
                            node_dict[i, j+1])
            if i + j < self.N-1:
                yield (node_dict[i + 1, j+1], node_dict[i, j + 1],
                        node_dict[i + 1, j])





class Discretization2D:
    def __init__(self, ldis, Nv, VX, VY, K, EToV):
        l = self.ldis = ldis

        self.Nv = Nv
        self.VX   = VX
        self.K  = K

        va = np.int32(EToV[:, 0].T)
        vb = np.int32(EToV[:, 1].T)
        vc = np.int32(EToV[:, 2].T)

        x = self.x = 0.5*(
                -np.outer(l.r+l.s, VX[va])
                +np.outer(1+l.r, VX[vb])
                +np.outer(1+l.s, VX[vc]))
        y = self.y = 0.5*(
                -np.outer(l.r+l.s, VY[va])
                +np.outer(1+l.r, VY[vb])
                +np.outer(1+l.s, VY[vc]))

        self.rx, self.sx, self.ry, self.sy, self.J = GeometricFactors2D(x, y, l.Dr, l.Ds)
        self.nx, self.ny, sJ = Normals2D(l, x, y, K)
        self.Fscale = sJ/self.J[l.FmaskF,:]
        self.EToE, self.EToF = Connect2D(EToV)

        self.mapM, self.mapP, self.vmapM, self.vmapP, self.vmapB, self.mapB = \
                BuildMaps2D(l, l.Fmask, VX, VY, EToV, self.EToE, self.EToF, K, l.N, x, y)

    def grad(self, u):
        """Compute 2D gradient field of scalar u."""
        l = self.ldis

        ur = np.dot(l.Dr, u)
        us = np.dot(l.Ds, u)
        ux = self.rx*ur + self.sx*us
        uy = self.ry*ur + self.sy*us
        return ux, uy

    def curl(self, ux, uy, uz):
        """Compute 2D curl-operator in (x, y) plane."""
        l = self.ldis
        d = self

        uxr = np.dot(l.Dr, ux)
        uxs = np.dot(l.Ds, ux)
        uyr = np.dot(l.Dr, uy)
        uys = np.dot(l.Ds, uy)
        vz =  d.rx*uyr + d.sx*uys - d.ry*uxr - d.sy*uxs
        vx = 0; vy = 0

        if uz != 0:
            uzr = np.dot(l.Dr, uz)
            uzs = np.dot(l.Ds, uz)
            vx =  d.ry*uzr + d.sy*uzs
            vy = -d.rx*uzr - d.sx*uzs

        return vx, vy, vz

    def dt_scale(self):
        """Compute inscribed circle diameter as characteristic for 
        grid to choose timestep
        """

        r = self.ldis.r
        s = self.ldis.s

        # Find vertex nodes
        vmask1   = (abs(s+r+2) < NODETOL).nonzero()[0]
        vmask2   = (abs(r-1)   < NODETOL).nonzero()[0]
        vmask3   = (abs(s-1)   < NODETOL).nonzero()[0]
        vmask    = np.vstack((vmask1, vmask2, vmask3))
        vmask    = vmask.T
        vmaskF   = vmask.reshape(vmask.shape[0]*vmask.shape[1], order='F')

        vx = self.x[vmaskF[:], :]; vy = self.y[vmaskF[:], :]

        # Compute semi-perimeter and area
        len1 = np.sqrt((vx[0,:]-vx[1,:])**2\
                +(vy[0,:]-vy[1,:])**2)
        len2 = np.sqrt((vx[1,:]-vx[2,:])**2\
                +(vy[1,:]-vy[2,:])**2)
        len3 = np.sqrt((vx[2,:]-vx[0,:])**2\
                +(vy[2,:]-vy[0,:])**2)
        sper = (len1 + len2 + len3)/2.0
        area = np.sqrt(sper*(sper-len1)*(sper-len2)*(sper-len3))

        # Compute scale using radius of inscribed circle
        return area/sper

    def gen_vis_triangles(self):
        submesh_indices = np.array(list(self.ldis.gen_submesh_indices()))

        result = np.empty((self.K, submesh_indices.shape[0], submesh_indices.shape[1]),
                dtype=submesh_indices.dtype)

        Np = self.ldis.Np
        return (np.arange(0, self.K*Np, Np)[:,np.newaxis,np.newaxis]
                + submesh_indices[np.newaxis,:,:]).reshape(-1, submesh_indices.shape[1])





# {{{ Maxwell's equations

def Maxwell2D(d, Hx, Hy, Ez, final_time):
    """Integrate TM-mode Maxwell's until final_time starting 
    with initial conditions Hx, Hy, Ez.
    """
    l = d.ldis

    time = 0

    # Runge-Kutta residual storage
    resHx = np.zeros_like(Hx)
    resHy = np.zeros_like(Hx)
    resEz = np.zeros_like(Hx)

    # compute time step size
    rLGL = JacobiGQ(0,0, l.N)[0]
    rmin = abs(rLGL[0]-rLGL[1])
    dt_scale = d.dt_scale()
    dt = dt_scale.min()*rmin*2/3

    if do_vis:
        vis_mesh = mv.triangular_mesh(d.x.T.flatten(), d.y.T.flatten(), Ez.T.flatten(),
                d.gen_vis_triangles())

    # outer time step loop
    while time < final_time:

        if time+dt>final_time:
            dt = final_time-time

        for a, b in zip(rk4a, rk4b):
            # compute right hand side of TM-mode Maxwell's equations
            rhsHx, rhsHy, rhsEz = MaxwellRHS2D(d, Hx, Hy, Ez)

            # initiate and increment Runge-Kutta residuals
            resHx = a*resHx + dt*rhsHx
            resHy = a*resHy + dt*rhsHy
            resEz = a*resEz + dt*rhsEz

            # update fields
            Hx = Hx+b*resHx
            Hy = Hy+b*resHy
            Ez = Ez+b*resEz

        # Increment time
        time = time+dt

        if do_vis:
            vis_mesh.mlab_source.z = Ez.T.flatten()

    return Hx, Hy, Ez, time



def MaxwellRHS2D(discr, Hx, Hy, Ez):
    """Evaluate RHS flux in 2D Maxwell TM form."""

    d = discr
    l = discr.ldis

    # Define field differences at faces
    vmapM = d.vmapM.reshape(l.Nfp*l.Nfaces, d.K, order='F')
    vmapP = d.vmapP.reshape(l.Nfp*l.Nfaces, d.K, order='F')
    Im, Jm = ind2sub(vmapM, l.Np)
    Ip, Jp = ind2sub(vmapP, l.Np)

    flux_shape = (l.Nfp*l.Nfaces, d.K)
    dHx = np.zeros(flux_shape)
    dHx = Hx[Im, Jm]-Hx[Ip, Jp]
    dHy = np.zeros(flux_shape)
    dHy = Hy[Im, Jm]-Hy[Ip, Jp]
    dEz = np.zeros(flux_shape)
    dEz = Ez[Im, Jm]-Ez[Ip, Jp]

    # Impose reflective boundary conditions (Ez+ = -Ez-)
    size_H = l.Nfp*l.Nfaces
    I, J = ind2sub(d.mapB, size_H)
    Iz, Jz = ind2sub(d.vmapB, l.Np)

    dHx[I, J] = 0
    dHy[I, J] = 0
    dEz[I, J] = 2*Ez[Iz, Jz]

    # evaluate upwind fluxes
    alpha  = 1.0
    ndotdH =  d.nx*dHx + d.ny*dHy
    fluxHx =  d.ny*dEz + alpha*(ndotdH*d.nx-dHx)
    fluxHy = -d.nx*dEz + alpha*(ndotdH*d.ny-dHy)
    fluxEz = -d.nx*dHy + d.ny*dHx - alpha*dEz

    # local derivatives of fields
    Ezx, Ezy = d.grad(Ez)
    CuHx, CuHy, CuHz = d.curl(Hx, Hy,0)

    # compute right hand sides of the PDE's
    rhsHx = -Ezy  + np.dot(l.LIFT, d.Fscale*fluxHx)/2.0
    rhsHy =  Ezx  + np.dot(l.LIFT, d.Fscale*fluxHy)/2.0
    rhsEz =  CuHz + np.dot(l.LIFT, d.Fscale*fluxEz)/2.0
    return rhsHx, rhsHy, rhsEz

# }}}




# {{{ test

def test():
    d = Discretization2D(LocalDiscretization2D(5),
            *MeshReaderGambit2D('Maxwell025.neu'))

    # set initial conditions
    mmode = 1; nmode = 1
    Ez = np.sin(mmode*np.pi*d.x)*np.sin(nmode*np.pi*d.y)
    Hx = np.zeros((d.ldis.Np, d.K))
    Hy = np.zeros((d.ldis.Np, d.K))

    final_time = 5
    Hx, Hy, Ez, time = Maxwell2D(d, Hx, Hy, Ez, final_time)

if __name__ == "__main__":
    test()

# }}}

# vim: foldmethod=marker
