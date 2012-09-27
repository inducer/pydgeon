# Pydgeon - the Python DG Environment
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

from math import sqrt
import numpy as np

from numpy import linalg as la
from pydgeon.tools import fact, NODETOL

from pytools import memoize_method




# {{{ helper routines

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
            aold = anew

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
        return np.array([(alpha-beta)/(alpha+beta+2)]), np.array([2])

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

def JacobiGL(alpha, beta, N):
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
    Np = (N+1)*(N+2)//2

    # Create equidistributed nodes on equilateral triangle
    L1 = np.zeros((Np,1))
    L2 = np.zeros((Np,1))
    L3 = np.zeros((Np,1))
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


def evalwarp(N, xnodes, xout):

    # Purpose: compute one-dimensional edge warping function

    warp = np.zeros((len(xout),1))
    xeq  = np.zeros((N+1,1))
    for i in range(N+1):
        xeq[i] = -1. + (2.*(N-i))/N;

    for i in range(N+1):
        d = xnodes[i]-xeq[i]

        for j in range(1,N):
            if(i!=j):
                d = d*(xout-xeq[j])/(xeq[i]-xeq[j]);

        if(i!=0):
            d = -d/(xeq[i]-xeq[0])

        if(i!=N):
            d = d/(xeq[i]-xeq[N])

        warp = warp+d;

    return warp


def evalshift(N, pval, L1, L2, L3):

    # Purpose: compute two-dimensional Warp & Blend transform

    # 1) compute Gauss-Lobatto-Legendre node distribution
    gaussX = -JacobiGL(0,0,N)

    # 3) compute blending function at each node for each edge
    blend1 = L2*L3
    blend2 = L1*L3
    blend3 = L1*L2

    # 4) amount of warp for each node, for each edge
    warpfactor1 = 4*evalwarp(N, gaussX, L3-L2)
    warpfactor2 = 4*evalwarp(N, gaussX, L1-L3)
    warpfactor3 = 4*evalwarp(N, gaussX, L2-L1)


    # 5) combine blend & warp
    warp1 = blend1*warpfactor1*(1 + (pval*L1)**2)
    warp2 = blend2*warpfactor2*(1 + (pval*L2)**2)
    warp3 = blend3*warpfactor3*(1 + (pval*L3)**2)

    # 6) evaluate shift in equilateral triangle
    dx = 1*warp1 + np.cos(2.*np.pi/3.)*warp2 + np.cos(4.*np.pi/3.)*warp3;
    dy = 0*warp1 + np.sin(2.*np.pi/3.)*warp2 + np.sin(4.*np.pi/3.)*warp3;

    return dx, dy


def  WarpShiftFace3D(p, pval, pval2, L1, L2, L3, L4):

    # Purpose: compute warp factor used in creating 3D Warp & Blend nodes

    dtan1,dtan2 = evalshift(p, pval, L2, L3, L4);

    warpx = dtan1
    warpy = dtan2

    return warpx, warpy

def EquiNodes3D(N):

    # Purpose: compute the equidistributed nodes on the
    #         reference tetrahedron

    # total number of nodes
    Np = (N+1)*(N+2)*(N+3)//6

    # 2) create equidistributed nodes on equilateral triangle
    X = np.zeros((Np,1))
    Y = np.zeros((Np,1))
    Z = np.zeros((Np,1))

    sk = 0
    for n in range(N+1):
        for m in range(N+1-n):
            for q in range(N+1-n-m):

                X[sk] = -1 + (q*2.)/N
                Y[sk] = -1 + (m*2.)/N
                Z[sk] = -1 + (n*2.)/N;

                sk = sk+1;

    return X, Y, Z

def Nodes3D(N):
    """Compute (x, y, z) nodes in equilateral tet for polynomial of degree N.
    """

    alpopt = np.array([0, 0, 0, 0.1002,  1.1332, 1.5608, 1.3413, 1.2577, 1.1603,\
                           1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655])

    if(N<=15):
        alpha = alpopt[N-1]
    else:
        alpha = 1.

    # total number of nodes and tolerance
    Np = (N+1)*(N+2)*(N+3)//6
    tol = 1e-8

    r,s,t = EquiNodes3D(N)

    L1 = (1.+t)/2
    L2 = (1.+s)/2
    L3 = -(1.+r+s+t)/2
    L4 =  (1+r)/2

    # set vertices of tetrahedron
    v1 = np.array([-1., -1./sqrt(3.), -1./sqrt(6.)]) # row array
    v2 = np.array([ 1., -1./sqrt(3.), -1./sqrt(6.)])
    v3 = np.array([ 0,   2./sqrt(3.), -1./sqrt(6.)])
    v4 = np.array([ 0,            0,  3./sqrt(6.)])

    # orthogonal axis tangents on faces 1-4
    t1 = np.zeros((4,3))
    t1[0,:] = v2-v1
    t1[1,:] = v2-v1
    t1[2,:] = v3-v2
    t1[3,:] = v3-v1

    t2 = np.zeros((4,3))
    t2[0,:] = v3-0.5*(v1+v2)
    t2[1,:] = v4-0.5*(v1+v2)
    t2[2,:] = v4-0.5*(v2+v3)
    t2[3,:] = v4-0.5*(v1+v3)

    for n in range(4):
        # normalize tangents
        norm_t1 = la.norm(t1[n,:])
        norm_t2 = la.norm(t2[n,:])
        t1[n,:] = t1[n,:]/norm_t1 # 2-norm np.array ?
        t2[n,:] = t2[n,:]/norm_t2

    # Warp and blend for each face (accumulated in shiftXYZ)
    XYZ = L3*v1+L4*v2+L2*v3+L1*v4  # form undeformed coordinates
    shift = np.zeros((Np,3))
    for face in range(4):
        if(face==0):
            La = L1; Lb = L2; Lc = L3; Ld = L4;  # check  syntax

        if(face==1):
            La = L2; Lb = L1; Lc = L3; Ld = L4;

        if(face==2):
            La = L3; Lb = L1; Lc = L4; Ld = L2;

        if(face==3):
            La = L4; Lb = L1; Lc = L3; Ld = L2;

        #  compute warp tangential to face
        warp1, warp2 = WarpShiftFace3D(N, alpha, alpha, La, Lb, Lc, Ld)

        # compute volume blending
        blend = Lb*Lc*Ld

        # modify linear blend
        denom = (Lb+0.5*La)*(Lc+0.5*La)*(Ld+0.5*La)
        ids = np.argwhere(denom>tol) # syntax
        ids = ids[:,0]

        blend[ids] = (1+(alpha*La[ids])**2)*blend[ids]/denom[ids]

        # compute warp & blend
        shift = shift + (blend*warp1)*t1[face,:]
        shift = shift + (blend*warp2)*t2[face,:]

        # fix face warp
        ids = np.argwhere((La<tol) *( (Lb>tol) + (Lc>tol) + (Ld>tol) < 3)) # syntax ??
        ids = ids[:,0]

        shift[ids,:] = warp1[ids]*t1[face,:] + warp2[ids]*t2[face,:]



    # shift nodes and extract individual coordinates
    XYZ = XYZ + shift
    x = XYZ[:,0]
    y = XYZ[:,1]
    z = XYZ[:,2]

    return x, y, z



def xytors(x, y):
    """From (x, y) in equilateral triangle to (r, s) coordinates in standard triangle."""

    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0

    r = -L2 + L3 - L1; s = -L2 - L3 + L1
    return r, s

def xyztorst(x, y, z):

    # TO BE CONVERTED

    v1 = np.array([-1,-1/sqrt(3), -1/sqrt(6)]) # sqrt ?
    v2 = np.array([ 1,-1/sqrt(3), -1/sqrt(6)])
    v3 = np.array([ 0, 2/sqrt(3), -1/sqrt(6)])
    v4 = np.array([ 0, 0/sqrt(3),  3/sqrt(6)])

    # back out right tet nodes
    rhs = np.zeros((3, len(x)))
    rhs[0,:] = x
    rhs[1,:] = y
    rhs[2,:] = z

    tmp = np.zeros((3, 1))
    tmp[:,0] =  0.5*(v2+v3+v4-v1)
    rhs = rhs - tmp*np.ones((1,len(x)))

    A = np.zeros((3,3))
    A[:,0] = 0.5*(v2-v1)
    A[:,1] = 0.5*(v3-v1)
    A[:,2] = 0.5*(v4-v1)

    RST = la.solve(A,rhs)

    r = RST[0,:] # need to transpose ?
    s = RST[1,:] # need to transpose ?
    t = RST[2,:] # need to transpose ?

    return r, s, t


def rstoab(r, s):
    """Transfer from (r, s) -> (a, b) coordinates in triangle.
    """

    Np = len(r)
    a = np.zeros((Np,1))
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n])-1
        else:
            a[n] = -1

    b = s
    return a, b


def rsttoabc(r,s,t):

    """Transfer from (r,s,t) -> (a,b,c) coordinates in triangle
    """

    Np = len(r)
    tol = 1e-10

    a = np.zeros((Np,1))
    b = np.zeros((Np,1))
    c = np.zeros((Np,1))
    for n in range(Np):
        if abs(s[n]+t[n])>tol:
            a[n] = 2*(1+r[n])/(-s[n]-t[n])-1
        else:
            a[n] = -1

        if abs(t[n]-1.)>tol:
            b[n] = 2*(1+s[n])/(1-t[n])-1
        else:
            b[n] = -1

        c[n] = t[n]

    return a, b, c


def Simplex2DP(a, b, i, j):
    """Evaluate 2D orthonormal polynomial
    on simplex at (a, b) of order (i, j).
    """

    h1 = JacobiP(a,0,0, i).reshape(len(a),1)
    h2 = JacobiP(b,2*i+1,0, j).reshape(len(a),1)
    P  = np.sqrt(2.0)*h1*h2*(1-b)**i
    return P[:,0]


def Simplex3DP(a, b, c, i, j, k):
    """Evaluate 3D orthonormal polynomial
    on simplex at (a, b, c) of order (i, j, k).
    """

    h1 = JacobiP(a,0,0, i).reshape(len(a),1)
    h2 = JacobiP(b,2*i+1,0, j).reshape(len(a),1)
    h3 = JacobiP(c,2*(i+j)+2,0, k).reshape(len(a),1)
    P  = 2.0*np.sqrt(2.0)*h1*h2*((1-b)**i)*h3*((1-c)**(i+j))
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

def Vandermonde3D(N, r, s, t):
    """Initialize the 3D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i, t_i)
    """

    print 'Np computed as ', ((N+1)*(N+2)*(N+3))//6

    V3D = np.zeros((len(r),((N+1)*(N+2)*(N+3))//6))

    # Transfer to (a, b) coordinates
    a, b, c = rsttoabc(r, s, t)

    # build the Vandermonde matrix
    sk = 0

    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                V3D[:, sk] = Simplex3DP(a, b, c, i, j, k)
                sk = sk+1
    return V3D

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


def GradSimplex3DP(a, b, c, id, jd, kd):
    """Return the derivatives of the modal basis (id, jd, kd) on the
    3D simplex at (a, b, c).
    """

    fa  = JacobiP(a, 0, 0, id).reshape(len(a),1)
    dfa = GradJacobiP(a, 0, 0, id)
    gb  = JacobiP(b, 2*id+1,0, jd).reshape(len(b),1)
    dgb = GradJacobiP(b, 2*id+1,0, jd)
    hc  = JacobiP(c, 2*(id+jd)+2,0, kd).reshape(len(c),1)
    dhc = GradJacobiP(c, 2*(id+jd)+2,0, kd)

    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db + dc/dr d/dx
    dmodedr = dfa*gb*hc
    if(id>0):
        dmodedr = dmodedr*((0.5*(1-b))**(id-1))
    if(id+jd>0):
        dmodedr = dmodedr*((0.5*(1-c))**(id+jd-1))

    # s-derivative
    dmodeds = 0.5*(1+a)*dmodedr
    tmp = dgb*((0.5*(1-b))**id)
    if(id>0):
        tmp = tmp+(-0.5*id)*(gb*(0.5*(1-b))**(id-1))

    if(id+jd>0):
        tmp = tmp*((0.5*(1-c))**(id+jd-1))

    tmp = fa*tmp*hc
    dmodeds = dmodeds + tmp

    # t-derivative
    dmodedt = 0.5*(1+a)*dmodedr+0.5*(1+b)*tmp
    tmp = dhc*((0.5*(1-c))**(id+jd))
    if(id+jd>0):
        tmp = tmp-0.5*(id+jd)*(hc*((0.5*(1-c))**(id+jd-1)));

    tmp = fa*(gb*tmp)
    tmp = tmp*((0.5*(1-b))**id);
    dmodedt = dmodedt+tmp;

    # Normalize
    dmodedr = 2**(2*id+jd+1.5)*dmodedr
    dmodeds = 2**(2*id+jd+1.5)*dmodeds
    dmodedt = 2**(2*id+jd+1.5)*dmodedt

    return dmodedr[:,0], dmodeds[:,0], dmodedt[:,0]



def GradVandermonde2D(N, Np, r, s):
    """Initialize the gradient of the modal basis
    (i, j) at (r, s) at order N.
    """

    V2Dr = np.zeros((len(r), Np))
    V2Ds = np.zeros((len(r), Np))

    # find tensor-product coordinates
    a, b = rstoab(r, s)
    # Initialize matrices
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2Dr[:, sk], V2Ds[:, sk] = GradSimplex2DP(a, b, i, j)
            sk = sk+1
    return V2Dr, V2Ds


def GradVandermonde3D(N, Np, r, s, t):
    """Initialize the gradient of the modal basis
    (i, j, k) at (r, s, t) at order N.
    """

    V3Dr = np.zeros((len(r), Np))
    V3Ds = np.zeros((len(r), Np))
    V3Dt = np.zeros((len(r), Np))

    # find tensor-product coordinates
    a, b, c = rsttoabc(r, s, t)
    # Initialize matrices
    sk = 0
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                V3Dr[:, sk], V3Ds[:, sk], V3Dt[:, sk] = GradSimplex3DP(a, b, c, i, j, k)
                sk = sk+1

    return V3Dr, V3Ds, V3Dt


def Dmatrices2D(N, Np, r, s, V):
    """Initialize the (r, s) differentiation matriceon the simplex,
    evaluated at (r, s) at order N.
    """

    Vr, Vs = GradVandermonde2D(N, Np, r, s)
    invV   = la.inv(V)
    Dr     = np.dot(Vr, invV)
    Ds     = np.dot(Vs, invV)
    return Dr, Ds

def Dmatrices3D(N, Np, r, s, t, V):
    """Initialize the (r, s, t) differentiation matriceon the simplex,
    evaluated at (r, s, t) at order N.
    """

    Vr, Vs, Vt = GradVandermonde3D(N, Np, r, s, t)
    invV   = la.inv(V)

    print 'len(Vr)', len(Vr)

    Dr     = np.dot(Vr, invV)
    Ds     = np.dot(Vs, invV)
    Dt     = np.dot(Vt, invV)
    return Dr, Ds, Dt

def Lift2D(ldis, r, s, V, Fmask):
    """Compute surface to volume lift term for DG formulation."""
    l = ldis

    Emat = np.zeros((l.Np, l.Nfaces*l.Nfp))

    # face 1
    faceR = r[Fmask[0,:]]
    V1D = Vandermonde1D(l.N, faceR)
    massEdge1 = la.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[0,:],0:l.Nfp] = massEdge1

    # face 2
    faceR = r[Fmask[1,:]]
    V1D = Vandermonde1D(l.N, faceR)
    massEdge2 = la.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[1,:], l.Nfp:2*l.Nfp] = massEdge2

    # face 3
    faceS = s[Fmask[2,:]]
    V1D = Vandermonde1D(l.N, faceS)
    massEdge3 = la.inv(np.dot(V1D, V1D.T))
    Emat[Fmask[2,:],2*l.Nfp:3*l.Nfp] = massEdge3

    # inv(mass matrix)*\I_n (L_i, L_j)_{edge_n}
    LIFT = np.dot(V, np.dot(V.T, Emat))
    return LIFT

# }}}


def Lift3D(ldis, r, s, t, V, Fmask):
    """Compute surface to volume lift term for DG formulation."""
    l = ldis

    Emat = np.zeros((l.Np, l.Nfaces*l.Nfp))

    # face 1
    faceR = r[Fmask[0,:]]
    faceS = s[Fmask[0,:]]
    V2D = Vandermonde2D(l.N, faceR, faceS)
    massFace1 = la.inv(np.dot(V2D, V2D.T))
    Emat[Fmask[0,:],0:l.Nfp] = massFace1

    # face 2
    faceR = r[Fmask[1,:]]
    faceS = t[Fmask[1,:]]
    V1D = Vandermonde2D(l.N, faceR, faceS)
    massFace2 = la.inv(np.dot(V2D, V2D.T))
    Emat[Fmask[1,:], l.Nfp:2*l.Nfp] = massFace2

    # face 3
    faceR = s[Fmask[2,:]]
    faceS = t[Fmask[2,:]]
    V2D = Vandermonde2D(l.N, faceR, faceS)
    massFace3 = la.inv(np.dot(V2D, V2D.T))
    Emat[Fmask[2,:],2*l.Nfp:3*l.Nfp] = massFace3

    # face 4
    faceR = s[Fmask[3,:]]
    faceS = t[Fmask[3,:]]
    V2D = Vandermonde2D(l.N, faceR, faceS)
    massFace4 = la.inv(np.dot(V2D, V2D.T))
    Emat[Fmask[3,:],3*l.Nfp:4*l.Nfp] = massFace4

    # inv(mass matrix)*\I_n (L_i, L_j)_{edge_n}
    LIFT = np.dot(V, np.dot(V.T, Emat))
    return LIFT

# }}}




class LocalDiscretization2D:
    def __init__(self, N):
        self.dimensions = 2

        self.Np = (N+1)*(N+2)//2
        self.N = N
        self.Nfp = N+1
        self.Nfaces = 3
        self.Nafp = self.Nfp * self.Nfaces

        # compute nodal set
        x, y = self.x, self.y = Nodes2D(N)
        r, s = self.r, self.s = xytors(self.x, self.y)

        # face masks
        fmask1   = (np.abs(s+1) < NODETOL).nonzero()[0];
        fmask2   = (np.abs(r+s) < NODETOL).nonzero()[0]
        fmask3   = (np.abs(r+1) < NODETOL).nonzero()[0]
        Fmask = self.Fmask = np.vstack((fmask1, fmask2, fmask3))
        FmaskF = self.FmaskF = Fmask.flatten()

        self.Fx = x[FmaskF[:], :]
        self.Fy = y[FmaskF[:], :]

        # Build reference element matrices
        V = self.V  = Vandermonde2D(N, r, s)
        invV = la.inv(self.V)
        MassMatrix = invV.T*invV
        self.Dr, self.Ds = Dmatrices2D(N, self.Np, r, s, self.V)

        self.LIFT = Lift2D(self, r, s, V, Fmask)

        # weak operators
        Vr, Vs = GradVandermonde2D(N, self.Np, r, s)
        invVV = la.inv(np.dot(V, V.T))
        self.Drw = np.dot(np.dot(V, Vr.T), invVV);
        self.Dsw = np.dot(np.dot(V, Vs.T), invVV)

    def get_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element.
        """

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


class LocalDiscretization3D:
    def __init__(self, N):
        self.dimensions = 3

        self.Np = (N+1)*(N+2)*(N+3)//6
        self.N = N
        self.Nfp = (N+1)*(N+2)//2
        self.Nfaces = 4
        self.Nafp = self.Nfp * self.Nfaces


        # compute nodal set
        x, y, z = self.x, self.y, self.z = Nodes3D(N)
        r, s, t = self.r, self.s, self.t = xyztorst(self.x, self.y, self.z)

        # face masks
        fmask1   = (np.abs(t+1) < NODETOL).nonzero()[0];
        fmask2   = (np.abs(s+1) < NODETOL).nonzero()[0]
        fmask3   = (np.abs(r+s+t+1) < NODETOL).nonzero()[0]
        fmask4   = (np.abs(r+1) < NODETOL).nonzero()[0]
        Fmask = self.Fmask = np.vstack((fmask1, fmask2, fmask3, fmask4))
        FmaskF = self.FmaskF = Fmask.flatten()

        # Build reference element matrices
        V = self.V  = Vandermonde3D(N, r, s, t)
        invV = la.inv(self.V)
        MassMatrix = invV.T*invV
        self.Dr, self.Ds, self.Dt = Dmatrices3D(N, self.Np, r, s, t, self.V)

        self.LIFT = Lift3D(self, r, s, t, V, Fmask)

        # weak operators
        Vr, Vs, Vt = GradVandermonde3D(N, self.Np, r, s, t)
        invVV = la.inv(np.dot(V, V.T))
        self.Drw = np.dot(np.dot(V, Vr.T), invVV);
        self.Dsw = np.dot(np.dot(V, Vs.T), invVV)
        self.Dtw = np.dot(np.dot(V, Vt.T), invVV)

    @memoize_method
    def node_tuples(self):
        """Generate tuples enumerating the node indices present
        in this element. Each tuple has a length equal to the dimension
        of the element. The tuples constituents are non-negative integers
        whose sum is less than or equal to the order of the element.

        The order in which these nodes are generated dictates the local
        node numbering.
        """
        from pytools import \
                generate_nonnegative_integer_tuples_summing_to_at_most
        return list(
                generate_nonnegative_integer_tuples_summing_to_at_most(
                    self.N, self.dimensions))

    @memoize_method
    def get_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element."""

        node_dict = dict(
                (ituple, idx)
                for idx, ituple in enumerate(self.node_tuples()))

        def add_tuples(a, b):
            return tuple(ac+bc for ac, bc in zip(a, b))

        def try_add_tet(d1, d2, d3, d4):
            try:
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    node_dict[add_tuples(current, d3)],
                    node_dict[add_tuples(current, d4)],
                    ))
            except KeyError:
                pass

        result = []
        for current in self.node_tuples():
            # this is a tesselation of a cube into six tets.
            # subtets that fall outside of the master tet are simply not added.

            # positively oriented
            try_add_tet((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
            try_add_tet((1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0))
            try_add_tet((1, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 1))

            try_add_tet((1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0))
            try_add_tet((0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1))
            try_add_tet((0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 1, 0))

        return result

    # }}}

# vim: foldmethod=marker
