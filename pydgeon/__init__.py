# Pydgeon - the Python DG Environment
# (C) 2009, 2010 Tim Warburton, Jan Hesthaven, Xueyu Zhu, Andreas Kloeckner
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

import numpy as np
from pydgeon.tools import eldot, NODETOL


# {{{ mesh reading

def read_2d_gambit_mesh(file_name):
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
        Nv = int(dims[0])
        K = int(dims[1])

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


def read_3d_gambit_mesh(file_name):
    """Read in basic grid information to build grid
    Note: Gambit(Fluent, Inc) *.neu format is assumed.

    Returns (Nv, VX, VY, VZ, K, EToV).
    """

    with open(file_name, 'r') as inf:
        # read after intro
        for i in range(6):
            line = inf.readline()

        # Find number of nodes and number of elements
        dims = inf.readline().split()
        Nv = int(dims[0])
        K = int(dims[1])

        for i in range(2):
            line = inf.readline()

        # read node coordinates
        VX = np.zeros(Nv);
        VY = np.zeros(Nv)
        VZ = np.zeros(Nv)
        for i  in range(Nv):
            tmpx = inf.readline().split()
            VX[i] = float(tmpx[1]);
            VY[i] = float(tmpx[2]);
            VZ[i] = float(tmpx[3])

        for i in range(2):
            line = inf.readline()

        # read element to node connectivity (check this)
        EToV = np.zeros((K, 4))
        for k in range(K):
            tmpcon= inf.readline().split()
            EToV[k,0] = np.int32(tmpcon[3])-1
            EToV[k,1] = np.int32(tmpcon[4])-1
            EToV[k,2] = np.int32(tmpcon[5])-1
            EToV[k,3] = np.int32(tmpcon[6])-1

        return Nv, VX, VY, VZ, K, EToV


# }}}

# {{{ mesh assembly

def GeometricFactors2D(x, y, Dr, Ds):
    """Compute the metric elements for the local mappings of the elements
    Returns [rx, sx, ry, sy, J].
    """
    # Calculate geometric factors
    xr = eldot(Dr, x)
    xs = eldot(Ds, x)
    yr = eldot(Dr, y)
    ys = eldot(Ds, y)
    J = -xs*yr + xr*ys
    rx = ys/J
    sx =-yr/J
    ry =-xs/J
    sy = xr/J
    return rx, sx, ry, sy, J

def GeometricFactors3D(x, y, z, Dr, Ds, Dt):
    """Compute the metric elements for the local mappings of the elements
    Returns [rx, sx, tx, ry, sy, ty, rz, sz, tz, J].
    """
    # Calculate geometric factors
    xr = eldot(Dr, x)
    xs = eldot(Ds, x)
    xt = eldot(Dt, x)

    yr = eldot(Dr, y)
    ys = eldot(Ds, y)
    yt = eldot(Dt, y)

    zr = eldot(Dr, z)
    zs = eldot(Ds, z)
    zt = eldot(Dt, z)

    J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt)
    rx  =  (ys*zt - zs*yt)/J
    ry  = -(xs*zt - zs*xt)/J
    rz  =  (xs*yt - ys*xt)/J

    sx  = -(yr*zt - zr*yt)/J
    sy  =  (xr*zt - zr*xt)/J
    sz  = -(xr*yt - yr*xt)/J

    tx  =  (yr*zs - zr*ys)/J
    ty  = -(xr*zs - zr*xs)/J
    tz  =  (xr*ys - yr*xs)/J

    return rx, sx, tx, ry, sy, ty, rz, sz, tz, J


def Normals2D(ldis, x, y, K):
    """Compute outward pointing normals at elements faces
    and surface Jacobians.
    """

    l = ldis
    xr = eldot(l.Dr, x)
    yr = eldot(l.Dr, y)
    xs = eldot(l.Ds, x)
    ys = eldot(l.Ds, y)
    J = xr*ys-xs*yr

    # interpolate geometric factors to face nodes
    fxr = xr[:, l.FmaskF]; fxs = xs[:, l.FmaskF]
    fyr = yr[:, l.FmaskF]; fys = ys[:, l.FmaskF]

    # build normals
    nx = np.zeros((K, l.Nafp))
    ny = np.zeros((K, l.Nafp))
    fid1 = np.arange(l.Nfp).reshape(1, l.Nfp)
    fid2 = fid1+l.Nfp
    fid3 = fid2+l.Nfp

    # face 1
    nx[:, fid1] =  fyr[:, fid1]
    ny[:, fid1] = -fxr[:, fid1]

    # face 2
    nx[:, fid2] =  fys[:, fid2]-fyr[:, fid2]
    ny[:, fid2] = -fxs[:, fid2]+fxr[:, fid2]

    # face 3
    nx[:, fid3] = -fys[:, fid3]
    ny[:, fid3] =  fxs[:, fid3]

    # normalise
    sJ = np.sqrt(nx*nx+ny*ny)
    nx = nx/sJ
    ny = ny/sJ
    return nx, ny, sJ


def Normals3D(ldis, x, y, z, K):
    """Compute outward pointing normals at elements faces
    and surface Jacobians.
    """

    l = ldis
    rx,sx,tx,ry,sy,ty,rz,sz,tz,J = GeometricFactors3D(x,y,z,l.Dr,l.Ds,l.Dt)

    # interpolate geometric factors to face nodes
    frx = rx[:, l.FmaskF]; fsx = sx[:, l.FmaskF]; ftx = tx[:, l.FmaskF];
    fry = ry[:, l.FmaskF]; fsy = sy[:, l.FmaskF]; fty = ty[:, l.FmaskF];
    frz = rz[:, l.FmaskF]; fsz = sz[:, l.FmaskF]; ftz = tz[:, l.FmaskF];

    # build normals
    nx = np.zeros((K, l.Nafp))
    ny = np.zeros((K, l.Nafp))
    nz = np.zeros((K, l.Nafp))
    fid1 = np.arange(l.Nfp).reshape(1, l.Nfp)
    fid2 = fid1+l.Nfp
    fid3 = fid2+l.Nfp
    fid4 = fid3+l.Nfp

    # face 1
    nx[:, fid1] = -ftx[:, fid1]
    ny[:, fid1] = -fty[:, fid1]
    nz[:, fid1] = -ftz[:, fid1]

    # face 2
    nx[:, fid2] = -fsx[:, fid2]
    ny[:, fid2] = -fsy[:, fid2]
    nz[:, fid2] = -fsz[:, fid2]

    # face 3
    nx[:, fid3] = frx[:, fid3]+fsx[:, fid3]+ftx[:, fid3]
    ny[:, fid3] = fry[:, fid3]+fsy[:, fid3]+fty[:, fid3]
    nz[:, fid3] = frz[:, fid3]+fsz[:, fid3]+ftz[:, fid3]

    # face 4
    nx[:, fid4] = -frx[:, fid4]
    ny[:, fid4] = -fry[:, fid4]
    nz[:, fid4] = -frz[:, fid4]

    # normalise
    sJ = np.sqrt(nx*nx+ny*ny+nz*nz)
    nx = nx/sJ
    ny = ny/sJ
    nz = nz/sJ
    sJ = sJ*J[:, l.FmaskF]

    return nx, ny, nz, sJ


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

    # Build global face to node connectivity
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
    ind = element1*Nfaces + face1

    EToE = np.outer(np.arange(K), np.ones((1, Nfaces)))
    EToF = np.outer(np.ones((K,1)), np.arange(Nfaces))
    EToE = EToE.reshape(K*Nfaces)
    EToF = EToF.reshape(K*Nfaces)

    EToE[np.int32(ind)] = element2
    EToF[np.int32(ind)] = face2

    EToE = EToE.reshape(K, Nfaces)
    EToF = EToF.reshape(K, Nfaces)

    return  EToE, EToF

def Connect3D(EToV):
    """Build global connectivity arrays for grid based on
    standard EToV input array from grid generator.
    """

    EToV = EToV.astype(np.intp)
    Nfaces = 4
    # Find number of elements and vertices
    K = EToV.shape[0]
    Nv = EToV.max()+1

    # Create face to node connectivity matrix
    TotalFaces = Nfaces*K

    # List of local face to local vertex connections
    vn = np.int32([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])

    # Build global face to node connectivity
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

    # check this
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
    ind = element1*Nfaces + face1

    EToE = np.outer(np.arange(K), np.ones((1, Nfaces)))
    EToF = np.outer(np.ones((K,1)), np.arange(Nfaces))
    EToE = EToE.reshape(K*Nfaces)
    EToF = EToF.reshape(K*Nfaces)

    EToE[np.int32(ind)] = element2
    EToF[np.int32(ind)] = face2

    EToE = EToE.reshape(K, Nfaces)
    EToF = EToF.reshape(K, Nfaces)

    return  EToE, EToF



def BuildMaps2D(ldis, Fmask, VX, VY, EToV, EToE, EToF, K, N, x, y):
    """Connectivity and boundary tables in the K # of Np elements
    Returns [mapM, mapP, vmapM, vmapP, vmapB, mapB].
    """

    l = ldis

    # number volume nodes consecutively
    nodeids = np.arange(K*l.Np).reshape(K, l.Np)

    vmapM = nodeids[:, Fmask]
    vmapP = np.zeros((K, l.Nfaces, l.Nfp), dtype=np.intp)
    mapP = np.empty((K, l.Nfaces, l.Nfp), dtype=np.intp)
    mapM = np.arange(mapP.size, dtype=np.intp).reshape(mapP.shape)

    # find index of face nodes with respect to volume node ordering
    xtemp = x.reshape(K*l.Np, 1)
    ytemp = y.reshape(K*l.Np, 1)

    one = np.ones((1, l.Nfp))
    for k1 in range(K):
        for f1 in range(l.Nfaces):
            # find neighbor
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]

            # reference length of edge
            v1 = EToV[k1, f1]
            v2 = EToV[k1, 1+np.mod(f1, l.Nfaces-1)]

            # find find volume node numbers of left and right nodes
            vidM = vmapM[k1, f1, :]
            vidP = vmapM[k2, f2, :]

            x1 = xtemp[vidM]
            y1 = ytemp[vidM]
            x2 = xtemp[vidP]
            y2 = ytemp[vidP]
            x1 = np.dot(x1, one);  y1 = np.dot(y1, one)
            x2 = np.dot(x2, one);  y2 = np.dot(y2, one)

            # Compute distance matrix
            D = (x1-x2.T)**2 + (y1-y2.T)**2

            ref_distance = np.sqrt(
                    (VX[v1]-VX[v2])**2 + (VY[v1]-VY[v2])**2 )

            idM, idP = np.nonzero(np.sqrt(abs(D))<NODETOL*ref_distance)
            vmapP[k1, f1, idM] = vidP[idP]
            mapP[k1, f1, idM] = idP + f2*l.Nfp+k2*l.Nfaces*l.Nfp

    # create boundary node list
    mapB  = np.array((vmapP.ravel()==vmapM.ravel()).nonzero()[0])
    vmapB = vmapM.ravel()[mapB]

    return mapM, mapP, vmapM, vmapP, vmapB, mapB


def BuildMaps3D(ldis, Fmask, VX, VY, VZ, EToV, EToE, EToF, K, N, x, y, z):
    """Connectivity and boundary tables in the K # of Np elements
    Returns [mapM, mapP, vmapM, vmapP, vmapB, mapB].
    """

    l = ldis

    # number volume nodes consecutively
    nodeids = np.arange(K*l.Np).reshape(K, l.Np)

    vmapM = nodeids[:, Fmask]
    vmapP = np.zeros((K, l.Nfaces, l.Nfp), dtype=np.intp)
    mapP = np.empty((K, l.Nfaces, l.Nfp), dtype=np.intp)
    mapM = np.arange(mapP.size, dtype=np.intp).reshape(mapP.shape)

    # find index of face nodes with respect to volume node ordering
    xtemp = x.reshape(K*l.Np, 1)
    ytemp = y.reshape(K*l.Np, 1)
    ztemp = z.reshape(K*l.Np, 1)

    one = np.ones((1, l.Nfp))
    for k1 in range(K):
        for f1 in range(l.Nfaces):
            # find neighbor
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]

            # reference length of edge
            v1 = EToV[k1, f1]
            v2 = EToV[k1, 1+np.mod(f1, l.Nfaces-1)]

            # find find volume node numbers of left and right nodes
            vidM = vmapM[k1, f1, :]
            vidP = vmapM[k2, f2, :]

            x1 = xtemp[vidM]
            y1 = ytemp[vidM]
            z1 = ztemp[vidM]
            x2 = xtemp[vidP]
            y2 = ytemp[vidP]
            z2 = ztemp[vidP]
            x1 = np.dot(x1, one);  y1 = np.dot(y1, one);  z1 = np.dot(z1, one)
            x2 = np.dot(x2, one);  y2 = np.dot(y2, one);  z2 = np.dot(z2, one)

            # Compute distance matrix
            D = (x1-x2.T)**2 + (y1-y2.T)**2 + (z1-z2.T)**2

            ref_distance = np.sqrt(
                    (VX[v1]-VX[v2])**2 + (VY[v1]-VY[v2])**2 + (VZ[v1]-VZ[v2])**2 )

            idM, idP = np.nonzero(np.sqrt(abs(D))<NODETOL*ref_distance)
            vmapP[k1, f1, idM] = vidP[idP]
            mapP[k1, f1, idM] = idP + f2*l.Nfp+k2*l.Nfaces*l.Nfp

    # create boundary node list
    mapB  = np.array((vmapP.ravel()==vmapM.ravel()).nonzero()[0])
    vmapB = vmapM.ravel()[mapB]

    return mapM, mapP, vmapM, vmapP, vmapB, mapB


# }}}

# {{{ discretization data

# {{{ 2D

class Discretization2D:
    def __init__(self, ldis, Nv, VX, VY, K, EToV):
        l = self.ldis = ldis

        self.dimensions = ldis.dimensions

        self.Nv = Nv
        self.VX   = VX
        self.K  = K

        va = np.intp(EToV[:, 0].T)
        vb = np.intp(EToV[:, 1].T)
        vc = np.intp(EToV[:, 2].T)

        x = self.x = 0.5*(
                -np.outer(VX[va], l.r+l.s, )
                +np.outer(VX[vb], 1+l.r)
                +np.outer(VX[vc], 1+l.s))
        y = self.y = 0.5*(
                -np.outer(VY[va], l.r+l.s)
                +np.outer(VY[vb], 1+l.r)
                +np.outer(VY[vc], 1+l.s))

        self.rx, self.sx, self.ry, self.sy, self.J = GeometricFactors2D(x, y, l.Dr, l.Ds)
        self.nx, self.ny, self.sJ = Normals2D(l, x, y, K)
        self.Fscale = self.sJ/self.J[:, l.FmaskF]

        # element-to-element, element-to-face connectivity
        self.EToE, self.EToF = Connect2D(EToV)

        self.mapM, self.mapP, self.vmapM, self.vmapP, self.vmapB, self.mapB = \
                BuildMaps2D(l, l.Fmask, VX, VY, EToV, self.EToE, self.EToF, K, l.N, x, y)

    def grad(self, u):
        """Compute 2D gradient field of scalar u."""
        l = self.ldis

        ur = eldot(l.Dr, u)
        us = eldot(l.Ds, u)
        ux = self.rx*ur + self.sx*us
        uy = self.ry*ur + self.sy*us
        return ux, uy

    def curl(self, ux, uy, uz):
        """Compute 2D curl-operator in (x, y) plane."""
        l = self.ldis
        d = self

        uxr = eldot(l.Dr, ux)
        uxs = eldot(l.Ds, ux)
        uyr = eldot(l.Dr, uy)
        uys = eldot(l.Ds, uy)
        vz =  d.rx*uyr + d.sx*uys - d.ry*uxr - d.sy*uxs
        vx = 0; vy = 0

        if uz != 0:
            uzr = eldot(l.Dr, uz)
            uzs = eldot(l.Ds, uz)
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

        vx = self.x[:, vmaskF[:]]; vy = self.y[:, vmaskF[:]]

        # Compute semi-perimeter and area
        len1 = np.sqrt((vx[:,0]-vx[:,1])**2\
                +(vy[:,0]-vy[:,1])**2)
        len2 = np.sqrt((vx[:,1]-vx[:,2])**2\
                +(vy[:,1]-vy[:,2])**2)
        len3 = np.sqrt((vx[:,2]-vx[:,0])**2\
                +(vy[:,2]-vy[:,0])**2)
        sper = (len1 + len2 + len3)/2.0
        area = np.sqrt(sper*(sper-len1)*(sper-len2)*(sper-len3))

        # Compute scale using radius of inscribed circle
        return area/sper

    def gen_vis_triangles(self):
        submesh_indices = np.array(list(self.ldis.gen_submesh_indices()))

        Np = self.ldis.Np
        return (np.arange(0, self.K*Np, Np)[:,np.newaxis,np.newaxis]
                + submesh_indices[np.newaxis,:,:]).reshape(-1, submesh_indices.shape[1])

# }}}

# {{{ 3D

class Discretization3D:
    def __init__(self, ldis, Nv, VX, VY, VZ, K, EToV):
        l = self.ldis = ldis

        self.dimensions = ldis.dimensions

        self.Nv = Nv
        self.VX = VX
        self.VY = VY; # check ?
        self.VZ = VZ
        self.K  = K

        va = np.intp(EToV[:, 0].T)
        vb = np.intp(EToV[:, 1].T)
        vc = np.intp(EToV[:, 2].T)
        vd = np.intp(EToV[:, 3].T)

        x = self.x = 0.5*(
                -np.outer(VX[va], 1+l.r+l.s+l.t, )
                +np.outer(VX[vb], 1+l.r)
                +np.outer(VX[vc], 1+l.s)
                +np.outer(VX[vd], 1+l.t))

        y = self.y = 0.5*(
                -np.outer(VY[va], 1+l.r+l.s+l.t, )
                +np.outer(VY[vb], 1+l.r)
                +np.outer(VY[vc], 1+l.s)
                +np.outer(VY[vd], 1+l.t))

        z = self.z = 0.5*(
                -np.outer(VZ[va], 1+l.r+l.s+l.t, )
                +np.outer(VZ[vb], 1+l.r)
                +np.outer(VZ[vc], 1+l.s)
                +np.outer(VZ[vd], 1+l.t))

        drst_dxyz = np.empty((3,3), dtype=object)
        geo_fac = GeometricFactors3D(x, y, z, l.Dr, l.Ds, l.Dt)
        self.J = geo_fac[-1]
        drst_dxyz.reshape(-1)[:] = geo_fac[:-1]

        self.drst_dxyz = drst_dxyz.T

        self.nx, self.ny, self.nz, self.sJ = Normals3D(l, x, y, z, K)
        self.Fscale = self.sJ/self.J[:, l.FmaskF]

        # element-to-element, element-to-face connectivity
        self.EToE, self.EToF = Connect3D(EToV)

        self.mapM, self.mapP, self.vmapM, self.vmapP, self.vmapB, self.mapB = \
                BuildMaps3D(l, l.Fmask, VX, VY, VZ, EToV, self.EToE, self.EToF, K, l.N, x, y, z)

        self.bc = np.ones((K, l.Nfp*l.Nfaces))
        self.bc.reshape(-1)[self.mapB] = -1

    @property
    def rx(self): return self.drst_dxyz[0,0]
    @property
    def sx(self): return self.drst_dxyz[1,0]
    @property
    def tx(self): return self.drst_dxyz[2,0]

    @property
    def ry(self): return self.drst_dxyz[0,1]
    @property
    def sy(self): return self.drst_dxyz[1,1]
    @property
    def ty(self): return self.drst_dxyz[2,1]

    @property
    def rz(self): return self.drst_dxyz[0,2]
    @property
    def sz(self): return self.drst_dxyz[1,2]
    @property
    def tz(self): return self.drst_dxyz[2,2]

    def gradient(self, u):
        """Compute 3D gradient field of scalar u."""
        l = self.ldis

        ur = eldot(l.Dr, u)
        us = eldot(l.Ds, u)
        ut = eldot(l.Dt, u)

        ux = self.rx*ur + self.sx*us + self.tx*ut
        uy = self.ry*ur + self.sy*us + self.ty*ut
        uz = self.rz*ur + self.sz*us + self.tz*ut

        return ux, uy, uz

    def divergence(self, ux, uy, uz):
        """Compute 3D curl-operator in (x, y, z) ."""
        l = self.ldis
        d = self

        duxdr = eldot(l.Dr, ux)
        duxds = eldot(l.Ds, ux)
        duxdt = eldot(l.Dt, ux)

        duydr = eldot(l.Dr, uy)
        duyds = eldot(l.Ds, uy)
        duydt = eldot(l.Dt, uy)

        duzdr = eldot(l.Dr, uz)
        duzds = eldot(l.Ds, uz)
        duzdt = eldot(l.Dt, uz)

        duxdx = d.rx*duxdr + d.sx*duxds + d.tx*duxdt
        duydy = d.ry*duydr + d.sy*duyds + d.ty*duydt
        duzdz = d.rz*duzdr + d.sz*duzds + d.tz*duzdt

        divU = duxdx+duydy+duzdz

        return divU

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

        vx = self.x[:, vmaskF[:]]; vy = self.y[:, vmaskF[:]]

        # Compute semi-perimeter and area
        len1 = np.sqrt((vx[:,0]-vx[:,1])**2\
                +(vy[:,0]-vy[:,1])**2)
        len2 = np.sqrt((vx[:,1]-vx[:,2])**2\
                +(vy[:,1]-vy[:,2])**2)
        len3 = np.sqrt((vx[:,2]-vx[:,0])**2\
                +(vy[:,2]-vy[:,0])**2)
        sper = (len1 + len2 + len3)/2.0
        area = np.sqrt(sper*(sper-len1)*(sper-len2)*(sper-len3))

        # Compute scale using radius of inscribed circle
        return area/sper


    def dt_scale3d(self):
        """Compute inscribed circle diameter as characteristic for
        grid to choose timestep
        """

        h = min(self.Fscale) # check: fix min

        # Compute scale using radius of inscribed circle
        return h

    def gen_vis_simplices(self):
        submesh_indices = np.array(list(self.ldis.gen_submesh_indices()))

        Np = self.ldis.Np
        return (np.arange(0, self.K*Np, Np)[:,np.newaxis,np.newaxis]
                + submesh_indices[np.newaxis,:,:]).reshape(-1, submesh_indices.shape[1])

class CLDiscretizationInfo3D(Discretization3D):
    def __init__(self, queue, discr, dtype, allocator):
        context = queue.context
        self.discr = discr
        import pyopencl as cl

        self.allocator = allocator

        dtype4 = cl.array.vec.types[np.dtype(dtype), 4]

        l = discr.ldis
        drdsdt_unvec = np.zeros((l.Np, l.Np, 4), dtype)
        for i, mat in enumerate([l.Dr, l.Ds, l.Dt]):
            drdsdt_unvec[:, :, i] = mat

        self.drdsdt = cl.array.to_device(
            queue,
            drdsdt_unvec
            .view(dtype=dtype4)[:, :, 0].copy(order="F"))
        self.drdsdt_img = cl.image_from_array(context, drdsdt_unvec.view(dtype=dtype4)[:, :, 0])

        drst_dx_unvec = np.zeros((discr.K, 4), dtype)
        drst_dy_unvec = np.zeros((discr.K, 4), dtype)
        drst_dz_unvec = np.zeros((discr.K, 4), dtype)

        for i in range(3):
            drst_dx_unvec[:, i] = discr.drst_dxyz[i, 0][:,0]
            drst_dy_unvec[:, i] = discr.drst_dxyz[i, 1][:,0]
            drst_dz_unvec[:, i] = discr.drst_dxyz[i, 2][:,0]

        self.drst_dx = cl.array.to_device(queue, drst_dx_unvec.view(dtype=dtype4)[:, 0])
        self.drst_dy = cl.array.to_device(queue, drst_dy_unvec.view(dtype=dtype4)[:, 0])
        self.drst_dz = cl.array.to_device(queue, drst_dz_unvec.view(dtype=dtype4)[:, 0])

        self.vmapP = cl.array.to_device(queue,
                discr.vmapP.astype(np.int32).copy().reshape(discr.K, -1))
        self.vmapM = cl.array.to_device(queue,
                discr.vmapM.astype(np.int32).copy().reshape(discr.K, -1))

        self.nx = cl.array.to_device(queue, discr.nx.astype(dtype))
        self.ny = cl.array.to_device(queue, discr.ny.astype(dtype))
        self.nz = cl.array.to_device(queue, discr.nz.astype(dtype))

        self.Fscale = cl.array.to_device(queue, discr.Fscale.astype(dtype))
        self.bc = cl.array.to_device(queue, discr.bc.astype(dtype))

        self.LIFT = cl.array.to_device(queue, l.LIFT.copy(order="F").astype(dtype))
        self.LIFT_img = cl.image_from_array(context, l.LIFT.astype(dtype))

        self.volume_events = []
        self.surface_events = []

# }}}

# }}}





# vim: foldmethod=marker
