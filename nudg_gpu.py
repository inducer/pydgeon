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

import numpy as np
import numpy.linalg as la
from nudg import Discretization2D
import pyopencl as cl
import pyopencl.array as cl_array




class CLDiscretization2D(Discretization2D):
    def __init__(self, ldis, Nv, VX, VY, K, EToV):
        Discretization2D.__init__(self, ldis, Nv, VX, VY, K, EToV)

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self.block_size = 16*((ldis.Np+15)//16)

        self.prepare_gpu_data()

    def prepare_gpu_data(self):
        ldis = self.ldis

        drds_dev = np.empty((ldis.Np, self.block_size, 4), dtype=np.float32)
        drds_dev[:,:ldis.Np,0] = ldis.Dr.T
        drds_dev[:,:ldis.Np,1] = ldis.Ds.T
        self.diffmatrices_dev = cl_array.to_device(self.ctx, self.queue, drds_dev)

        drdx_dev = np.empty((self.K, self.dimensions**2), dtype=np.float32)
        drdx_dev[:,0] = self.rx[0]
        drdx_dev[:,1] = self.ry[0]
        drdx_dev[:,2] = self.sx[0]
        drdx_dev[:,3] = self.sy[0]
        self.drdx_dev = cl_array.to_device(self.ctx, self.queue, drdx_dev)

    def to_dev(self, vec):
        dev_vec = np.empty(dtype=np.float32, order="F",
                shape=(self.block_size, self.K))
        dev_vec[:self.ldis.Np, :] = vec
        return cl_array.to_device(self.ctx, self.queue, dev_vec)

    def from_dev(self, vec):
        return vec.get()[:self.ldis.Np, :]

    def volume_empty(self):
        return cl_array.Array(
                self.ctx, queue=self.queue,
                shape=(self.block_size, self.K),
                dtype=np.float32, order="F")




# {{{ kernels

MAXWELL_VOLUME_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define BSIZE %(BSIZE)d

__kernel void MaxwellsVolume2d(int K,
                               read_only __global float *g_Hx,
                               read_only __global float *g_Hy,
                               read_only __global float *g_Ez,
                               __global float *g_rhsHx,
                               __global float *g_rhsHy,
                               __global float *g_rhsEz,
                               read_only __global float4 *g_DrDs,
                               read_only __global float *g_vgeo)
{
  /* LOCKED IN to using Np threads per block */
  const int n = get_local_id(0);
  const int k = get_group_id(0);

  /* "coalesced"  */
  int m = n+k*BSIZE;
  int id = n;

  __local float l_Hx[BSIZE];
  __local float l_Hy[BSIZE];
  __local float l_Ez[BSIZE];

  l_Hx[id] = g_Hx[m];
  l_Hy[id] = g_Hy[m];
  l_Ez[id] = g_Ez[m];

  barrier(CLK_LOCAL_MEM_FENCE);

  float dHxdr=0,dHxds=0;
  float dHydr=0,dHyds=0;
  float dEzdr=0,dEzds=0;

  float Q;
  for(m=0; m<p_Np; ++m)
  {
    float4 D = g_DrDs[(n+m*BSIZE)];

    id = m;
    Q = l_Hx[m]; dHxdr += D.x*Q; dHxds += D.y*Q;
    Q = l_Hy[m]; dHydr += D.x*Q; dHyds += D.y*Q;
    Q = l_Ez[m]; dEzdr += D.x*Q; dEzds += D.y*Q;
  }

  const float drdx = g_vgeo[0+4*k];
  const float drdy = g_vgeo[1+4*k];
  const float dsdx = g_vgeo[2+4*k];
  const float dsdy = g_vgeo[3+4*k];

  m = n+BSIZE*k;
  g_rhsHx[m] = -(drdy*dEzdr+dsdy*dEzds);
  g_rhsHy[m] =  (drdx*dEzdr+dsdx*dEzds);
  g_rhsEz[m] =  (drdx*dHydr+dsdx*dHyds - drdy*dHxdr-dsdy*dHxds);
}
"""

MAXWELL_SURFACE_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define p_Nfp %(Nfp)d
#define p_Nfaces %(Nfaces)d
#define p_Nafp (p_Nfaces*p_Nfp)
#define BSIZE %(BSIZE)d

__kernel void MaxwellsSurface2d(int K,
                              read_only __global float *g_Hx,
                              read_only __global float *g_Hy,
                              read_only __global float *g_Ez,
                              __global float *g_rhsHx,
                              __global float *g_rhsHy,
                              __global float *g_rhsEz,
                              read_only __global float *g_surfinfo,
                              read_only __global float4 *g_LIFT)
{
  /* LOCKED IN to using Np threads per block */
  const int n = get_local_id(0);
  const int k = get_group_id(0);

  __local l_fluxHx[p_Nafp];
  __local l_fluxHy[p_Nafp];
  __local l_fluxEz[p_Nafp];

  int m;

  /* grab surface nodes and store flux in shared memory */
  if (n < p_Nafp)
  {
    /* coalesced reads (maybe) */
    m = 6*(k*p_Nafp)+n;

    const  int   idM = g_surfinfo[m]; m += p_Nfp*p_Nfaces;
    int          idP = g_surfinfo[m]; m += p_Nfp*p_Nfaces;
    const  float Fsc = g_surfinfo[m]; m += p_Nfp*p_Nfaces;
    const  float Bsc = g_surfinfo[m]; m += p_Nfp*p_Nfaces;
    const  float nx  = g_surfinfo[m]; m += p_Nfp*p_Nfaces;
    const  float ny  = g_surfinfo[m];

    /* check if idP<0  */
    float dHx=0, dHy=0, dEz=0;
    if (idP>=0)
    {
      dHx = Fsc*(    g_Hx[idP] - g_Hx[idM]);
      dHy = Fsc*(    g_Hy[idP] - g_Hy[idM]);
      dEz = Fsc*(Bsc*g_Ez[idP] - g_Ez[idM]);
    }

    const float ndotdH = nx*dHx + ny*dHy;

    m = n;
    l_fluxHx[m] = -ny*dEz + dHx - ndotdH*nx;
    l_fluxHy[m] =  nx*dEz + dHy - ndotdH*ny;
    l_fluxEz[m] =  nx*dHy - ny*dHx + dEz;
  }

  /* make sure all element data points are cached */
  barrier(CLK_LOCAL_MEM_FENCE);

  if (n < p_Np)
  {
    float rhsHx = 0, rhsHy = 0, rhsEz = 0;

    int sk = n;

    /* can manually unroll to 3 because there are 3 faces */
    for (m=0;p_Nfaces*p_Nfp-m;)
    {
      float4 L = g_LIFT[sk];
      sk += p_Np;

      rhsHx += L.x*l_fluxHx[m];
      rhsHy += L.x*l_fluxHy[m];
      rhsEz += L.x*l_fluxEz[m];
      ++m;

      /* broadcast */
      rhsHx += L.y*l_fluxHx[m];
      rhsHy += L.y*l_fluxHy[m];
      rhsEz += L.y*l_fluxEz[m];
      ++m;

      /* broadcast */
      rhsHx += L.z*l_fluxHx[m];
      rhsHy += L.z*l_fluxHy[m];
      rhsEz += L.z*l_fluxEz[m];
      ++m;
    }

    m = n+k*BSIZE;

    g_rhsHx[m] += rhsHx;
    g_rhsHy[m] += rhsHy;
    g_rhsEz[m] += rhsEz;
  }
}
"""

# }}}




def MaxwellLocalRef(d, Hx, Hy, Ez):
    # local derivatives of fields
    Ezx, Ezy = d.grad(Ez)
    CuHx, CuHy, CuHz = d.curl(Hx, Hy,0)

    # compute right hand sides of the PDE's
    rhsHx = -Ezy
    rhsHy =  Ezx
    rhsEz =  CuHz
    return rhsHx, rhsHy, rhsEz




class CLMaxwellsRhs2D:
    def __init__(self, discr):
        self.discr = discr

        self.volume_kernel = cl.Program(discr.ctx,
                MAXWELL_VOLUME_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "BSIZE": discr.block_size,
                    }
                ).build().MaxwellsVolume2d
        self.volume_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

        self.surface_kernel = cl.Program(discr.ctx,
                MAXWELL_SURFACE_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "Nfp": discr.ldis.Nfp,
                    "Nfaces": discr.ldis.Nfaces,
                    "BSIZE": discr.block_size,
                    }
                ).build().MaxwellsSurface2d
        self.surface_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

    def local(self, Hx, Hy, Ez):
        d = self.discr

        rhsHx = d.volume_empty()
        rhsHy = d.volume_empty()
        rhsEz = d.volume_empty()

        self.volume_kernel(d.queue, 
                (d.K*d.block_size,), (d.block_size,),
                d.K,
                Hx.data, Hy.data, Ez.data,
                rhsHx.data, rhsHy.data, rhsEz.data,
                d.diffmatrices_dev.data, d.drdx_dev.data)

        return rhsHx, rhsHy, rhsEz




# {{{ test

def test():
    from nudg import read_2d_gambit_mesh
    from nudg import LocalDiscretization2D
    d = CLDiscretization2D(LocalDiscretization2D(5),
            *read_2d_gambit_mesh('Maxwell025.neu'))

    # set initial conditions
    mmode = 1; nmode = 1
    Ez = np.sin(mmode*np.pi*d.x)*np.sin(nmode*np.pi*d.y)
    Hx = np.zeros((d.ldis.Np, d.K))
    Hy = np.zeros((d.ldis.Np, d.K))

    Ez_dev = d.to_dev(Ez.astype(np.float32))
    Hx_dev = d.to_dev(Hx.astype(np.float32))
    Hy_dev = d.to_dev(Hy.astype(np.float32))

    dev_max = CLMaxwellsRhs2D(d)
    rhsHx_dev, rhsHy_dev, rhsEz_dev = dev_max.local(Ez_dev, Hx_dev, Hy_dev)
    rhsHx, rhsHy, rhsEz = MaxwellLocalRef(d, Ez, Hx, Hy)

    rhsEz2 = d.from_dev(rhsEz_dev)
    print rhsEz2[:,0]
    print rhsEz[:,0]

    print la.norm(rhsEz2 - rhsEz)/la.norm(rhsEz)
    #Hx, Hy, Ez, time = Maxwell2D(d, Hx, Hy, Ez, final_time=5)

if __name__ == "__main__":
    test()

# }}}

# vim: foldmethod=marker
