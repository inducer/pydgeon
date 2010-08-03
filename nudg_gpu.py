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




class GPUDiscretization2D(Discretization2D):
    def __init__(self, ldis, Nv, VX, VY, K, EToV):
        Discretization2D.__init__(self, ldis, Nv, VX, VY, K, EToV)

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # FIXME BSIZE
        drds_gpu = np.empty((ldis.Np, ldis.Np, 4), dtype=np.float32)
        drds_gpu[:,:,0] = ldis.Dr.T
        drds_gpu[:,:,1] = ldis.Ds.T
        self.diffmatrices_gpu = cl_array.to_device(self.ctx, self.queue, drds_gpu)

        drdx_gpu = np.empty((self.K, self.dimensions**2), dtype=np.float32)
        drdx_gpu[:,0] = self.rx[0]
        drdx_gpu[:,1] = self.ry[0]
        drdx_gpu[:,2] = self.sx[0]
        drdx_gpu[:,3] = self.sy[0]
        self.drdx_gpu = cl_array.to_device(self.ctx, self.queue, drdx_gpu)

    def to_gpu(self, vec):
        vec = np.asarray(vec, dtype=np.float32, order="F")
        return cl_array.to_device(self.ctx, self.queue, vec)

    def volume_empty(self):
        # FIXME BSIZE
        return cl_array.Array(
                self.ctx, queue=self.queue,
                shape=(self.ldis.Np, self.K, ), 
                dtype=np.float32, order="F")






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

  __local float s_Hx[BSIZE];
  __local float s_Hy[BSIZE];
  __local float s_Ez[BSIZE];

  s_Hx[id] = g_Hx[m];
  s_Hy[id] = g_Hy[m];
  s_Ez[id] = g_Ez[m];

  barrier(CLK_LOCAL_MEM_FENCE);

  float dHxdr=0,dHxds=0;
  float dHydr=0,dHyds=0;
  float dEzdr=0,dEzds=0;

  float Q;
  for(m=0; m<p_Np; ++m)
  {
    float4 D = g_DrDs[(n+m*BSIZE)];

    id = m;
    Q = s_Hx[m]; dHxdr += D.x*Q; dHxds += D.y*Q;
    Q = s_Hy[m]; dHydr += D.x*Q; dHyds += D.y*Q; 
    Q = s_Ez[m]; dEzdr += D.x*Q; dEzds += D.y*Q;
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




def MaxwellLocalRef(d, Hx, Hy, Ez):
    # local derivatives of fields
    Ezx, Ezy = d.grad(Ez)
    CuHx, CuHy, CuHz = d.curl(Hx, Hy,0)

    # compute right hand sides of the PDE's
    rhsHx = -Ezy
    rhsHy =  Ezx
    rhsEz =  CuHz
    return rhsHx, rhsHy, rhsEz




class MaxwellsRhs2DGPU:
    def __init__(self, discr):
        self.discr = discr

        self.volume_kernel = cl.Program(discr.ctx,
                MAXWELL_VOLUME_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "BSIZE": discr.ldis.Np,
                    }
                ).build().MaxwellsVolume2d
        self.volume_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

    def local(self, Hx, Hy, Ez):
        d = self.discr

        rhsHx = d.volume_empty()
        rhsHy = d.volume_empty()
        rhsEz = d.volume_empty()

        # FIXME BSIZE
        self.volume_kernel(d.queue, (d.K*d.ldis.Np,), (d.ldis.Np,),
                d.K, 
                Hx.data, Hy.data, Ez.data,
                rhsHx.data, rhsHy.data, rhsEz.data,
                d.diffmatrices_gpu.data, d.drdx_gpu.data)

        return rhsHx, rhsHy, rhsEz





# {{{ test

def test():
    from nudg import read_2d_gambit_mesh
    from nudg import LocalDiscretization2D
    d = GPUDiscretization2D(LocalDiscretization2D(5),
            *read_2d_gambit_mesh('Maxwell025.neu'))

    # set initial conditions
    mmode = 1; nmode = 1
    Ez = np.sin(mmode*np.pi*d.x)*np.sin(nmode*np.pi*d.y)
    Hx = np.zeros((d.ldis.Np, d.K))
    Hy = np.zeros((d.ldis.Np, d.K))

    Ez_gpu = d.to_gpu(Ez.astype(np.float32))
    Hx_gpu = d.to_gpu(Hx.astype(np.float32))
    Hy_gpu = d.to_gpu(Hy.astype(np.float32))

    gpu_max = MaxwellsRhs2DGPU(d)
    rhsHx_gpu, rhsHy_gpu, rhsEz_gpu = gpu_max.local(Ez_gpu, Hx_gpu, Hy_gpu)
    rhsHx, rhsHy, rhsEz = MaxwellLocalRef(d, Ez, Hx, Hy)

    rhsEz2 = rhsEz_gpu.get()
    print la.norm(rhsEz2 - rhsEz)/la.norm(rhsEz)
    #Hx, Hy, Ez, time = Maxwell2D(d, Hx, Hy, Ez, final_time=5)

if __name__ == "__main__":
    test()

# }}}

# vim: foldmethod=marker
