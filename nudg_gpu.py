# Pydgeon - the Python DG Environment
# (C) 2010 Tim Warburton, Andreas Kloeckner
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

        self.prepare_dev_data()

    def prepare_dev_data(self):
        ldis = self.ldis

        # differentiation matrix
        drds_dev = np.empty((ldis.Np, self.block_size, 4), dtype=np.float32)
        drds_dev[:,:ldis.Np,0] = ldis.Dr.T
        drds_dev[:,:ldis.Np,1] = ldis.Ds.T
        self.diffmatrices_dev = cl_array.to_device(self.ctx, self.queue, drds_dev)

        # geometric coefficients
        drdx_dev = np.empty((self.K, self.dimensions**2), dtype=np.float32)
        drdx_dev[:,0] = self.rx[:, 0]
        drdx_dev[:,1] = self.ry[:, 0]
        drdx_dev[:,2] = self.sx[:, 0]
        drdx_dev[:,3] = self.sy[:, 0]
        self.drdx_dev = cl_array.to_device(self.ctx, self.queue, drdx_dev)

        # lift matrix
        lift_dev = np.empty((ldis.Nfp, ldis.Np, 4), dtype=np.float32)
        partitioned_lift = ldis.LIFT.reshape(ldis.Np, -1, ldis.Nfaces)

        for i in range(ldis.Nfaces):
            lift_dev[:, :, i] = partitioned_lift[:, :, i].T
        self.lift_dev = cl_array.to_device(self.ctx, self.queue, lift_dev)

        # surface info
        surfinfo_dev = np.empty((self.K, 6, ldis.Nafp), dtype=np.float32)

        el_p, face_i_p = divmod(self.vmapP.reshape(-1, ldis.Nafp), ldis.Np)
        el_m, face_i_m = divmod(self.vmapM.reshape(-1, ldis.Nafp), ldis.Np)

        ind_p = el_p * self.block_size + face_i_p
        ind_m = el_m * self.block_size + face_i_m

        surfinfo_dev[:, 0, :] = ind_m
        surfinfo_dev[:, 1, :] = ind_p
        surfinfo_dev[:, 2, :] = self.Fscale
        surfinfo_dev[:, 3, :] = np.where(ind_m==ind_p, -1, 1)
        surfinfo_dev[:, 4, :] = self.nx
        surfinfo_dev[:, 5, :] = self.ny

        self.surfinfo_dev = cl_array.to_device(self.ctx, self.queue, surfinfo_dev)

    def to_dev(self, vec):
        dev_vec = np.empty((self.K, self.block_size), dtype=np.float32)
        dev_vec[:, :self.ldis.Np] = vec
        return cl_array.to_device(self.ctx, self.queue, dev_vec)

    def from_dev(self, vec):
        return vec.get()[:, :self.ldis.Np]

    def volume_empty(self):
        return cl_array.Array(
                self.ctx, queue=self.queue,
                shape=(self.K, self.block_size),
                dtype=np.float32)




# {{{ kernels

MAXWELL2D_VOLUME_KERNEL = """
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

MAXWELL2D_SURFACE_KERNEL = """
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

  __local float l_fluxHx[p_Nafp];
  __local float l_fluxHy[p_Nafp];
  __local float l_fluxEz[p_Nafp];

  int m;

  /* grab surface nodes and store flux in shared memory */
  if (n < p_Nafp)
  {
    /* coalesced reads (maybe) */
    m = 6*(k*p_Nafp)+n;

    const  int   idM = g_surfinfo[m]; m += p_Nafp;
    int          idP = g_surfinfo[m]; m += p_Nafp;
    const  float Fsc = g_surfinfo[m]; m += p_Nafp;
    const  float Bsc = g_surfinfo[m]; m += p_Nafp;
    const  float nx  = g_surfinfo[m]; m += p_Nafp;
    const  float ny  = g_surfinfo[m];

    /* check if idP<0  */
    float dHx=0, dHy=0, dEz=0;
    if (idP>=0)
    {
      dHx = 0.5f*Fsc*(    g_Hx[idP] - g_Hx[idM]);
      dHy = 0.5f*Fsc*(    g_Hy[idP] - g_Hy[idM]);
      dEz = 0.5f*Fsc*(Bsc*g_Ez[idP] - g_Ez[idM]);
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
                MAXWELL2D_VOLUME_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "BSIZE": discr.block_size,
                    }
                ).build().MaxwellsVolume2d
        self.volume_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

        self.surface_kernel = cl.Program(discr.ctx,
                MAXWELL2D_SURFACE_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "Nfp": discr.ldis.Nfp,
                    "Nfaces": discr.ldis.Nfaces,
                    "BSIZE": discr.block_size,
                    }
                ).build().MaxwellsSurface2d
        self.surface_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

    def __call__(self, Hx, Hy, Ez):
        d = self.discr
        ldis = d.ldis

        rhsHx = d.volume_empty()
        rhsHy = d.volume_empty()
        rhsEz = d.volume_empty()

        self.volume_kernel(d.queue, 
                (d.K*d.block_size,), (d.block_size,),
                d.K,
                Hx.data, Hy.data, Ez.data,
                rhsHx.data, rhsHy.data, rhsEz.data,
                d.diffmatrices_dev.data, d.drdx_dev.data)

        surf_block_size = max(
                ldis.Nfp*ldis.Nfaces,
                ldis.Np)

        self.surface_kernel(d.queue, 
                (d.K*surf_block_size,), (surf_block_size,),
                d.K,
                Hx.data, Hy.data, Ez.data,
                rhsHx.data, rhsHy.data, rhsEz.data,
                d.surfinfo_dev.data, d.lift_dev.data)

        return rhsHx, rhsHy, rhsEz




# {{{ test

def test():
    from nudg import (LocalDiscretization2D,
            read_2d_gambit_mesh, 
            make_obj_array, JacobiGQ, runge_kutta)
    d = CLDiscretization2D(LocalDiscretization2D(N=9),
            *read_2d_gambit_mesh('Maxwell025.neu'))

    dev_max = CLMaxwellsRhs2D(d)

    # set initial conditions
    #mmode = 1.3; nmode = 1.2
    mmode = 3; nmode = 2
    Hx = np.zeros((d.K, d.ldis.Np))
    Hy = np.zeros((d.K, d.ldis.Np))
    Ez = np.sin(mmode*np.pi*d.x)*np.sin(nmode*np.pi*d.y)

    Ez_dev = d.to_dev(Ez.astype(np.float32))
    Hx_dev = d.to_dev(Hx.astype(np.float32))
    Hy_dev = d.to_dev(Hy.astype(np.float32))

    state = make_obj_array([Hx_dev, Hy_dev, Ez_dev])

    # compute time step size
    rLGL = JacobiGQ(0,0, d.ldis.N)[0]
    rmin = abs(rLGL[0]-rLGL[1])
    dt_scale = d.dt_scale()
    dt = dt_scale.min()*rmin*2/3

    do_vis = False
    do_vis = True
    if do_vis:
        try:
            import enthought.mayavi.mlab as mayavi
        except ImportError:
            do_vis = False

    if do_vis:
        vis_mesh = mayavi.triangular_mesh(
                d.x.ravel(), d.y.ravel(), Ez.ravel(),
                d.gen_vis_triangles())

    def vis_hook(step, t, state):
        Hx, Hy, Ez = state
        if step % 10 == 0 and do_vis:
            Ez = d.from_dev(Ez)

            vis_mesh.mlab_source.z = Ez.ravel()
            print la.norm(Ez)

    def rhs(t, state):
        return make_obj_array(dev_max(*state))

    time, final_state = runge_kutta(state, rhs, dt, final_time=5,
            vis_hook=vis_hook)


if __name__ == "__main__":
    test()

# }}}

# vim: foldmethod=marker
