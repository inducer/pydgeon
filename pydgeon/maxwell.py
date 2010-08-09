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

import numpy as np




# {{{ CPU

def MaxwellRHS2D(discr, Hx, Hy, Ez):
    """Evaluate RHS flux in 2D Maxwell TM form."""

    from pydgeon.tools import eldot

    d = discr
    l = discr.ldis

    # Define field differences at faces
    vmapM = d.vmapM.reshape(d.K, -1)
    vmapP = d.vmapP.reshape(d.K, -1)

    dHx = Hx.flat[vmapM]-Hx.flat[vmapP]
    dHy = Hy.flat[vmapM]-Hy.flat[vmapP]
    dEz = Ez.flat[vmapM]-Ez.flat[vmapP]

    # Impose reflective boundary conditions (Ez+ = -Ez-)
    dHx.flat[d.mapB] = 0
    dHy.flat[d.mapB] = 0
    dEz.flat[d.mapB] = 2*Ez.flat[d.vmapB]

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
    rhsHx = -Ezy  + eldot(l.LIFT, (d.Fscale*fluxHx))/2.0
    rhsHy =  Ezx  + eldot(l.LIFT, (d.Fscale*fluxHy))/2.0
    rhsEz =  CuHz + eldot(l.LIFT, (d.Fscale*fluxEz))/2.0
    return rhsHx, rhsHy, rhsEz

# }}}

# {{{ OpenCL

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
  __local float l_Hx[BSIZE];
  __local float l_Hy[BSIZE];
  __local float l_Ez[BSIZE];

  /* LOCKED IN to using Np work items per group */
// start_vol_kernel
  const int n = get_local_id(0);
  const int k = get_group_id(0);

  /* "coalesced"  */
  int m = n+k*BSIZE;
  int id = n;

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
  g_rhsEz[m] =  (drdx*dHydr+dsdx*dHyds 
    - drdy*dHxdr-dsdy*dHxds);
// end
}
"""

MAXWELL2D_SURFACE_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define p_Nfp %(Nfp)d
#define p_Nfaces %(Nfaces)d
#define p_Nafp (p_Nfaces*p_Nfp)
#define BSIZE %(BSIZE)d

// start_surf_kernel
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
// end
"""

# }}}




class CLMaxwellsRhs2D:
    def __init__(self, discr):
        import pyopencl as cl

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

# }}}

# vim: foldmethod=marker
