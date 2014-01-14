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

def AcousticsRHS3D(discr, Ux, Uy, Uz, Pr):
    """Evaluate RHS flux in 3D Acoustics."""

    from pydgeon.tools import eldot

    d = discr
    l = discr.ldis

    # Define field differences at faces
    vmapM = d.vmapM.reshape(d.K, -1)
    vmapP = d.vmapP.reshape(d.K, -1)

    dUx = Ux.flat[vmapP] - Ux.flat[vmapM]
    dUy = Uy.flat[vmapP] - Uy.flat[vmapM]
    dUz = Uz.flat[vmapP] - Uz.flat[vmapM]
    dPr = Pr.flat[vmapP] - Pr.flat[vmapM]

    # Impose reflective boundary conditions (Uz+ = -Uz-)
    dUx.flat[d.mapB] = 0.0
    dUy.flat[d.mapB] = 0.0
    dUz.flat[d.mapB] = 0.0
    dPr.flat[d.mapB] = (-2.0)*Pr.flat[d.vmapB]

    # evaluate upwind fluxes
    #alpha = 1.0
    R = dPr - d.nx*dUx - d.ny*dUy - d.nz*dUz
    fluxUx = -d.nx*R
    fluxUy = -d.ny*R
    fluxUz = -d.nz*R
    fluxPr = R

    # local derivatives of fields
    dPrdx, dPrdy, dPrdz = d.gradient(Pr)
    divU = d.divergence(Ux, Uy, Uz)

    # compute right hand sides of the PDE's
    rhsUx = -dPrdx + eldot(l.LIFT, (d.Fscale*fluxUx))/2.0
    rhsUy = -dPrdx + eldot(l.LIFT, (d.Fscale*fluxUy))/2.0
    rhsUz = -dPrdz + eldot(l.LIFT, (d.Fscale*fluxUz))/2.0
    rhsPr = -divU + eldot(l.LIFT, (d.Fscale*fluxPr))/2.0

    return rhsUx, rhsUy, rhsUz, rhsPr

# }}}


# {{{ loopy

class LoopyAcousticsRHS3D:
    def __init__(self, queue, cl_discr_info, dtype=np.float64,
            profile=False):
        context = queue.context
        discr = self.discr = cl_discr_info.discr
        self.cl_discr_info = cl_discr_info

        self.profile = profile

        import pyopencl as cl
        import pyopencl.array  # noqa

        dtype4 = cl.array.vec.types[np.dtype(dtype), 4]

        ldis = discr.ldis

        from pyopencl.characterize import get_fast_inaccurate_build_options
        build_options = get_fast_inaccurate_build_options(context.devices[0])

        # {{{ volume kernel

        import loopy as lp
        volume_kernel = lp.make_kernel(context.devices[0], [
            "{[n,m,k]: 0<= n,m < Np and 0<= k < K}",
            ],
            """
                <> du_drst = sum(m, DrDsDt[n,m]*u[k,m])
                <> dv_drst = sum(m, DrDsDt[n,m]*v[k,m])
                <> dw_drst = sum(m, DrDsDt[n,m]*w[k,m])
                <> dp_drst = sum(m, DrDsDt[n,m]*p[k,m])

                rhsu[k,n] = - dot(drst_dx[k],dp_drst)
                rhsv[k,n] = - dot(drst_dy[k],dp_drst)
                rhsw[k,n] = - dot(drst_dz[k],dp_drst)
                rhsp[k,n] = - (dot(drst_dx[k], du_drst) + dot(drst_dy[k], dv_drst) \
                    + dot(drst_dz[k], dw_drst))
                """,
            [
                lp.GlobalArg("DrDsDt", dtype4, shape="Np, Np", order="F"),
                "...",
                ],
            name="dg_volume", assumptions="K>=1",
            defines=dict(Np=discr.ldis.Np),
            options=dict(no_numpy=True, cl_build_options=build_options))

        def transform_vol(knl):
            knl = lp.tag_inames(knl, dict(n="l.0", k="g.0"))
            #knl = lp.change_arg_to_image(knl, "DrDsDt")

            # knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
            for name in ["u", "v", "w", "p"]:
                knl = lp.add_prefetch(knl, "%s[k,:]" % name)
            for name in ["drst_dx", "drst_dy", "drst_dz"]:
                knl = lp.add_prefetch(knl, "%s" % name)
            knl = lp.add_prefetch(knl, "DrDsDt")
            return knl

        self.volume_kernel = transform_vol(volume_kernel)

        self.volume_flops = discr.K * (
                (
                    4  # num components
                    * 3*discr.ldis.Np**2*2
                    )
                +
                (
                    (3*2-1)*discr.ldis.Np * 6
                    )
                + 2)

        self.volume_bytes = np.dtype(dtype).itemsize * discr.K * (
                (
                    4  # num components
                    * 2  # load, store
                    * discr.ldis.Np
                    )
                +
                # geometric factors
                6)

        # }}}

        # {{{ surface kernel

        NfpNfaces = ldis.Nfaces*ldis.Nfp

        surface_kernel = lp.make_kernel(context.devices[0],
                "{[m,mp,n,k]: 0<= m,mp < NfpNfaces and 0<= n < Np and 0<= k < K }",
                """
                    <> idP = vmapP[k,m]
                    <> idM = vmapM[k,m]

                    <> du = u[[idP]]-u[[idM]]
                    <> dv = v[[idP]]-v[[idM]]
                    <> dw = w[[idP]]-w[[idM]]
                    <> dp = bc[k,m]*p[[idP]] - p[[idM]]

                    <> dQ = 0.5*Fscale[k,m]* \
                            (dp - nx[k,m]*du - ny[k,m]*dv - nz[k,m]*dw)

                    <> fluxu[m] = -nx[k,m]*dQ
                    <> fluxv[m] = -ny[k,m]*dQ
                    <> fluxw[m] = -nz[k,m]*dQ
                    <> fluxp[m] =          dQ

                    # reduction here
                    rhsu[k,n] = rhsu[k,n] + sum(mp, LIFT[n,mp]*fluxu[mp])
                    rhsv[k,n] = rhsv[k,n] + sum(mp, LIFT[n,mp]*fluxv[mp])
                    rhsw[k,n] = rhsw[k,n] + sum(mp, LIFT[n,mp]*fluxw[mp])
                    rhsp[k,n] = rhsp[k,n] + sum(mp, LIFT[n,mp]*fluxp[mp])
                    """,
                [
                    lp.GlobalArg("u,v,w,p", dtype, shape="K, Np", order="C"),
                    lp.GlobalArg("LIFT", dtype, shape="Np, NfpNfaces", order="F"),
                    "...",
                    ],
                name="dg_surface", assumptions="K>=1",
                defines=dict(Np=ldis.Np, Nfp=ldis.Nfp, NfpNfaces=NfpNfaces),
                options=dict(no_numpy=True, cl_build_options=build_options))

        def transform_surface_kernel(knl):
            print knl
            knl = lp.tag_inames(knl, dict(k="g.0", n="l.0", m="l.0"))
            knl = lp.split_iname(knl, "mp", 4, inner_tag="unr")
            knl = lp.add_prefetch(knl, "LIFT")
            for name in ["nx", "ny", "nz", "Fscale", "bc"]:
                knl = lp.add_prefetch(knl, name)
            knl = lp.set_loop_priority(knl, "mp_outer,mp_inner")
            return knl

        self.surface_kernel = transform_surface_kernel(surface_kernel)

        self.surface_flops = (discr.K
                * (
                    NfpNfaces*15
                    +
                    4*discr.ldis.Np*NfpNfaces*2
                    ))

        # }}}

    def __call__(self, queue, Ux, Uy, Uz, Pr):
        """Evaluate RHS flux in 3D Acoustics."""

        d = self.discr

        cl_info = self.cl_discr_info

        # local derivatives of fields
        evt, (rhsUx, rhsUy, rhsUz, rhsPr) = self.volume_kernel(
                queue, u=Ux, v=Uy, w=Uz, p=Pr,
                DrDsDt=cl_info.drdsdt,
                drst_dx=cl_info.drst_dx, drst_dy=cl_info.drst_dy,
                drst_dz=cl_info.drst_dz,
                K=d.K,
                allocator=cl_info.allocator)

        cl_info.volume_events.append(evt)

        if 1:
            evt, (rhsUx, rhsUy, rhsUz, rhsPr) = self.surface_kernel(
                queue,
                vmapP=cl_info.vmapP,
                vmapM=cl_info.vmapM,
                u=Ux, v=Uy, w=Uz, p=Pr,
                rhsu=rhsUx, rhsv=rhsUy, rhsw=rhsUz, rhsp=rhsPr,
                nx=cl_info.nx,
                ny=cl_info.ny,
                nz=cl_info.nz,
                Fscale=cl_info.Fscale,
                bc=cl_info.bc,
                LIFT=cl_info.LIFT, K=d.K,
                allocator=cl_info.allocator)

            cl_info.surface_events.append(evt)

        return rhsUx, rhsUy, rhsUz, rhsPr

# }}}

# {{{ OpenCL

# {{{ kernels

ACOUSTICS3D_VOLUME_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define BSIZE %(BSIZE)d

__kernel void AcousticsVolume3d(
   int K,
   __read_only __global float *g_Ux,
   __read_only __global float *g_Uy,
   __read_only __global float *g_Uz,
   __read_only __global float *g_Pr,
   __global float *g_rhsUx,
   __global float *g_rhsUy,
   __global float *g_rhsUz,
   __global float *g_rhsPr,
 //  __global float4 *g_DrDsDt,
  image2d_t  i_DrDsDt,
  __read_only __global float4 *g_drst_dx,
  __read_only __global float4 *g_drst_dy,
  __read_only __global float4 *g_drst_dz)
{
  const sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP
    | CLK_FILTER_NEAREST;

  __local float l_Ux[p_Np];
  __local float l_Uy[p_Np];
  __local float l_Uz[p_Np];
  __local float l_Pr[p_Np];

  /* LOCKED IN to using Np work items per group */
  const int n = get_local_id(0);
  const int k = get_group_id(0);

  int m = n+k*BSIZE;

  l_Ux[n] = g_Ux[m];
  l_Uy[n] = g_Uy[m];
  l_Uz[n] = g_Uz[m];
  l_Pr[n] = g_Pr[m];

  barrier(CLK_LOCAL_MEM_FENCE);

  float dUxdr=0,dUxds=0,dUxdt=0;
  float dUydr=0,dUyds=0,dUydt=0;
  float dUzdr=0,dUzds=0,dUzdt=0;
  float dPrdr=0,dPrds=0,dPrdt=0;

  float Q;
  for(m=0; m<p_Np; ++m)
  {
    float4 D = read_imagef(i_DrDsDt, samp, (int2)(n, m));
    // float4 D = g_DrDsDt[ n + m*p_Np ]; // column major

    Q = l_Ux[m]; dUxdr += D.x*Q; dUxds += D.y*Q; dUxdt += D.z*Q;
    Q = l_Uy[m]; dUydr += D.x*Q; dUyds += D.y*Q; dUydt += D.z*Q;
    Q = l_Uz[m]; dUzdr += D.x*Q; dUzds += D.y*Q; dUzdt += D.z*Q;
    Q = l_Pr[m]; dPrdr += D.x*Q; dPrds += D.y*Q; dPrdt += D.z*Q;
  }

  const float4 drst_dx = g_drst_dx[k];
  const float4 drst_dy = g_drst_dy[k];
  const float4 drst_dz = g_drst_dz[k];

  m = n+BSIZE*k;
  g_rhsUx[m] = -(drst_dx.x*dPrdr+drst_dx.y*dPrds+drst_dx.z*dPrdt);
  g_rhsUy[m] = -(drst_dy.x*dPrdr+drst_dy.y*dPrds+drst_dy.z*dPrdt);
  g_rhsUz[m] = -(drst_dz.x*dPrdr+drst_dz.y*dPrds+drst_dz.z*dPrdt);

  g_rhsPr[m] = -(drst_dx.x*dUxdr+drst_dx.y*dUxds+drst_dx.z*dUxdt)
               -(drst_dy.x*dUydr+drst_dy.y*dUyds+drst_dy.z*dUydt)
               -(drst_dz.x*dUzdr+drst_dz.y*dUzds+drst_dz.z*dUzdt);

// end
}
"""

ACOUSTICS3D_SURFACE_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define p_Nfp %(Nfp)d
#define p_Nfaces %(Nfaces)d
#define p_Nafp (p_Nfaces*p_Nfp)
#define BSIZE %(BSIZE)d

// start_surf_kernel
__kernel void AcousticssSurface3d(
  int K,
  __read_only __global float *g_Ux,
  __read_only __global float *g_Uy,
  __read_only __global float *g_Uz,
  __read_only __global float *g_Pr,
  __global float *g_rhsUx,
  __global float *g_rhsUy,
  __global float *g_rhsUz,
  __global float *g_rhsPr,
  read_only __global float *g_nx,
  read_only __global float *g_ny,
  read_only __global float *g_nz,
  read_only __global float *g_Fscale,
  read_only __global float *g_bc,
  read_only __global int   *g_vmapM,
  read_only __global int   *g_vmapP,
  read_only __global float *g_LIFT)
{

  /* LOCKED IN to using Np threads per block */
  const int n = get_local_id(0);
  const int k = get_group_id(0);

  __local float l_fluxUx[p_Nafp];
  __local float l_fluxUy[p_Nafp];
  __local float l_fluxUz[p_Nafp];
  __local float l_fluxPr[p_Nafp];

  int m;

  /* grab surface nodes and store flux in shared memory */
  if (n < p_Nafp)
  {
    /* coalesced reads (maybe) */
    m = k*p_Nafp+n;

    const  int   idM = g_vmapM [m]; 
    const  int   idP = g_vmapP [m];
    const  float Fsc = g_Fscale[m];
    const  float Bsc = (idP==idM) ? -1.f:1.f; // broken: g_bc    [m]; 
    const  float nx  = g_nx    [m];
    const  float ny  = g_ny    [m];
    const  float nz  = g_nz    [m];

    float dUx =     g_Ux[idP] - g_Ux[idM];
    float dUy =     g_Uy[idP] - g_Uy[idM];
    float dUz =     g_Uz[idP] - g_Uz[idM];
    float dPr = Bsc*g_Pr[idP] - g_Pr[idM];

    const float dQ = 0.5f*Fsc*(dPr - nx*dUx - ny*dUy - nz*dUz);

    l_fluxUx[n] =  -nx*dQ;
    l_fluxUy[n] =  -ny*dQ;
    l_fluxUz[n] =  -nz*dQ;
    l_fluxPr[n] =      dQ;
  }

  /* make sure all element data points are cached */
  barrier(CLK_LOCAL_MEM_FENCE);

  if (n < p_Np)
  {
    float rhsUx = 0, rhsUy = 0, rhsUz = 0, rhsPr = 0;

    /* can manually unroll to 4 because there are 4 faces */
    for (m=0;m < (p_Nfaces*p_Nfp);++m)
    {
      const float L = g_LIFT[n+p_Np*m];

      rhsUx += L*l_fluxUx[m];
      rhsUy += L*l_fluxUy[m];
      rhsUz += L*l_fluxUz[m];
      rhsPr += L*l_fluxPr[m];
    }

    m = n+k*BSIZE;

    g_rhsUx[m] += rhsUx;
    g_rhsUy[m] += rhsUy;
    g_rhsUz[m] += rhsUz;
    g_rhsPr[m] += rhsPr;
  }

}
// end
"""

# }}}




class CLAcousticsRHS3D:
    def __init__(self, queue, cl_info, dtype):
        assert dtype == np.float32

        discr = cl_info.discr
        self.cl_info = cl_info

        import pyopencl as cl
        from pydgeon.opencl import CL_OPTIONS

        self.queue = queue
        self.discr = discr

        self.volume_kernel = cl.Program(queue.context,
                ACOUSTICS3D_VOLUME_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "BSIZE": discr.ldis.Np,
                    }
                ).build(options=CL_OPTIONS).AcousticsVolume3d
        self.volume_kernel.set_scalar_arg_dtypes([np.int32] + 12*[None])

        self.volume_flops = discr.K * (
                ( 4 # num components
                * 3*discr.ldis.Np**2*2
                )
                +
                (
                    (3*2-1)*discr.ldis.Np * 6
                    )
                + 2)

        self.surface_kernel = cl.Program(queue.context,
                ACOUSTICS3D_SURFACE_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "Nfp": discr.ldis.Nfp,
                    "Nfaces": discr.ldis.Nfaces,
                    "BSIZE": discr.ldis.Np,
                    }
                ).build(options=CL_OPTIONS).AcousticssSurface3d
        self.surface_kernel.set_scalar_arg_dtypes([np.int32] + 16*[None])

        NfpNfaces = discr.ldis.Nfp * discr.ldis.Nfaces
        self.surface_flops = (discr.K
                *(
                    NfpNfaces*15
                    +
                    4*discr.ldis.Np*NfpNfaces*2
                    ))

    def __call__(self, Ux, Uy, Uz, Pr):
        d = self.discr
        ldis = d.ldis

        cl_info = self.cl_info

        import pyopencl as cl

        rhsUx = cl.array.empty_like(Ux)
        rhsUy = cl.array.empty_like(Ux)
        rhsUz = cl.array.empty_like(Ux)
        rhsPr = cl.array.empty_like(Ux)
        block_size = ldis.Np

        vol_evt = self.volume_kernel(self.queue,
                                     (d.K,), (block_size,),
                                     d.K,
                                     Ux.data, Uy.data, Uz.data, Pr.data,
                                     rhsUx.data, rhsUy.data, rhsUz.data, rhsPr.data,
                                     cl_info.drdsdt_img,  # cl_info.drdsdt.data,
                                     cl_info.drst_dx.data,
                                     cl_info.drst_dy.data,
                                     cl_info.drst_dz.data,
                                     g_times_l=True)


        surf_block_size = max(ldis.Nfp*ldis.Nfaces, block_size)

        # cl_info.bc.data seems broken
        sfc_evt = self.surface_kernel(self.queue,
                                      (d.K,), (surf_block_size,),
                                      d.K,
                                      Ux.data, Uy.data, Uz.data, Pr.data,
                                      rhsUx.data, rhsUy.data, rhsUz.data, rhsPr.data,
                                      cl_info.nx.data,
                                      cl_info.ny.data,
                                      cl_info.nz.data,
                                      cl_info.Fscale.data,
                                      cl_info.bc.data,
                                      cl_info.vmapM.data,
                                      cl_info.vmapP.data,
                                      cl_info.LIFT.data,
                g_times_l=True)

        cl_info.surface_events.append(sfc_evt)

        cl_info.volume_events.append(vol_evt)

        return rhsUx, rhsUy, rhsUz, rhsPr

    def rhs_flops(self):
        discr = self.discr
        ldis = discr.ldis
        K = discr.K
        Np = ldis.Np
        Nafp = ldis.Nafp
        d = discr.dimensions

        # need to fix flops
        rs_diff_flops = 2 * K * Np**2
        l2g_diff_flops = 2 * K * Np * (3*4+1)
        lift_flops = K * (2* Np * Nafp + Np) # including jacobian mult
        flux_flops = K * Nafp * (
            3 # jumps
            + 15 # actual upwind flux
            )

        return (6*rs_diff_flops  # each component in r and s
            + l2g_diff_flops
            + 3*lift_flops # each component
            + flux_flops)

# }}}

# vim: foldmethod=marker
