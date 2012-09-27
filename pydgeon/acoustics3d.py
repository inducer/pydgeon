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

    dUx = Ux.flat[vmapP]-Ux.flat[vmapM]
    dUy = Uy.flat[vmapP]-Uy.flat[vmapM]
    dUz = Uz.flat[vmapP]-Uz.flat[vmapM]
    dPr = Pr.flat[vmapP]-Pr.flat[vmapM]

    # Impose reflective boundary conditions (Uz+ = -Uz-)
    dUx.flat[d.mapB] = 0
    dUy.flat[d.mapB] = 0
    dUz.flat[d.mapB] = 0
    dPr.flat[d.mapB] =-2*Pr.flat[d.vmapB]

    # evaluate upwind fluxes
    alpha  = 1.0
    R = dPr - d.nx*dUx - d.ny*dUy - d.nz*dUz
    fluxUx = -d.nx*R
    fluxUy = -d.ny*R
    fluxUz = -d.nz*R
    fluxPr =     R

    # local derivatives of fields
    dPrdx, dPrdy, dPrdz = d.gradient(Pr)
    divU = d.divergence(Ux, Uy, Uz)

    # compute right hand sides of the PDE's
    rhsUx = -dPrdx  + eldot(l.LIFT, (d.Fscale*fluxUx))/2.0
    rhsUy = -dPrdx  + eldot(l.LIFT, (d.Fscale*fluxUy))/2.0
    rhsUz = -dPrdz  + eldot(l.LIFT, (d.Fscale*fluxUz))/2.0
    rhsPr = -divU   + eldot(l.LIFT, (d.Fscale*fluxPr))/2.0

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
        import pyopencl.array

        dtype4 = cl.array.vec.types[np.dtype(dtype), 4]

        ldis = discr.ldis

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

                # volume flux
                rhsu[k,n] = dot(drst_dx[k],dp_drst)
                rhsv[k,n] = dot(drst_dy[k],dp_drst)
                rhsw[k,n] = dot(drst_dz[k],dp_drst)
                rhsp[k,n] = dot(drst_dx[k], du_drst) + dot(drst_dy[k], dv_drst) \
                    + dot(drst_dz[k], dw_drst)
                """,
            [
                lp.GlobalArg("u,v,w,p,rhsu,rhsv,rhsw,rhsp",
                    dtype, shape="K, Np", order="C"),
                lp.GlobalArg("DrDsDt", dtype4, shape="Np, Np", order="C"),
                lp.GlobalArg("drst_dx,drst_dy,drst_dz", dtype4, shape="K"),
                lp.ValueArg("K", np.int32, approximately=1000),
                ],
            name="dg_volume", assumptions="K>=1",
            defines=dict(Np=discr.ldis.Np))

        def transform_vol(knl):
            knl = lp.tag_inames(knl, dict(n="l.0"))
            #knl = lp.change_arg_to_image(knl, "DrDsDt")

            knl = lp.split_iname(knl, "k", 3, outer_tag="g.0", inner_tag="l.1")
            for name in ["u", "v", "w", "p"]:
                knl = lp.add_prefetch(knl, "%s[k,:]" % name, ["k_inner"])

            return knl

        volume_kernel = transform_vol(volume_kernel)
        self.c_volume_kernel = lp.CompiledKernel(context, volume_kernel)
        self.c_volume_kernel.print_code()

        self.volume_flops = (discr.K
                * 4 # num components
                *(
                    3*discr.ldis.Np**2*2
                    + 3*discr.ldis.Np*2
                    ))

        # }}}

        # {{{ surface kernel
        NfpNfaces=ldis.Nfaces*ldis.Nfp

        surface_kernel = lp.make_kernel(context.devices[0],
                ["{[m,n,k]: 0<= m < NfpNfaces and 0<= n < Np and 0<= k < K }"
                    ],
                """
                    <> idP = vmapP[k,m]
                    <> idM = vmapM[k,m]

                    <> du = u[[idP]]-u[[idM]]
                    <> dv = v[[idP]]-v[[idM]]
                    <> dw = w[[idP]]-w[[idM]]
                    <> dp = bc[k,m]*p[[idP]] - p[[idM]]

                    <> dQ = 0.5*Fscale[k,m]* \
                            (dp - nx[k,m]*du - ny[k,m]*dv - nz[k,m]*dw)

                    <> fluxu = -nx[k,m]*dQ
                    <> fluxv = -ny[k,m]*dQ
                    <> fluxw = -nz[k,m]*dQ
                    <> fluxp =          dQ

                    # reduction here
                    rhsu[k,n] = sum(m, LIFT[n,m]*fluxu)
                    rhsv[k,n] = sum(m, LIFT[n,m]*fluxv)
                    rhsw[k,n] = sum(m, LIFT[n,m]*fluxw)
                    rhsp[k,n] = sum(m, LIFT[n,m]*fluxp)
                    """,
                [
                    lp.GlobalArg("vmapP,vmapM",
                        np.int64, shape="K, NfpNfaces", order="C"),
                    lp.GlobalArg("u,v,w,p,rhsu,rhsv,rhsw,rhsp",
                        dtype, shape="K, Np", order="C"),
                    lp.GlobalArg("nx,ny,nz,Fscale,bc",
                        dtype, shape="K, NfpNfaces", order="C"),
                    lp.GlobalArg("LIFT", dtype, shape="Np, NfpNfaces", order="C"),
                    lp.ValueArg("K", np.int32, approximately=1000),
                    ],
                name="dg_surface", assumptions="K>=1",
                defines=dict(Np=ldis.Np, Nfp=ldis.Nfp, NfpNfaces=NfpNfaces)
                )

        surface_kernel = lp.tag_inames(surface_kernel, dict(k="g.0", n="l.0"))

        self.c_surface_kernel = lp.CompiledKernel(context, surface_kernel)
        #self.c_surface_kernel.print_code()

        self.surface_flops = (discr.K
                *(
                    NfpNfaces*15
                    +
                    4*discr.ldis.Np*NfpNfaces*15
                    ))

        # }}}

    def __call__(self, queue, Ux, Uy, Uz, Pr):
        """Evaluate RHS flux in 3D Acoustics."""

        d = self.discr

        cl_info = self.cl_discr_info
        evt, (rhsUx, rhsUy, rhsUz, rhsPr) = self.c_surface_kernel(
                queue,
                vmapP=cl_info.vmapP,
                vmapM=cl_info.vmapM,
                u=Ux, v=Uy, w=Uz, p=Pr,
                nx=cl_info.nx,
                ny=cl_info.ny,
                nz=cl_info.nz,
                Fscale=cl_info.Fscale,
                bc=cl_info.bc,
                LIFT=cl_info.LIFT, K=d.K, warn_numpy=True)

        cl_info.surface_events.append(evt)

        # local derivatives of fields
        evt, (dPrdx, dPrdy, dPrdz, divU) = self.c_volume_kernel(
                queue, u=Ux, v=Uy, w=Uz, p=Pr,
                DrDsDt=cl_info.drdsdt,
                drst_dx=cl_info.drst_dx, drst_dy=cl_info.drst_dy, drst_dz=cl_info.drst_dz,
                K=d.K, warn_numpy=True)

        cl_info.volume_events.append(evt)

        # compute right hand sides of the PDE's
        rhsUx += dPrdx
        rhsUy += dPrdy
        rhsUz += dPrdz
        rhsPr += divU

        return rhsUx, rhsUy, rhsUz, rhsPr

# }}}

# {{{ OpenCL

# {{{ kernels

MAXWELL3D_VOLUME_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define BSIZE %(BSIZE)d

__kernel void MaxwellsVolume3d(int K,
   __read_only __global float *g_Ux,
   __read_only __global float *g_Uy,
   __read_only __global float *g_Uz,
   __read_only __global float *g_Pr,
   __global float *g_rhsUx,
   __global float *g_rhsUy,
   __global float *g_rhsUz,
   __global float *g_rhsPr,
   __read_only __global image3d_t i_DrDsDt,
   __read_only __global float *g_vgeo)
{
  const sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP
    | CLK_FILTER_NEAREST;

  __local float l_Ux[BSIZE];
  __local float l_Uy[BSIZE];
  __local float l_Uz[BSIZE];

  /* LOCKED IN to using Np work items per group */
// start_vol_kernel
  const int n = get_local_id(0);
  const int k = get_group_id(0);

  int m = n+k*BSIZE;
  int id = n;

  l_Ux[id] = g_Ux[m];
  l_Uy[id] = g_Uy[m];
  l_Uz[id] = g_Uz[m];
  l_Pr[id] = g_Pr[m];

  barrier(CLK_LOCAL_MEM_FENCE);

  float dUxdr=0,dUxds=0,dUxdt;
  float dUydr=0,dUyds=0,dUydt;
  float dUzdr=0,dUzds=0,dUzdt;
  float dPrdr=0,dPrds=0,dPrdt;

  float Q;
  for(m=0; m<p_Np; ++m)
  {
    float4 D = read_imagef(i_DrDsDt, samp, (int2)(n, m));

    Q = l_Ux[m]; dUxdr += D.x*Q; dUxds += D.y*Q; dUxdt += D.z*Q;
    Q = l_Uy[m]; dUydr += D.x*Q; dUyds += D.y*Q; dUydt += D.z*Q;
    Q = l_Uz[m]; dUzdr += D.x*Q; dUzds += D.y*Q; dUzdt += D.z*Q;
    Q = l_Pr[m]; dPrdr += D.x*Q; dPrds += D.y*Q; dPrdt += D.z*Q;
  }

  const float drdx = g_vgeo[0+12*k];
  const float drdy = g_vgeo[1+12*k];
  const float drdz = g_vgeo[2+12*k];
  const float dsdx = g_vgeo[3+12*k];
  const float dsdy = g_vgeo[4+12*k];
  const float dsdz = g_vgeo[5+12*k];
  const float dtdx = g_vgeo[6+12*k];
  const float dtdy = g_vgeo[7+12*k];
  const float dtdz = g_vgeo[8+12*k];

  m = n+BSIZE*k;
  g_rhsUx[m] = -(drdx*dPrdr+dsdx*dPrds+dtdx*dPrdt);
  g_rhsUy[m] = -(drdy*dPrdr+dsdy*dPrds+dtdy*dPrdt);
  g_rhsUz[m] = -(drdz*dPrdr+dsdz*dPrds+dtdz*dPrdt);
  g_rhsPr[m] = -(drdx*dUxdr+dsdx*dUxds+dtdx*dUxdt)
               -(drdy*dUydr+dsdy*dUyds+dtdy*dUydt)
               -(drdz*dUzdr+dsdz*dUzds+dtdz*dUzdt) ;

// end
}
"""

MAXWELL3D_SURFACE_KERNEL = """
#define p_N %(N)d
#define p_Np %(Np)d
#define p_Nfp %(Nfp)d
#define p_Nfaces %(Nfaces)d
#define p_Nafp (p_Nfaces*p_Nfp)
#define BSIZE %(BSIZE)d

// start_surf_kernel
__kernel void MaxwellsSurface3d(int K,
  __read_only __global float *g_Ux,
  __read_only __global float *g_Uy,
  __read_only __global float *g_Uz,
  __read_only __global float *g_Pr,
  __global float *g_rhsUx,
  __global float *g_rhsUy,
  __global float *g_rhsUz,
  __global float *g_rhsPr,
  read_only __global float *g_surfinfo,
  __read_only image3d_t i_LIFT)
{
  const sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP
    | CLK_FILTER_NEAREST;

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
    m = 7*(k*p_Nafp)+n;

    const  int   idM = g_surfinfo[m]; m += p_Nafp; // dangerous for large meshes - overflow
    int          idP = g_surfinfo[m]; m += p_Nafp;
    const  float Fsc = g_surfinfo[m]; m += p_Nafp;
    const  float Bsc = g_surfinfo[m]; m += p_Nafp;
    const  float nx  = g_surfinfo[m]; m += p_Nafp;
    const  float ny  = g_surfinfo[m]; m += p_Nafp;
    const  float nz  = g_surfinfo[m]; m += p_Nafp;

    float dUx=0, dUy=0, dUz=0;
    dUx = 0.5f*Fsc*(    g_Ux[idP] - g_Ux[idM]);
    dUy = 0.5f*Fsc*(    g_Uy[idP] - g_Uy[idM]);
    dUz = 0.5f*Fsc*(    g_Uz[idP] - g_Uz[idM]);
    dPr = 0.5f*Fsc*(Bsc*g_Pr[idP] - g_Pr[idM]);

    const float R = dPr - nx*dUx - ny*dUy - nz*dUz;

    l_fluxUx[n] =  nx*R;
    l_fluxUy[n] =  ny*R;
    l_fluxUz[n] =  nz*R;
    l_fluxPr[n] =    -R;
  }

  /* make sure all element data points are cached */
  barrier(CLK_LOCAL_MEM_FENCE);

  if (n < p_Np)
  {
    float rhsUx = 0, rhsUy = 0, rhsUz = 0, rhsPr = 0;
    int col = 0;

    /* can manually unroll to 4 because there are 4 faces */
    for (m=0;m < p_Nfaces*p_Nfp;)
    {
      float4 L = read_imagef(i_LIFT, samp, (int2)(col, n));
      ++col;

      rhsUx += L.x*l_fluxUx[m];
      rhsUy += L.x*l_fluxUy[m];
      rhsUz += L.x*l_fluxUz[m];
      rhsPr += L.x*l_fluxPr[m];
      ++m;

      rhsUx += L.y*l_fluxUx[m];
      rhsUy += L.y*l_fluxUy[m];
      rhsUz += L.y*l_fluxUz[m];
      rhsPr += L.y*l_fluxPr[m];
      ++m;

      rhsUx += L.z*l_fluxUx[m];
      rhsUy += L.z*l_fluxUy[m];
      rhsUz += L.z*l_fluxUz[m];
      rhsPr += L.z*l_fluxPr[m];
      ++m;

      rhsUx += L.w*l_fluxUx[m];
      rhsUy += L.w*l_fluxUy[m];
      rhsUz += L.w*l_fluxUz[m];
      rhsPr += L.w*l_fluxPr[m];
      ++m;
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




class CLMaxwellsRhs3D:
    def __init__(self, discr):
        import pyopencl as cl
        from pydgeon.opencl import CL_OPTIONS

        self.discr = discr

        self.volume_kernel = cl.Program(discr.ctx,
                MAXWELL3D_VOLUME_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "BSIZE": discr.block_size,
                    }
                ).build(options=CL_OPTIONS).MaxwellsVolume3d
        self.volume_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

        self.surface_kernel = cl.Program(discr.ctx,
                MAXWELL3D_SURFACE_KERNEL % {
                    "N": discr.ldis.N,
                    "Np": discr.ldis.Np,
                    "Nfp": discr.ldis.Nfp,
                    "Nfaces": discr.ldis.Nfaces,
                    "BSIZE": discr.block_size,
                    }
                ).build(options=CL_OPTIONS).MaxwellsSurface3d
        self.surface_kernel.set_scalar_arg_dtypes([np.int32] + 8*[None])

        self.flops = self.rhs_flops()

    def __call__(self, Ux, Uy, Uz):
        d = self.discr
        ldis = d.ldis

        rhsUx = d.volume_empty()
        rhsUy = d.volume_empty()
        rhsUz = d.volume_empty()
        rhsPr = d.volume_empty()

        vol_evt = self.volume_kernel(d.queue,
                (d.K*d.block_size,), (d.block_size,),
                d.K,
                Ux.data, Uy.data, Uz.data, Pr.data,
                rhsUx.data, rhsUy.data, rhsUz.data, rhsPr.data,
                d.diffmatrices_img, d.drdx_dev.data)

        surf_block_size = max(
                ldis.Nfp*ldis.Nfaces,
                ldis.Np)

        sfc_evt = self.surface_kernel(d.queue,
                (d.K*surf_block_size,), (surf_block_size,),
                d.K,
                Ux.data, Uy.data, Uz.data, Pr.data,
                rhsUx.data, rhsUy.data, rhsUz.data, rhsPr.data,
                d.surfinfo_dev.data, d.lift_img)

        if d.profile >= 5:
            sfc_evt.wait()
            vol_t = (vol_evt.profile.end - vol_evt.profile.start)*1e-9
            sfc_t = (sfc_evt.profile.end - sfc_evt.profile.start)*1e-9
            print "vol: %g s sfc: %g s" % (vol_t, sfc_t)

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
