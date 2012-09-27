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
import numpy.linalg as la





def main():
    from optparse import OptionParser
    parser = OptionParser(usage="Usage: %prog [options] <mesh.neu>")
    parser.add_option("--loopy", action="store_true",
            help="use loopy")
    parser.add_option("-v", "--vis-every", type="int", metavar="S",
            help="visualize on-line every S steps")
    parser.add_option("-i", "--ic", metavar="NAME",
            help="use initial condition NAME (try 'help')",
            default="gaussian")
    parser.add_option("-t", "--final-time", metavar="T",
            help="set final time", type="float",
            default=5)
    parser.add_option("-n", metavar="N", type="int", default=4,
            help="use polynomial degree N")

    options, args = parser.parse_args()
    if not args:
        parser.print_help()
        return

    from pydgeon.local import LocalDiscretization3D, JacobiGQ
    from pydgeon.runge_kutta import integrate_in_time
    from pydgeon.tools import make_obj_array
    import pydgeon

    ldis = LocalDiscretization3D(N=options.n)
    print "loading mesh"
    mesh = pydgeon.read_3d_gambit_mesh(args[0])

    print "building discretization"
    d = pydgeon.Discretization3D(ldis, *mesh)

    from pydgeon.visualize import Visualizer
    vis = Visualizer(d)

    print "%d elements" % d.K

    # set initial conditions
    if options.ic == "sine":
        m_mode, n_mode, o_mode = 3, 2, 4
        ux = np.zeros((d.K, d.ldis.Np))
        uy = np.zeros((d.K, d.ldis.Np))
        uz = np.zeros((d.K, d.ldis.Np))
        pr = np.sin(m_mode*np.pi*d.x)*np.sin(n_mode*np.pi*d.y)*np.sin(n_mode*np.pi*d.z)
    else:
        print "available ICs: sine"
        return

    state = make_obj_array([ux, uy, uz, pr])
    if 0:
        state = make_obj_array([d.to_dev(x) for x in state])

    # compute time step size
    dt = 1e-5

    # setup

    if options.loopy:
        from pydgeon.acoustics3d import LoopyAcousticsRHS3D
        import pyopencl as cl
        import pyopencl.array
        ctx = cl.create_some_context()
        profile = True

        if profile:
            queue = cl.CommandQueue(ctx,
                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            queue = cl.CommandQueue(ctx)

        dtype = np.float32

        from pydgeon import CLDiscretizationInfo3D
        cl_info = CLDiscretizationInfo3D(queue, d, dtype)

        lpy_rhs = LoopyAcousticsRHS3D(queue, cl_info, dtype=dtype)

        state = make_obj_array([
            cl.array.to_device(queue,x).astype(dtype) for x in state])

        def rhs(t, state):
            return make_obj_array(lpy_rhs(queue, *state))
    else:
        from pydgeon.acoustics3d import AcousticsRHS3D

        def rhs(t, state):
            return make_obj_array(AcousticsRHS3D(d, *state))

    def vis_hook(step, t, state):
        if options.vis_every and step % options.vis_every == 0:
            print la.norm(state[-1])
            vis.write_vtk("out-%04d.vtu" % step,
                    [
                        ("pressure", state[-1])
                        ]
                    )

        from time import time as wall_time
        progress_every = 20
        start_timing_at_step = progress_every
        if step % 20 == 0:
            if step == start_timing_at_step:
                start_time[0] = wall_time()
            elif step > start_timing_at_step:
                elapsed = wall_time()-start_time[0]
                timed_steps = step - start_timing_at_step
                time_per_step = elapsed/timed_steps

                line = ("step=%d, sim_time=%f, elapsed wall time=%.2f s,"
                        "time per step=%f s" % (
                        step, t, elapsed, time_per_step))

                print line

                for evt in cl_info.volume_events:
                    evt.wait()
                for evt in cl_info.surface_events:
                    evt.wait()
                vol_time = 1e-9*sum(
                    evt.profile.END-evt.profile.START
                    for evt in cl_info.volume_events)/len(cl_info.volume_events)
                surf_time = 1e-9*sum(
                    evt.profile.END-evt.profile.START
                    for evt in cl_info.surface_events)/len(cl_info.surface_events)

                print "volume: %.4g GFlops/s time/step: %.3g s" % (
                        lpy_rhs.volume_flops/vol_time*1e-9,
                        vol_time)
                print "surface: %.4g GFlops/s time/step: %.3g s" % (
                        lpy_rhs.surface_flops/surf_time*1e-9,
                        surf_time)

                del cl_info.volume_events[:]

    # time loop
    print "entering time loop"
    start_time = [0]
    time, final_state = integrate_in_time(state, rhs, dt,
            final_time=options.final_time, vis_hook=vis_hook)




if __name__ == "__main__":
    main()
