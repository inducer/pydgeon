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

import pydgeon
from pydgeon.local import LocalDiscretization2D, JacobiGQ
from pydgeon.runge_kutta import integrate_in_time
from pydgeon.tools import make_obj_array




def main(use_cl=True, visualize=True):
    import sys

    if len(sys.argv) == 1:
        print "Usage: %s <mesh.neu>" % sys.argv[0]
        return

    ldis = LocalDiscretization2D(N=9)
    mesh = pydgeon.read_2d_gambit_mesh(sys.argv[1])

    if use_cl:
        from pydgeon.opencl import CLDiscretization2D
        d = CLDiscretization2D(ldis, *mesh)
    else:
        d = pydgeon.Discretization2D(ldis, *mesh)

    # set initial conditions
    mmode = 3; nmode = 2
    Hx = np.zeros((d.K, d.ldis.Np))
    Hy = np.zeros((d.K, d.ldis.Np))
    Ez = np.sin(mmode*np.pi*d.x)*np.sin(nmode*np.pi*d.y)

    state = make_obj_array([Hx, Hy, Ez])
    if use_cl:
        state = make_obj_array([d.to_dev(x) for x in state])

    # compute time step size
    rLGL = JacobiGQ(0,0, d.ldis.N)[0]
    rmin = abs(rLGL[0]-rLGL[1])
    dt_scale = d.dt_scale()
    dt = dt_scale.min()*rmin*2/3

    # setup
    if visualize:
        try:
            import enthought.mayavi.mlab as mayavi
        except ImportError:
            visualize = False

    if visualize:
        vis_mesh = mayavi.triangular_mesh(
                d.x.ravel(), d.y.ravel(), Ez.ravel(),
                d.gen_vis_triangles())

    def vis_hook(step, t, state):
        if use_cl:
            Hx, Hy, Ez = [d.from_dev(x) for x in state]
        else:
            Hx, Hy, Ez = state

        if step % 10 == 0 and visualize:
            vis_mesh.mlab_source.z = Ez.ravel()

        print la.norm(Ez)

    if use_cl:
        from pydgeon.maxwell import CLMaxwellsRhs2D
        inner_rhs = CLMaxwellsRhs2D(d)

        def rhs(t, state):
            return make_obj_array(inner_rhs(*state))
    else:
        from pydgeon.maxwell import MaxwellRHS2D

        def rhs(t, state):
            return make_obj_array(MaxwellRHS2D(d, *state))

    # time loop
    time, final_state = integrate_in_time(state, rhs, dt, final_time=5,
            vis_hook=vis_hook)




if __name__ == "__main__":
    main()
