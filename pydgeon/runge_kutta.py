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


# {{{ Runge-Kutta coefficients

rk4a = np.array([0,
        -567301805773/1357537059087,
        -2404267990393/2016746695238,
        -3550918686646/2091501179385,
        -1275806237668/842570457699])
rk4b = [1432997174477/9575080441755,
         5161836677717/13612068292357,
         1720146321549/2090206949498,
         3134564353537/4481467310338,
         2277821191437/14882151754819]
rk4c = [0,
         1432997174477/9575080441755,
         2526269341429/6820363962896,
         2006345519317/3224310063776,
         2802321613138/2924317926251]

# }}}


def integrate_in_time(state, rhs_func, dt, final_time, vis_hook=None):
    time = 0
    step = 0

    residual = 0*state

    # outer time step loop
    while time < final_time:
        if time+dt > final_time:
            dt = final_time-time

        for a, b in zip(rk4a, rk4b):
            rhs = rhs_func(time, state)
            residual = a*residual + dt*rhs
            state = state + b*residual

        if vis_hook is not None:
            vis_hook(step, time, state)

        # Increment time
        time = time+dt
        step += 1

    return time, state


def integrate_in_time_cl(context, dtype, state, rhs_func, dt, final_time,
        vis_hook=None):
    time = 0
    step = 0

    residual = 0*state

    from pyopencl.elementwise import ElementwiseKernel, VectorArg, ScalarArg
    from pytools.obj_array import as_oarray_func_n_args
    axpby_knl = ElementwiseKernel(context, [
        ScalarArg(dtype, "a"),
        VectorArg(dtype, "x"),
        ScalarArg(dtype, "b"),
        VectorArg(dtype, "y"),
        VectorArg(dtype, "z"),
        ],
        "z[i] = a*x[i] + b*y[i]")

    # The decorator module won't work on callable objects. D'oh.
    def axpby_wrapper(*args):
        return axpby_knl(*args)

    axpby = as_oarray_func_n_args(axpby_wrapper)
    # outer time step loop
    while time < final_time:
        if time+dt > final_time:
            dt = final_time-time

        for a, b in zip(rk4a, rk4b):
            rhs = rhs_func(time, state)

            # residual = a*residual + dt*rhs
            axpby(a, residual, dt, rhs, residual)

            # state = state + b*residual
            axpby(1, state, b, residual, state)

        if vis_hook is not None:
            vis_hook(step, time, state)

        # Increment time
        time = time+dt
        step += 1

    return time, state

# vim: foldmethod=marker
