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




class Visualizer:
    def __init__(self, discr):
        self.discr = discr

        submesh_indices = np.array(list(discr.ldis.get_submesh_indices()))

        Np = discr.ldis.Np
        el_starts = np.arange(0, discr.K*Np, Np)
        self.vis_simplices = (el_starts[:,np.newaxis,np.newaxis]
                + submesh_indices[np.newaxis,:,:]).reshape(-1)

        self.nsimplices_per_el = len(submesh_indices)

        self.nodes = np.array([discr.x, discr.y, discr.z]).reshape(3, -1)

    def write_vtk(self, file_name, fields, compressor=None):
        from pyvisfile.vtk import (
                UnstructuredGrid, DataArray,
                AppendedDataXMLGenerator,
                VTK_TETRA, VF_LIST_OF_VECTORS, VF_LIST_OF_COMPONENTS)

        grid = UnstructuredGrid(
                (self.nodes.shape[-1],
                    DataArray("points", self.nodes,
                        vector_format=VF_LIST_OF_COMPONENTS)),
                cells=self.vis_simplices,
                cell_types=np.asarray([VTK_TETRA]
                    * self.discr.K*self.nsimplices_per_el,
                    dtype=np.uint8))

        for name, field in fields:
            grid.add_pointdata(DataArray(name, field.reshape(-1),
                vector_format=VF_LIST_OF_COMPONENTS))

        from os.path import exists
        if exists(file_name):
            raise RuntimeError("output file '%s' already exists"
                    % file_name)

        outf = open(file_name, "w")
        AppendedDataXMLGenerator(compressor)(grid).write(outf)
        outf.close()








