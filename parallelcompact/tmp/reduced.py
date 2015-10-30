import numpy as np
import kernels
import os

class ReducedSolver:
    def __init__(self, shape):
        '''
        Create context for pThomas (thread-parallel Thomas algorithm)
        '''
        self.nz, self.ny, self.nx = shape 
        thisdir = os.path.dirname(os.path.realpath(__file__))
        self.solver, = kernels.get_funcs(thisdir + '/' + 'kernels.cu', 'reducedSolverKernel')
        self.solver.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intc, np.intc, np.intc])

    def solve(self, a_d, b_d, c_d, c2_d, x_d):
        self.solver.prepared_call((self.nx/16, self.ny/16, 1), (16, 16, 1),
             a_d.gpudata, b_d.gpudata, c_d.gpudata, c2_d.gpudata, x_d.gpudata,
                np.int32(self.nx), np.int32(self.ny), np.int32(self.nz))
