from mpi4py import MPI
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy.linalg import solve_banded
from gpuDA import *
import os
import time
from timer import *

import kernels
from reduced import *
import solvers.globalmem.near_toeplitz
import solvers.templated.near_toeplitz

class CompactFiniteDifferenceSolver:

    def __init__(self, line_da, solver='templated'):
        '''
        :param line_da: DA object carrying the grid information along
            the line
        :type line_da: gpuDA.DA
        '''
        self.line_da = line_da
        self.solver = solver
        self.init_cu()
        self.init_solvers()

    def dfdx(self, f_d, dx, x_d, f_local_d):
        '''
        :param f_d: The 3-d array with function values
        :type f_d: GPUArray
        :param dx: Spacing in x-direction
        :type dx: float
        :param x_d: Space for solution
        :type x_d: GPUArray
        :param f_local_d: Space for function values and ghost elements
        :type f_local_d: GPUArray
        '''
        self.compute_RHS(f_d, dx, x_d, f_local_d)
        x_UH_d, x_LH_d = self.solve_secondary_systems()
        self.solve_primary_system(x_d)
        alpha_d, beta_d = self.solve_reduced_system(x_UH_d, x_LH_d, x_d)
        self.sum_solutions(x_UH_d, x_LH_d, x_d, alpha_d, beta_d)
    
    @timeit
    def compute_RHS(self, f_d, dx, x_d, f_local_d):
        self.line_da.global_to_local(f_d, f_local_d)
        self.compute_RHS_kernel.prepared_call((self.line_da.nx/8, self.line_da.ny/8, self.line_da.nz/8), (8, 8, 8),
                    f_local_d.gpudata, x_d.gpudata, np.float64(dx),
                        np.int32(self.line_da.rank), np.int32(self.line_da.size))
    @timeit 
    def sum_solutions(self, x_UH_d, x_LH_d, x_R_d, alpha_d, beta_d):
        self.sum_solutions_kernel.prepared_call(
                (self.line_da.nx/8, self.line_da.ny/8, self.line_da.nz/8),
                    (8, 8, 8),
                        x_R_d.gpudata, x_UH_d.gpudata,
                        x_LH_d.gpudata, alpha_d.gpudata, beta_d.gpudata,
                            np.int32(self.line_da.nx),
                            np.int32(self.line_da.ny),
                            np.int32(self.line_da.nz))
    @timeit
    def solve_primary_system(self, x_d):
        self._primary_solver.solve(x_d)
    @timeit
    def solve_reduced_system(self, x_UH_d, x_LH_d, x_R_d):
        x_UH = x_UH_d.get()
        x_LH = x_LH_d.get()

        nz, ny, nx = self.line_da.nz, self.line_da.ny, self.line_da.nx
        line_rank = self.line_da.rank
        line_size = self.line_da.size
        
        x_UH_line = np.zeros(2*line_size, dtype=np.float64)
        x_LH_line = np.zeros(2*line_size, dtype=np.float64)

        self.line_da.gather(
                [np.array([x_UH[0], x_UH[-1]]), 2, MPI.DOUBLE],
                [x_UH_line, 2, MPI.DOUBLE])
        self.line_da.gather(
                [np.array([x_LH[0], x_LH[-1]]), 2, MPI.DOUBLE],
                [x_LH_line, 2, MPI.DOUBLE])

        
        x_R_faces_d = gpuarray.zeros((2, nz, ny), np.float64)
        
        self.copy_faces_kernel.prepared_call((ny/16, nz/16, 1), (16, 16, 1),
                x_R_d.gpudata, x_R_faces_d.gpudata,
                    np.int32(nx), np.int32(ny), np.int32(nz),
                        np.int32(self.line_da.mx), np.int32(self.line_da.npx))
        x_R_faces_line_d = gpuarray.zeros((2*line_size, nz, ny), dtype=np.float64)

        self.line_da.gather([x_R_faces_d.gpudata.as_buffer(x_R_faces_d.nbytes), 2*nz*ny, MPI.DOUBLE],
                [x_R_faces_line_d.gpudata.as_buffer(x_R_faces_line_d.nbytes), 2*nz*ny, MPI.DOUBLE])

        if line_rank == 0:
            a_reduced = np.zeros(2*line_size, dtype=np.float64)
            b_reduced = np.zeros(2*line_size, dtype=np.float64)
            c_reduced = np.zeros(2*line_size, dtype=np.float64)
            a_reduced[0::2] = -1.
            a_reduced[1::2] = x_UH_line[1::2]
            b_reduced[0::2] = x_UH_line[0::2]
            b_reduced[1::2] = x_LH_line[1::2]
            c_reduced[0::2] = x_LH_line[0::2]
            c_reduced[1::2] = -1.
            a_reduced[0], c_reduced[0] = 0.0, 0.0
            b_reduced[0] = 1.0
            a_reduced[-1], c_reduced[-1] = 0.0, 0.0
            b_reduced[-1] = 1.0
            a_reduced[1] = 0.
            c_reduced[-2] = 0.

            a_reduced_d = gpuarray.to_gpu(a_reduced)
            b_reduced_d = gpuarray.to_gpu(b_reduced)
            c_reduced_d = gpuarray.to_gpu(c_reduced)
            c2_reduced_d = gpuarray.to_gpu(c_reduced)

            self._reduced_solver.solve(a_reduced_d, b_reduced_d,
                    c_reduced_d, c2_reduced_d, x_R_faces_line_d)

        self.line_da.scatter([x_R_faces_line_d.gpudata.as_buffer(x_R_faces_line_d.nbytes), 2*nz*ny, MPI.DOUBLE],
                [x_R_faces_d.gpudata.as_buffer(x_R_faces_d.nbytes), 2*nz*ny, MPI.DOUBLE])

        alpha_d = x_R_faces_d[0, :, :]
        beta_d = x_R_faces_d[1, :, :]
        return alpha_d, beta_d
    @timeit   
    def solve_secondary_systems(self):
        nz, ny, nx = self.line_da.nz, self.line_da.ny, self.line_da.nx
        line_rank = self.line_da.rank
        line_size = self.line_da.size

        a = np.ones(nx, dtype=np.float64)*(1./4)
        b = np.ones(nx, dtype=np.float64)
        c = np.ones(nx, dtype=np.float64)*(1./4)
        r_UH = np.zeros(nx, dtype=np.float64)
        r_LH = np.zeros(nx, dtype=np.float64)

        if line_rank == 0:
            c[0] =  2.0
            a[0] = 0.0

        if line_rank == line_size-1:
            a[-1] = 2.0
            c[-1] = 0.0

        r_UH[0] = -a[0]
        r_LH[-1] = -c[-1]

        x_UH = scipy_solve_banded(a, b, c, r_UH)
        x_LH = scipy_solve_banded(a, b, c, r_LH)
        x_UH_d = gpuarray.to_gpu(x_UH)
        x_LH_d = gpuarray.to_gpu(x_LH)
        return x_UH_d, x_LH_d

    def setup_reduced_solver(self):
       return ReducedSolver((2*self.line_da.npx, self.line_da.nz, self.line_da.ny))

    def setup_primary_solver(self):
        line_rank = self.line_da.rank
        line_size = self.line_da.size
        coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
        if line_rank == 0:
            coeffs[1] = 2.
        if line_rank == line_size-1:
            coeffs[-2] = 2.
        
        if self.solver == 'globalmem':
            return solvers.globalmem.near_toeplitz.NearToeplitzSolver(
                    (self.line_da.nz, self.line_da.ny, self.line_da.nx), coeffs)
        else:
            return solvers.templated.near_toeplitz.NearToeplitzSolver(
                    (self.line_da.nz, self.line_da.ny, self.line_da.nx), coeffs)
        
    def init_cu(self):
        thisdir = os.path.dirname(os.path.realpath(__file__))
        self.compute_RHS_kernel, self.sum_solutions_kernel, self.copy_faces_kernel, = kernels.get_funcs(
                thisdir + '/' + 'kernels.cu', 'computeRHS', 'sumSolutions', 'negateAndCopyFaces')
        self.compute_RHS_kernel.prepare('PPdii')
        self.sum_solutions_kernel.prepare('PPPPPiii')
        self.copy_faces_kernel.prepare('PPiiiii')
        self.start = cuda.Event()
        self.end = cuda.Event()

    def init_solvers(self):
        self._primary_solver = self.setup_primary_solver()
        self._reduced_solver = self.setup_reduced_solver()

def scipy_solve_banded(a, b, c, rhs):
    '''
    Solve the tridiagonal system described
    by a, b, c, and rhs.
    a: lower off-diagonal array (first element ignored)
    b: diagonal array
    c: upper off-diagonal array (last element ignored)
    rhs: right hand side of the system
    '''
    l_and_u = (1, 1)
    ab = np.vstack([np.append(0, c[:-1]),
                    b,
                    np.append(a[1:], 0)])
    x = solve_banded(l_and_u, ab, rhs)
    return x
