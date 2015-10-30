import numpy as np
from mpi4py import MPI
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from neato import NearToeplitzSolver


import parallelcompact.init
from parallelcompact.da import DA, DA_arange
from parallelcompact.rhs import compute_RHS
from parallelcompact.misc import solve_secondary_systems

N = 32
comm = MPI.COMM_WORLD
da = DA(comm, (N, N, N), (2, 2, 2), 1)
x, y, z = DA_arange(da, (0, 1), (0, 1), (0, 1))
f = x*y*z
dfdx_true = y*z
dx = x[0, 0, 1] - x[0, 0, 0]

f_d = gpuarray.to_gpu(f)
x_d = da.create_global_vector()
f_local_d = da.create_local_vector()

# dfdx
line_da = da.get_line_DA(0)
line_rank = line_da.rank
line_size = line_da.size

if line_rank == 0:
    coeffs = (1., 2., 1./4, 1., 1./4, 1./4, 1.)
elif line_rank == line_size-1:
    coeffs = (1., 1./4, 1./4, 1., 1./4, 2., 1.)
else:
    coeffs = (1., 1./4, 1./4, 1., 1./4, 1./4, 1.)

solver = NearToeplitzSolver(N, N*N, coeffs)

da.global_to_local(f_d, f_local_d)
compute_RHS(f_local_d, x_d, dx, (N, N, N), line_rank, line_size)
xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size)
solver.solve(x_d)

