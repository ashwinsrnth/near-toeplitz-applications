import sys
import numpy as np
from mpi4py import MPI
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from numpy.testing import *

from neato import NearToeplitzSolver

import parallelcompact.init
from parallelcompact.da import DA, DA_arange
from parallelcompact.rhs import compute_RHS
from parallelcompact.misc import solve_secondary_systems
from parallelcompact.reduced import get_params
from parallelcompact.sum import sum_solutions
from parallelcompact.permute import permute

class Timer:
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()

    def tic(self):
        cuda.Context.synchronize()
        self.comm.Barrier()
        self.t1 = MPI.Wtime()

    def toc(self):
        cuda.Context.synchronize()
        self.comm.Barrier()
        self.t2 = MPI.Wtime()
        return self.t2 - self.t1

N = int(sys.argv[1]) # local size per dim
pz = int(sys.argv[2]) # procs per dim
py = int(sys.argv[3])
px = int(sys.argv[4])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
da = DA(comm, (N, N, N), (pz, py, px), 1)
x, y, z = DA_arange(da, (0, 1), (0, 1), (0, 1))
f = x*y*z
dx = x[0, 0, 1] - x[0, 0, 0]
dy = y[0, 1, 0] - y[0, 0, 0]
dz = z[1, 0, 0] - z[0, 0, 0]

f_d = gpuarray.to_gpu(f)
x_d = da.create_global_vector()
y_d = da.create_global_vector()
z_d = da.create_global_vector()
f_local_d = da.create_local_vector()

nsteps = 100
timer = Timer(comm)

# dfdx
line_da = da.get_line_DA(0)
line_rank = line_da.rank
line_size = line_da.size

coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
if line_rank == 0:
    coeffs[1] = 2
if line_rank == line_size-1:
    coeffs[-2] = 2

solver = NearToeplitzSolver(N, N*N, coeffs)

for step in range(nsteps+1):
    if step == 1:
        timer.tic()
    line_da.global_to_local(f_d, f_local_d)
    compute_RHS(f_local_d, x_d, dx, (N, N, N), line_rank, line_size)
    xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size, coeffs)
    solver.solve(x_d)
    alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
    sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
t = timer.toc()
if rank == 0: print 'dfdx: ', t*1000/nsteps

dfdx = x_d.get()
assert_allclose(dfdx, y*z)

# dfdy
line_da = da.get_line_DA(1)
line_rank = line_da.rank
line_size = line_da.size
coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
if line_rank == 0:
    coeffs[1] = 2
if line_rank == line_size-1:
    coeffs[-2] = 2

solver = NearToeplitzSolver(N, N*N, coeffs)

for step in range(nsteps+1):
    if step == 1:
        timer.tic()
    permute(f_d, x_d, (0, 2, 1))
    line_da.global_to_local(x_d, f_local_d)
    compute_RHS(f_local_d, x_d, dy, (N, N, N), line_rank, line_size)
    xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size, coeffs)
    solver.solve(x_d)
    alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
    sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
    permute(x_d, y_d, (0, 2, 1))
t = timer.toc()
if rank == 0: print 'dfdy: ', t*1000/nsteps

dfdy = y_d.get()
assert_allclose(dfdy, x*z)

# dfdz
line_da = da.get_line_DA(2)
line_rank = line_da.rank
line_size = line_da.size
coeffs = [1., 1./4, 1./4, 1., 1./4, 1./4, 1.]
if line_rank == 0:
    coeffs[1] = 2
if line_rank == line_size-1:
    coeffs[-2] = 2

solver = NearToeplitzSolver(N, N*N, coeffs)

for step in range(nsteps+1):
    if step == 1:
        timer.tic()
    permute(f_d, x_d, (1, 2, 0))
    line_da.global_to_local(x_d, f_local_d)
    compute_RHS(f_local_d, x_d, dz, (N, N, N), line_rank, line_size)
    xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size, coeffs)
    solver.solve(x_d)
    alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
    sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
    permute(x_d, z_d, (2, 0, 1))
t = timer.toc()
if rank == 0: print 'dfdz: ', t*1000/nsteps

dfdz = z_d.get()
assert_allclose(dfdz, x*y)

