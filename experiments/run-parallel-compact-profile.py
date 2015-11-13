import sys
sys.path.append('..')
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

t_gtol = 0
t_rhs = 0
t_sec = 0
t_primary = 0
t_params = 0
t_sum = 0

xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size, coeffs)

line_da.global_to_local(f_d, f_local_d)
compute_RHS(f_local_d, x_d, dx, (N, N, N), line_rank, line_size)
solver.solve(x_d)
alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))

for step in range(nsteps):
    timer.tic()
    line_da.global_to_local(f_d, f_local_d)
    t = timer.toc()
    t_gtol += t

    timer.tic()
    compute_RHS(f_local_d, x_d, dx, (N, N, N), line_rank, line_size)
    t = timer.toc()
    t_rhs += t

    timer.tic()
    solver.solve(x_d)
    t = timer.toc()

    t_primary += t

    timer.tic()
    alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
    t = timer.toc()

    t_params += t

    timer.tic()
    sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
    t = timer.toc()

    t_sum += t

if rank == 0:
    print '--------- dfdx ----------'
    print 'Global to local: ', t_gtol*1000/nsteps
    print 'RHS: ', t_rhs*1000/nsteps
    print 'Primary: ', t_primary*1000/nsteps
    print 'Params: ', t_params*1000/nsteps
    print 'Sum: ', t_sum*1000/nsteps
    print 'Total: ', (t_gtol+t_rhs+t_primary+t_params+t_sum)*1000/nsteps
    print
    print

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

t_gtol = 0
t_rhs = 0
t_sec = 0
t_primary = 0
t_params = 0
t_sum = 0
t_perm = 0

xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size, coeffs)

permute(f_d, x_d, (0, 2, 1))
line_da.global_to_local(x_d, f_local_d)
compute_RHS(f_local_d, x_d, dy, (N, N, N), line_rank, line_size)
solver.solve(x_d)
alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
permute(x_d, y_d, (0, 2, 1))

for step in range(nsteps):
    timer.tic()
    permute(f_d, x_d, (0, 2, 1))
    t = timer.toc()
    t_perm += t

    timer.tic()
    line_da.global_to_local(x_d, f_local_d)
    t = timer.toc()
    t_gtol += t

    timer.tic()
    compute_RHS(f_local_d, x_d, dy, (N, N, N), line_rank, line_size)
    t = timer.toc()
    t_rhs += t

    timer.tic()
    solver.solve(x_d)
    t = timer.toc()
    t_primary += t
    
    timer.tic()
    alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
    t = timer.toc()
    t_params += t

    timer.tic()
    sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
    t = timer.toc()
    t_sum += t 

    timer.tic()
    permute(x_d, y_d, (0, 2, 1))
    t = timer.toc()
    t_perm += t

if rank == 0:
    print '--------- dfdy ----------'
    print 'Global to local: ', t_gtol*1000/nsteps
    print 'RHS: ', t_rhs*1000/nsteps
    print 'Primary: ', t_primary*1000/nsteps
    print 'Params: ', t_params*1000/nsteps
    print 'Sum: ', t_sum*1000/nsteps
    print 'Permute ', t_perm*1000/nsteps
    print 'Total: ', (t_gtol+t_rhs+t_primary+t_params+t_sum+t_perm)*1000/nsteps
    print
    print

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

t_gtol = 0
t_rhs = 0
t_sec = 0
t_primary = 0
t_params = 0
t_sum = 0
t_perm = 0

permute(f_d, x_d, (1, 2, 0))
line_da.global_to_local(x_d, f_local_d)
compute_RHS(f_local_d, x_d, dz, (N, N, N), line_rank, line_size)
solver.solve(x_d)
alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
permute(x_d, z_d, (2, 0, 1))

xU_d, xL_d = solve_secondary_systems(N, line_rank, line_size, coeffs)
for step in range(nsteps):

    timer.tic()
    permute(f_d, x_d, (1, 2, 0))
    t = timer.toc()
    t_perm += t

    timer.tic()
    line_da.global_to_local(x_d, f_local_d)
    t = timer.toc()
    t_gtol += t

    timer.tic()
    compute_RHS(f_local_d, x_d, dz, (N, N, N), line_rank, line_size)
    t = timer.toc()
    t_rhs += t

    timer.tic()
    solver.solve(x_d)
    t = timer.toc()
    t_primary += t
    
    timer.tic()
    alpha_d, beta_d = get_params(line_da, xU_d, xL_d, x_d)
    t = timer.toc()
    t_params += t

    timer.tic()
    sum_solutions(xU_d, xL_d, x_d, alpha_d, beta_d, (N, N, N))
    t = timer.toc()
    t_sum += t 

    timer.tic()
    permute(x_d, z_d, (2, 0, 1))
    t = timer.toc()
    t_perm += t

if rank == 0:
    print '--------- dfdz ----------'
    print 'Global to local: ', t_gtol*1000/nsteps
    print 'RHS: ', t_rhs*1000/nsteps
    print 'Primary: ', t_primary*1000/nsteps
    print 'Params: ', t_params*1000/nsteps
    print 'Sum: ', t_sum*1000/nsteps
    print 'Permute ', t_perm*1000/nsteps
    print 'Total: ', (t_gtol+t_rhs+t_primary+t_params+t_sum+t_perm)*1000/nsteps
    print
    print

dfdz = z_d.get()
assert_allclose(dfdz, x*y)

