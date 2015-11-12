import sys
import numpy as np
from numpy import sin, cos
from scipy.linalg import solve_banded
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

from neato import NearToeplitzSolver
from compact.rhs import compute_rhs
from compact.permute import permute

def tridiagonal_solve(a, b, c, rhs):
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

def fun(x, y, z):
    return sin(x) + 2*sin(y) + 3*sin(z)

N = int(sys.argv[1])
L = 2*np.pi
dx = L/(N-1)
z, y, x = np.meshgrid(np.linspace(0, L, N),
        np.linspace(0, L, N),
        np.linspace(0, L, N),
        indexing='ij')
f = fun(x, y, z)
f_d = gpuarray.to_gpu(f)
dfdx_d = gpuarray.zeros(f.shape, dtype=np.float64)
dfdy_d = gpuarray.zeros(f.shape, dtype=np.float64)
dfdz_d = gpuarray.zeros(f.shape, dtype=np.float64)
tmp_d = gpuarray.zeros(f.shape, dtype=np.float64)

# dfdx:
solver = NearToeplitzSolver(N, N*N, (1., 2., 1./4, 1., 1./4, 2., 1.))

start = cuda.Event()
end = cuda.Event()

# dfdx:
nsteps = 100

total_time = 0
for i in range(nsteps):
    start.record()
    compute_rhs(f_d, dfdx_d, dx)
    solver.solve(dfdx_d)
    end.record()
    end.synchronize()
    total_time += start.time_till(end)

# dfdy:
total_time = 0
for i in range(nsteps):
    start.record()
    permute(f_d, dfdy_d, (0, 2, 1))
    compute_rhs(dfdy_d, tmp_d, dx)
    solver.solve(tmp_d)
    permute(tmp_d, dfdy_d, (0, 2, 1))
    end.record()
    end.synchronize()
    total_time += start.time_till(end)

# dfdz:
total_time = 0
for i in range(nsteps):
    start.record()
    permute(f_d, dfdz_d, (1, 2, 0))
    compute_rhs(dfdz_d, tmp_d, dx)
    solver.solve(tmp_d)
    permute(tmp_d, dfdz_d, (2, 0, 1))
    end.record()
    end.synchronize()
    total_time += start.time_till(end)

dfdx_true = cos(x)
dfdy_true = 2*cos(y)
dfdz_true = 3*cos(z)
dfdx = dfdx_d.get()
dfdy = dfdy_d.get()
dfdz = dfdz_d.get()

from numpy.testing import assert_allclose
print 'dfdx err: ', np.mean(np.abs(dfdx-dfdx_true))
print 'dfdy err: ', np.mean(np.abs(dfdy-dfdy_true))
print 'dfdz err: ', np.mean(np.abs(dfdz-dfdz_true))
