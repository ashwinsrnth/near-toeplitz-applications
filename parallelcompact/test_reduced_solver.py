from reduced import solve_reduced
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose

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

nx = 32
ny = 32
nz = 4
a = np.random.rand(nz)
b = np.random.rand(nz)
c = np.random.rand(nz)
d = np.random.rand(nz, ny, nx)

a_d = gpuarray.to_gpu(a)
b_d = gpuarray.to_gpu(b)
c_d = gpuarray.to_gpu(c)
d_d = gpuarray.to_gpu(d)
c2_d = gpuarray.to_gpu(c)

solve_reduced(a_d, b_d, c_d, c2_d, d_d, (nz, ny, nx))
x = d_d.get()

for i in range(ny):
    for j in range(nx):
        x_true = tridiagonal_solve(a, b, c, d[:, i, j])
        assert_allclose(x_true, x[:, i, j])
