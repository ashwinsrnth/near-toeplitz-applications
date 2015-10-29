import numpy as np
from scipy.linalg import solve_banded
from pycuda import autoinit
import pycuda.gpuarray as gpuarray

from neato import NearToeplitzSolver
from compact.rhs import compute_rhs

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
    return np.sin(x)

N = 32
L = 2*np.pi
dx = L/(N-1)
z, y, x = np.meshgrid(np.linspace(0, L, N),
        np.linspace(0, L, N),
        np.linspace(0, L, N),
        indexing='ij')
f = fun(x, y, z)
f_d = gpuarray.to_gpu(f)
d_d = gpuarray.zeros(f.shape, dtype=np.float64)
nz, ny, nx = f.shape
solver = NearToeplitzSolver(nx, nz*ny, (1., 2., 1./4, 1., 1./4, 2., 1.))
compute_rhs(f_d, d_d, dx)
solver.solve(d_d)

dfdx = d_d.get()
import matplotlib.pyplot as plt
plt.plot(x[0, 0, :], f[0, 0, :])
plt.plot(x[0, 0, :], dfdx[0, 0, :])
plt.savefig('deriv.png')
