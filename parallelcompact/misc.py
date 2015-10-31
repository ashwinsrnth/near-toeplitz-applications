import numpy as np
from scipy.linalg import solve_banded
import pycuda.gpuarray as gpuarray

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

def solve_secondary_systems(nx, rank, size):
    a = np.ones(nx, dtype=np.float64)
    b = np.ones(nx, dtype=np.float64)
    c = np.ones(nx, dtype=np.float64)
    rU = np.zeros(nx, dtype=np.float64)
    rL = np.zeros(nx, dtype=np.float64)

    if rank == 0:
        c[0] = 2.0
        a[0] = 0.0

    if rank == size-1:
        a[-1] = 2.0
        c[-1] = 0.0

    rU[0] = -a[0]
    rL[-1] = -c[-1]

    xU = tridiagonal_solve(a, b, c, rU)
    xL = tridiagonal_solve(a, b, c, rL)
    xU_d = gpuarray.to_gpu(xU)
    xL_d = gpuarray.to_gpu(xL)
    return xU_d, xL_d
