import numpy as np
from near_toeplitz import *
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose

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

nz = 32
ny = 32
nx = 32
d = np.random.rand(nz, ny, nx)
x = d.copy()
solver = NearToeplitzSolver(nx, ny*nz, (1., 2., 1./4, 1., 1./4, 2., 1.))
solver.solve(x.ravel())

a = np.ones(nx, dtype=np.float64)*1./4
b = np.ones(nx, dtype=np.float64)
c = np.ones(nx, dtype=np.float64)*1./4
a[0] = 0
c[-1] = 0
c[0] = 2
a[-1] = 2

for i in range(nz):
    for j in range(ny):
        x_true = scipy_solve_banded(a, b, c, d[i, j, :])
        assert_allclose(x_true, x[i, j, :])
