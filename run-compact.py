import numpy as np
from scipy.linalg import solve_banded

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

dfdx = np.zeros_like(f, dtype=np.float64)

a = np.ones(N, dtype=np.float64)*1./4
b = np.ones(N, dtype=np.float64)*1.
c = np.ones(N, dtype=np.float64)*1./4
d = np.zeros_like(a, dtype=np.float64)
c[0] = 2.0
a[-1] = 2.0

for i in range(N):
    for j in range(N):
        d[1:-1] = (3./(4*dx))*(f[i,j,2:] - f[i,j,:-2])
        d[0] = (1./(2*dx))*(-5*f[i,j,0] + 4*f[i,j,1] + f[i,j,2])
        d[-1] = -(1./(2*dx))*(-5*f[i,j,-1] + 4*f[i,j,-2] + f[i,j,-3])
        dfdx[i, j, :] = tridiagonal_solve(a, b, c, d)

import matplotlib.pyplot as plt
plt.plot(x[0, 0, :], f[0, 0, :])
plt.plot(x[0, 0, :], dfdx[0, 0, :])
plt.savefig('deriv.png')
