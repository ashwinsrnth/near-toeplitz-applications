import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

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

L = 1.0
nx = 32
ny = 32
dx = L/(nx-1)
dy = L/(ny-1)
dt = 1.0

# Initialize values:
u = np.zeros((ny, nx), dtype=np.float64)
u[:,:] = 0.5
u[:,0] = 1.0
u2 = u.copy()

# Initialize tridiagonal system coefficients:
a_x = np.ones(nx)*1./(dx*dx)
b_x = np.ones(nx)*-(2./dt + 2./(dx*dx))
c_x = np.ones(nx)*1./(dx*dx)
d_x = np.zeros(nx, dtype=np.float64)
b_x[0] = 1.
c_x[0] = 0.
a_x[-1] = 0.
b_x[-1] = 1.

a_y = np.ones(ny)*1./(dy*dy)
b_y = np.ones(ny)*-(2./dt + 2./(dy*dy))
c_y = np.ones(ny)*1./(dy*dy)
d_y = np.zeros(ny, dtype=np.float64)
b_y[0] = 1.
c_y[0] = 0.
a_y[-1] = 0.
b_y[-1] = 1.

# Time marching:
for step in range(100):

    # Implicit x, explicit y:
    for i in range(1, ny-1):
        d_x[1:-1] = -2*u[i,1:-1]/dt - (u[i-1,1:-1] - 2*u[i,1:-1] + u[i+1,1:-1])/(dy*dy)
        d_x[0] = u[i,0]
        d_x[-1] = u[i,-1]
        u2[i,:] = tridiagonal_solve(a_x, b_x, c_x, d_x)

    # Implicit y, explicit x:
    u2[...] = u2.transpose().copy()
    u[...] = u.transpose().copy()

    for i in range(1, nx-1):
        d_y[1:-1] = -2*u2[i,1:-1]/dt - (u2[i-1,1:-1] - 2*u2[i,1:-1] + u2[i+1,1:-1])/(dx*dx)
        d_y[0] = u2[i,0]
        d_y[-1] = u2[i,-1]
        u[i,:] = tridiagonal_solve(a_y, b_y, c_y, d_y)
    
    u[...] = u.transpose().copy()
    u2[...] = u2.transpose().copy()

plt.pcolor(u)
plt.colorbar()
plt.savefig('temp.png')
