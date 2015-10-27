import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from cpu import near_toeplitz

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
nx = 64
ny = 32
dx = L/(nx-1)
dy = L/(ny-1)
dt = 1.0 

# Initialize values:
u = np.zeros((ny, nx), dtype=np.float64)
u[:,:] = 0.5
u[:,0] = 1.0

# Initialize tridiagonal solvers
solver_x = near_toeplitz.NearToeplitzSolver(nx, ny-2,
        (1., 0., 1./(dx*dx), -2.*(1./dt + 1./(dx*dx)), 1./(dx*dx), 0., 1.))
solver_y = near_toeplitz.NearToeplitzSolver(ny, nx-2,
        (1., 0., 1./(dy*dy), -2.*(1./dt + 1./(dy*dy)), 1./(dy*dy), 0., 1.))
d_x = u.copy()
d_y = u.transpose().copy() 

# Time marching:
for step in range(100):

    # Implicit x, explicit y:
    for i in range(1, ny-1):
        d_x[i,1:-1] = -2*u[i,1:-1]/dt - (u[i-1,1:-1] - 2*u[i,1:-1] + u[i+1,1:-1])/(dy*dy)
        d_x[i,0] = u[i,0]
        d_x[i,-1] = u[i,-1]

    solver_x.solve(d_x[1:-1,:].ravel())
    u = d_x.transpose()
    
    # Implicit y, explicit x:
    for i in range(1, nx-1):
        d_y[i,1:-1] = -2*u[i,1:-1]/dt - (u[i-1,1:-1] - 2*u[i,1:-1] + u[i+1,1:-1])/(dx*dx)
        d_y[i,0] = u[i,0]
        d_y[i,-1] = u[i,-1]
    
    solver_y.solve(d_y[1:-1,:].ravel())
    u = d_y.transpose()

plt.pcolor(u)
plt.colorbar()
plt.savefig('temp.png')
