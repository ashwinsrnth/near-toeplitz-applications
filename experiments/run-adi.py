import sys
sys.path.append('..')
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from neato import NearToeplitzSolver
from adi.transpose import transpose
from adi.rhs import compute_rhs

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

if len(sys.argv) == 2:
    N = int(sys.argv[1])
else:
    N = 256

L = 1.0
dx = L/(N-1)
dt = 0.1

# Initialize values:
u = np.zeros((N, N), dtype=np.float64)
u[:,:] = 0.5
u[:,0] = 1.0
d_x = u.copy()
d_y = u.transpose().copy() 
u_gpu = gpuarray.to_gpu(u)
d_x_gpu = gpuarray.to_gpu(d_x)
d_y_gpu = gpuarray.to_gpu(d_y)

# Initialize tridiagonal solvers
solver_x = NearToeplitzSolver(N, N-2,
        (1., 0., 1./(dx*dx), -2.*(1./dt + 1./(dx*dx)), 1./(dx*dx), 0., 1.))
solver_y = NearToeplitzSolver(N, N-2,
        (1., 0., 1./(dx*dx), -2.*(1./dt + 1./(dx*dx)), 1./(dx*dx), 0., 1.))

# Time marching:
start = cuda.Event()
end = cuda.Event()

nsteps = 100
for step in range(nsteps):
    if step == 1:
        start.record()
    # Implicit x, explicit y:
    compute_rhs(u_gpu, d_x_gpu, dx, dt, N) 
    solver_x.solve(d_x_gpu[1, 0])
    transpose(d_x_gpu, u_gpu, N)

    # Implicit y, explicit x:
    compute_rhs(u_gpu, d_y_gpu, dx, dt, N)
    solver_y.solve(d_y_gpu[1, 0])
    transpose(d_y_gpu, u_gpu, N)
end.record()
end.synchronize()
time_in_ms = start.time_till(end)
time_per_step_in_ms = time_in_ms/nsteps

print 'Problem size: {0}; Time per step: {1} ms'.format(N, time_per_step_in_ms)

#print 'Plotting...'
#import colormaps as cmaps
#plt.register_cmap(name='inferno', cmap=cmaps.inferno)
#plt.set_cmap(cmaps.inferno)
#plt.pcolormesh(u)
#plt.colorbar()
#plt.savefig('temp.png')
