import sys
sys.path.append('..')
import numpy as np
from numpy import sin, cos
from scipy.linalg import solve_banded
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

from neato import NearToeplitzSolver
from compact.rhs import compute_rhs
from compact.permute import permute

sizes = (32, 64, 128, 256)
for i, N in enumerate(sizes):
    x1 = 2.0
    xn = 6.0
    L = xn - x1
    dx = (L)/(N-1)
    z, y, x = np.meshgrid(
                np.linspace(x1, xn,  N),
                np.linspace(x1, xn,  N),
                np.linspace(x1, xn,  N),
                indexing='ij')

    f = sin(x)/x**3

    f_d = gpuarray.to_gpu(f)
    dfdx_d = gpuarray.zeros(f.shape, dtype=np.float64)

    solver = NearToeplitzSolver(N, N*N, (1., 2., 1./4, 1., 1./4, 2., 1.))
    compute_rhs(f_d, dfdx_d, dx)
    solver.solve(dfdx_d)
    dfdx = dfdx_d.get()
    dfdx_true = (x*cos(x) - 3*sin(x))/x**4

    if i == 0:
        err_last_middle = np.abs(dfdx_true[0, 0, N/2] - dfdx[0, 0, N/2])
        err_last_boundary = np.abs(dfdx_true[0, 0, 0] - dfdx[0, 0, 0]) 
        err_last_mean = np.mean(np.abs(dfdx_true[0, 0, :] - dfdx[0, 0, :]))

    else:
        err_this_middle = np.abs(dfdx_true[0, 0, N/2] - dfdx[0, 0, N/2])
        err_this_boundary = np.abs(dfdx_true[0, 0, 0] - dfdx[0, 0, 0]) 
        err_this_mean = np.mean(np.abs(dfdx_true[0, 0, :] - dfdx[0, 0, :]))
        
        print 'Mid point: (N={0})/(N={1}) = {2}'.format(
                sizes[i-1], sizes[i], err_last_middle/err_this_middle)

        print 'Boundary: (N={0})/(N={1}) = {2}'.format(
                sizes[i-1], sizes[i], err_last_boundary/err_this_boundary)

        print 'Mean: (N={0})/(N={1}) = {2}'.format(
                sizes[i-1], sizes[i], err_last_mean/err_this_mean)

        err_last_middle = err_this_middle
        err_last_boundary = err_this_boundary
        err_last_mean = err_this_mean

