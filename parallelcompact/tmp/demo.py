import numpy as np
from mpi4py import MPI
from gpuDA import *
from compact import CompactFiniteDifferenceSolver
import matplotlib.pyplot as plt
import pycuda_init
import sys

args = sys.argv
nz, ny, nx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
npz, npy, npx = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
comm = MPI.COMM_WORLD 

# local sizes:
da = DA(comm, (nz, ny, nx), (npz, npy, npx), 1)
line_da = da.get_line_DA(0)
x, y, z = DA_arange(da, (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi))
f = x*np.cos(x*y) + np.sin(z)*y
dfdx_true = -(x*y)*np.sin(x*y) + np.cos(x*y)

dz = z[1, 0, 0] - z[0, 0, 0]
dy = y[0, 1, 0] - y[0, 0, 0]
dx = x[0, 0, 1] - x[0, 0, 0]

cfd = CompactFiniteDifferenceSolver(line_da, 'templated')

f_d = gpuarray.to_gpu(f)
x_d = da.create_global_vector()
f_local_d = da.create_local_vector()
cfd.dfdx(f_d, dx, x_d, f_local_d)
dfdx = x_d.get()

dfdx_global = np.zeros([npz*nz, npy*ny, npx*nx], dtype=np.float64)
dfdx_true_global = np.zeros([npz*nz, npy*ny, npx*nx], dtype=np.float64)

DA_gather_blocks(da, dfdx, dfdx_global)
DA_gather_blocks(da, dfdx_true, dfdx_true_global)

if comm.Get_rank() == 0:
    plt.plot(np.linspace(0, 2*np.pi, npx*nx), dfdx_global[nz*npz/2, ny*npy/2, :], '-', linewidth=5)
    plt.plot(np.linspace(0, 2*np.pi, npx*nx), dfdx_true_global[nz*npz/2, ny*npy/2, :], '--', linewidth=4)
    plt.savefig('demo.png')
