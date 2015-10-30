import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize


kernel_text = '''

__global__ void computeRHSKernel(const double *f_local_d,
                        double *rhs_d,
                        double dx,
                        int mx,
                        int npx)
{
    /*
    Computes the RHS for solving for the x-derivative
    of a function f. f_local is the "local" part of
    the function which includes ghost points.

    dx is the spacing.

    nx, ny, nz define the size of d. f_local is shaped
    [nz+2, ny+2, nx+2]

    mx and npx together decide if we are evaluating
    at a boundary.
    */

    int tix = blockIdx.x*blockDim.x + threadIdx.x;
    int tiy = blockIdx.y*blockDim.y + threadIdx.y;
    int tiz = blockIdx.z*blockDim.z + threadIdx.z;
    int nx = gridDim.x*blockDim.x;
    int ny = gridDim.y*blockDim.y;
    int nz = gridDim.z*blockDim.z;

    int i = tiz*(nx*ny) + tiy*nx + tix;
    int iloc = (tiz+1)*((nx+2)*(ny+2)) + (tiy+1)*(nx+2) + (tix+1);

    rhs_d[i] = (3./(4*dx))*(f_local_d[iloc+1] - f_local_d[iloc-1]);

    if (mx == 0) {
        if (tix == 0) {
            rhs_d[i] = (1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc+1] + f_local_d[iloc+2]);
        }
    }

    if (mx == npx-1) {
        if (tix == nx-1) {
            rhs_d[i] = -(1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc-1] + f_local_d[iloc-2]);
        }
    }
}
'''

@context_dependent_memoize
def _get_compute_RHS_kernel():
    module = compiler.SourceModule(kernel_text, options=['-O2'])
    compute_RHS_kernel = module.get_function(
        'computeRHSKernel')
    compute_RHS_kernel.prepare('PPdii')
    return compute_RHS_kernel

def compute_RHS(f_local, x, dx, shape, mx, npx):
    nz, ny, nx = shape
    compute_RHS_kernel = _get_compute_RHS_kernel()
    compute_RHS_kernel.prepared_call(
             (nx/8, ny/8, nz/8), (8, 8, 8),
             f_local.gpudata,
             x.gpudata,
             dx,
             mx,
             npx)
             
