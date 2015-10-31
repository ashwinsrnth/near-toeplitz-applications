import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize

kernel_text = '''
__global__ void sumSolutionsKernel(double* x_R_d,
                             double* x_UH_d,
                             double* x_LH_d,
                             double* alpha,
                             double* beta)
{
    /*
    Computes the sum of the solution x_R, x_UH and x_LH,
    where x_R is [nz, ny, nx] and x_LH & x_UH are [nx] sized.
    Performs the following:

    x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)
    */
    int tix = blockIdx.x*blockDim.x + threadIdx.x;
    int tiy = blockIdx.y*blockDim.y + threadIdx.y;
    int tiz = blockIdx.z*blockDim.z + threadIdx.z;
    int nx = gridDim.x*blockDim.x;
    int ny = gridDim.y*blockDim.y;
    int nz = gridDim.z*blockDim.z;
    int i2d = tiz*ny + tiy;
    int i3d = tiz*(ny*nx) + tiy*nx + tix;

    x_R_d[i3d] = x_R_d[i3d] + alpha[i2d]*x_UH_d[tix] + beta[i2d]*x_LH_d[tix];
}
'''

@context_dependent_memoize
def _get_sum_kernel():
    module = compiler.SourceModule(kernel_text, options=['-O2'])
    sum_kernel = module.get_function(
        'sumSolutionsKernel')
    sum_kernel.prepare('PPPPP')
    return sum_kernel

def sum_solutions(x_UH, x_LH, x_R, alpha, beta, shape):
    nz, ny, nx = shape
    sum_kernel = _get_sum_kernel()
    sum_kernel.prepared_call((nx/8, ny/8, nx/8),
            (8, 8, 8),
            x_R.gpudata,
            x_UH.gpudata,
            x_LH.gpudata,
            alpha.gpudata,
            beta.gpudata)
