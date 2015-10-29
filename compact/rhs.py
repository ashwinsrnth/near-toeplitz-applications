import numpy as np
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize

kernel_template = '''
__global__ void computeRHS(const double *f_d,
                            double *rhs_d,
                            double dx) {

    int tix = blockIdx.x*blockDim.x + threadIdx.x;
    int tiy = blockIdx.y*blockDim.y + threadIdx.y;
    int tiz = blockIdx.z*blockDim.z + threadIdx.z;
    int nx = gridDim.x*blockDim.x;
    int ny = gridDim.y*blockDim.y;
    int nz = gridDim.z*blockDim.z;

    int i = tiz*(nx*ny) + tiy*nx + tix;
    if (tix == 0) {
        rhs_d[i] = (1./(2*dx))* \
            (-5*f_d[i] + \
                4*f_d[i+1] + f_d[i+2]);
    }

    else if (tix == nx-1) {
        rhs_d[i] = -(1./(2*dx))* \
            (-5*f_d[i] +\
                4*f_d[i-1] + f_d[i-2]);
    }

    else {
        rhs_d[i] = (3./(4*dx))* \
            (f_d[i+1] - f_d[i-1]);
    }
}
'''

@context_dependent_memoize
def _get_rhs_kernel():
    module = compiler.SourceModule(kernel_template, options=['-O2'])
    rhs_kernel = module.get_function('computeRHS')
    rhs_kernel.prepare('PPd')
    return rhs_kernel

def compute_rhs(f, rhs, dx):
    rhs_kernel = _get_rhs_kernel()
    nz, ny, nx = f.shape
    rhs_kernel.prepared_call(
            (nx/8, ny/8, nz/8),
            (8, 8, 8),
            f.gpudata, rhs.gpudata, dx)

if __name__ == "__main__":
    f = np.random.rand(32, 32, 32)
    dfdx = np.zeros_like(f, dtype=np.float64)
    f_d = gpuarray.to_gpu(f)
    dfdx_d = gpuarray.to_gpu(dfdx)
    compute_rhs(f_d, dfdx_d, 0.1)
