import numpy as np
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize
import jinja2

kernel_template = """
    __global__ void applyVerticalStencil(const double *u_d,
                            double *d_d,
                            const double bottom,
                            const double center,
                            const double top,
                            const int N) {

        int tix = blockDim.x*blockIdx.x + threadIdx.x;
        int tiy = blockDim.y*blockIdx.y + threadIdx.y;
        int i;

        if (tix > 0 && tix < N-1 && tiy > 0 && tiy < N-1) {
           i = tiy*N+tix;
           d_d[i] = bottom*u_d[i-N] + center*u_d[i] + top*u_d[i+N];
        }
    }
"""

@context_dependent_memoize
def _get_stencil_kernel():
    module = compiler.SourceModule(kernel_template, options=['-O2'])
    applyVerticalStencilFunc = module.get_function('applyVerticalStencil')
    applyVerticalStencilFunc.prepare('PPdddi')
    return applyVerticalStencilFunc

def compute_rhs(u, d, dx, dt, N):
    '''
    Apply a vertical stencil at
    each of the inner grid points
    to compute the RHS.
    The stencil coefficients
    are a function of dx and dt.

    u is the input array and
    d is the output.
    '''
    block_size = (32, 32)
    bottom = -1./(dx*dx)
    top = -1./(dx*dx)
    center = 2*(-1./dt + 1./(dx*dx))
    f = _get_stencil_kernel()
    f.prepared_call((N/block_size[0], N/block_size[1], 1),
                    (block_size[0], block_size[1], 1),
                     u.gpudata, d.gpudata,
                     bottom, center, top,
                     N)
