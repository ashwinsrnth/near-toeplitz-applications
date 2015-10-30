import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize

kernel_text = '''
__global__ void reducedSolverKernel(double *a_d,
                                    double *b_d,
                                    double *c_d,
                                    double *c2_d,
                                    double *d_d,
                                    int nx,
                                    int ny,
                                    int nz) {
    /*
    The reduced solver kernel essentially
    does a pThomas solve along the
    z (slowest) direction.
    */

    int gix = blockIdx.x*blockDim.x + threadIdx.x;
    int giy = blockIdx.y*blockDim.y + threadIdx.y;
    int start = giy*(nx) + gix;
    int stride = nx*ny;
    double bmac;

    /* do a serial TDMA on the local system */

    c2_d[0] = c_d[0]/b_d[0]; // we need c2_d, because every thread will overwrite c_d[0] otherwise
    d_d[start] = d_d[start]/b_d[0];

    for (int i=1; i<nz; i++)
    {
        bmac = b_d[i] - a_d[i]*c2_d[i-1];
        c2_d[i] = c_d[i]/bmac;
        d_d[start+i*stride] = (d_d[start+i*stride] - a_d[i]*d_d[start+(i-1)*stride])/bmac;
    }

    for (int i=nz-2; i >= 0; i--)
    {
        d_d[start+i*stride] = d_d[start+i*stride] - c2_d[i]*d_d[start+(i+1)*stride];
    }
}
'''

@context_dependent_memoize
def _get_reduced_solver_kernel():
    module = compiler.SourceModule(kernel_text, options=['-O2'])
    reduced_solver_kernel = module.get_function(
        'reducedSolverKernel')
    reduced_solver_kernel.prepare('PPPPPiii')
    return reduced_solver_kernel

def solve_reduced(a_d, b_d, c_d, c2_d, x_d, shape):
    nz, ny, nx = shape
    reduced_solver_kernel = _get_reduced_solver_kernel()
    reduced_solver_kernel.prepared_call(
            (nx/32, ny/32, 1), (32, 32, 1),
            a_d.gpudata,
            b_d.gpudata,
            c_d.gpudata,
            c2_d.gpudata,
            x_d.gpudata,
            nx, ny, nz)

