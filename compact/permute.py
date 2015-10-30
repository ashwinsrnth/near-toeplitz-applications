import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler
from pycuda.tools import context_dependent_memoize

kernel_text = '''
__global__ void permuteKernel(double *in_d,
            double *out_d,
            const int n3,
            const int n2,
            const int n1,
            const int x_stride,
            const int y_stride,
            const int z_stride) {
    int tix = blockDim.x*blockIdx.x + threadIdx.x;
    int tiy = blockDim.x*blockIdx.y + threadIdx.y;
    int tiz = blockDim.x*blockIdx.z + threadIdx.z;
    out_d[tiz*n1*n2 + tiy*n1 + tix] = \
        in_d[tiz*z_stride + tiy*y_stride + tix*x_stride];
}
'''

@context_dependent_memoize
def _get_permute_kernel():
    module = compiler.SourceModule(kernel_text,
            options=['-O2'])
    permute_kernel = module.get_function(
            'permuteKernel')
    permute_kernel.prepare('PPiiiiii')
    return permute_kernel

def permute(a_d, b_d, permutation):
    '''
    Permute the data in a 3-dimensional
    GPUArray

    :param a_d: Array containing data to permute
    :type a_d: pycuda.gpuarray.GPUArray
    :param b_d: Space for output
    :type b_d: pycuda.gpuarray.GPUArray
    :param permutation: The desired permutation of the axes
        of a_d
    :type permutation: list or tuple
    '''
    a_strides = np.array(a_d.strides)/a_d.dtype.itemsize
    strides = a_strides[list(permutation)]
    f = _get_permute_kernel()
    f.prepared_call((b_d.shape[2]/8, b_d.shape[1]/8, b_d.shape[0]/8),
            (8, 8, 8),
            a_d.gpudata, b_d.gpudata,
            b_d.shape[2], b_d.shape[1], b_d.shape[0],
            strides[2], strides[1], strides[0])
