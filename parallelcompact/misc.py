import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize
from scipy.linalg import solve_banded

kernel_text = '''

__global__ void negateAndCopyFacesKernel( double* x,
             double* x_faces,
            int nx,
            int ny,
            int nz,
            int mx,
            int npx) {
    
    /*
    Negate and 
    copy the left and right face from the logically [nz, ny, nx] array x
    to a logically [2, nz, ny] array x_faces 
    */

    int tiy = blockIdx.x*blockDim.x + threadIdx.x;
    int tiz = blockIdx.y*blockDim.y + threadIdx.y;

    int i_source;
    int i_dest;
    
    i_source = tiz*(nx*ny) + tiy*nx + 0;
    i_dest = 0 + tiz*ny + tiy;
    
    x_faces[i_dest] = -x[i_source];

    if (mx == 0) {
        x_faces[i_dest] = 0.0;        
    }

    i_source = tiz*(nx*ny) + tiy*nx + nx-1;
    i_dest = nz*ny + tiz*ny + tiy;
    
    x_faces[i_dest] = -x[i_source];
    
    if (mx == npx-1) {
        x_faces[i_dest] = 0.0;        
    }
}

'''

@context_dependent_memoize
def _get_copy_kernel():
    module = compiler.SourceModule(kernel_text, options=['-O2'])
    copy_ernel = module.get_function(
        'negateAndCopyFacesKernel')
    copy_kernel.prepare('PPiiiii')
    return copy_kernel

def negate_and_copy_faces(x, x_faces, shape, mx, npx):
    copy_kernel = _get_copy_kernel()
    copy_kernel.prepared_call((ny/32, nz/32, 1),
            (32, 32, 1),
            x.gpudata,
            x_faces.gpudata,
            shape[2],
            shape[1],
            shape[0],
            mx, npx)

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

def solve_secondary_systems(nx, rank, size):
    a = np.ones(nx, dtype=np.float64)
    b = np.ones(nx, dtype=np.float64)
    c = np.ones(nx, dtype=np.float64)
    rU = np.zeros(nx, dtype=np.float64)
    rL = np.zeros(nx, dtype=np.float64)

    if rank == 0:
        c[0] =  2.0
        a[0] = 0.0

    if rank == size-1:
        a[-1] = 2.0
        c[-1] = 0.0

    rU[0] = -a[0]
    rL[-1] = -c[-1]

    xU = tridiagonal_solve(a, b, c, rU)
    xL = tridiagonal_solve(a, b, c, rL)
    xU_d = gpuarray.to_gpu(xU)
    xL_d = gpuarray.to_gpu(xL)
    return xU_d, xL_d
