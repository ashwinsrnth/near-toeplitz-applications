import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize
from mpi4py import MPI

kernel_text = r'''
#include <stdio.h>

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

    int tix = blockIdx.x*blockDim.x + threadIdx.x;
    int tiy = blockIdx.y*blockDim.y + threadIdx.y;
    int start = tiy*(nx) + tix;
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
    copy_kernel = module.get_function(
        'negateAndCopyFacesKernel')
    copy_kernel.prepare('PPiiiii')
    return copy_kernel

def negate_and_copy_faces(x, x_faces, shape, mx, npx):
    nz, ny, nx = shape
    copy_kernel = _get_copy_kernel()
    copy_kernel.prepared_call((ny/32, nz/32, 1),
            (32, 32, 1),
            x.gpudata,
            x_faces.gpudata,
            nx,
            ny,
            nz,
            mx, npx)

@context_dependent_memoize
def _get_reduced_solver_kernel():
    module = compiler.SourceModule(kernel_text, options=['-O2'])
    reduced_solver_kernel = module.get_function(
        'reducedSolverKernel')
    reduced_solver_kernel.prepare('PPPPPiii')
    return reduced_solver_kernel

def solve_reduced(a_d, b_d, c_d, c2_d, x_d, shape):
    '''
    Solves systems aligned along the z-axis.
    '''
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

def get_params(line_da, xU_d, xL_d, xR_d):
    xU = xU_d.get()
    xL = xL_d.get()
    nz, ny, nx = line_da.nz, line_da.ny, line_da.nx
    line_rank = line_da.rank
    line_size = line_da.size
    xU_line = np.zeros(2*line_size, dtype=np.float64)
    xL_line = np.zeros(2*line_size, dtype=np.float64)
    xR_faces_d = gpuarray.zeros((2, nz, ny), np.float64)
    xR_faces_line_d = gpuarray.zeros((2*line_size, nz, ny),
            dtype=np.float64)

    line_da.gather(
            [np.array([xU[0], xU[-1]]), 2, MPI.DOUBLE],
            [xU_line, 2, MPI.DOUBLE])
    line_da.gather(
            [np.array([xL[0], xL[-1]]), 2, MPI.DOUBLE],
            [xL_line, 2, MPI.DOUBLE])

    negate_and_copy_faces(xR_d, xR_faces_d,
        (nz, ny, nx), line_rank, line_size)

    line_da.gather(
        [xR_faces_d.gpudata.as_buffer(xR_faces_d.nbytes),
            2*nz*ny, MPI.DOUBLE],
        [xR_faces_line_d.gpudata.as_buffer(xR_faces_line_d.nbytes),
            2*nz*ny, MPI.DOUBLE])

    if line_rank == 0:
        a_reduced = np.zeros(2*line_size, dtype=np.float64)
        b_reduced = np.zeros(2*line_size, dtype=np.float64)
        c_reduced = np.zeros(2*line_size, dtype=np.float64)
        a_reduced[0::2] = -1.
        a_reduced[1::2] = xU_line[1::2]
        b_reduced[0::2] = xU_line[0::2]
        b_reduced[1::2] = xL_line[1::2]
        c_reduced[0::2] = xL_line[0::2]
        c_reduced[1::2] = -1.
        a_reduced[0], c_reduced[0] = 0.0, 0.0
        b_reduced[0] = 1.0
        a_reduced[-1], c_reduced[-1] = 0.0, 0.0
        b_reduced[-1] = 1.0
        a_reduced[1] = 0.
        c_reduced[-2] = 0.

        a_reduced_d = gpuarray.to_gpu(a_reduced)
        b_reduced_d = gpuarray.to_gpu(b_reduced)
        c_reduced_d = gpuarray.to_gpu(c_reduced)
        c2_reduced_d = gpuarray.to_gpu(c_reduced)
        
        solve_reduced(a_reduced_d, b_reduced_d,
                c_reduced_d, c2_reduced_d, xR_faces_line_d, (2*line_size, nz, ny))
     
    line_da.scatter(
      [xR_faces_line_d.gpudata.as_buffer(xR_faces_line_d.nbytes),
            2*nz*ny, MPI.DOUBLE], 
      [xR_faces_d.gpudata.as_buffer(xR_faces_d.nbytes),
            2*nz*ny, MPI.DOUBLE])

    alpha_d = xR_faces_d[0, :, :]
    beta_d = xR_faces_d[1, :, :]
    return alpha_d, beta_d
