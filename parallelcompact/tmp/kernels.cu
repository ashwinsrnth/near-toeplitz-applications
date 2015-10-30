#include <cuda.h>
extern "C" {

__global__ void computeRHS(const double *f_local_d,
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

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int nx = gridDim.x*blockDim.x;
    int ny = gridDim.y*blockDim.y;
    int nz = gridDim.z*blockDim.z;

    int i = iz*(nx*ny) + iy*nx + ix;
    int iloc = (iz+1)*((nx+2)*(ny+2)) + (iy+1)*(nx+2) + (ix+1);

    rhs_d[i] = (3./(4*dx))*(f_local_d[iloc+1] - f_local_d[iloc-1]);

    if (mx == 0) {
        if (ix == 0) {
            rhs_d[i] = (1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc+1] + f_local_d[iloc+2]);
        }
    }

    if (mx == npx-1) {
        if (ix == nx-1) {
            rhs_d[i] = -(1./(2*dx))*(-5*f_local_d[iloc] + 4*f_local_d[iloc-1] + f_local_d[iloc-2]);
        }
    }
}

__global__ void sumSolutions(double* x_R_d,
                             double* x_UH_d,
                             double* x_LH_d,
                             double* alpha,
                             double* beta,
                            int nx,
                            int ny,
                            int nz)
{
    /*
    Computes the sum of the solution x_R, x_UH and x_LH,
    where x_R is [nz, ny, nx] and x_LH & x_UH are [nx] sized.
    Performs the following:

    x_R + np.einsum('ij,k->ijk', alpha, x_UH_line) + np.einsum('ij,k->ijk', beta, x_LH_line)
    */
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int i3d, i2d;

    i2d = iz*ny + iy;
    i3d = iz*(ny*nx) + iy*nx + ix;

    x_R_d[i3d] = x_R_d[i3d] + alpha[i2d]*x_UH_d[ix] + beta[i2d]*x_LH_d[ix];
}

__global__ void negateAndCopyFaces( double* x,
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

    int iy = blockIdx.x*blockDim.x + threadIdx.x;
    int iz = blockIdx.y*blockDim.y + threadIdx.y;

    int i_source;
    int i_dest;
    
    i_source = iz*(nx*ny) + iy*nx + 0;
    i_dest = 0 + iz*ny + iy;
    
    x_faces[i_dest] = -x[i_source];

    if (mx == 0) {
        x_faces[i_dest] = 0.0;        
    }

    i_source = iz*(nx*ny) + iy*nx + nx-1;
    i_dest = nz*ny + iz*ny + iy;
    
    x_faces[i_dest] = -x[i_source];
    
    if (mx == npx-1) {
        x_faces[i_dest] = 0.0;        
    }
}

__global__ void reducedSolverKernel(double *a_d,
                                    double *b_d,
                                    double *c_d,
                                    double *c2_d,
                                    double *d_d,
                                    int nx,
                                    int ny,
                                    int nz) {
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
}
