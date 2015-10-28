import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
import kernels

'''
A tridiagonal solver for solving
the tridiagonal system that arises in the evaulation
of derivatives using compact finite-difference schemes.

For example, our solver handles the tridiagonal system
for the following tridiagonal scheme:

alpha(f'[i-1] - f'[i+1]) + f'[i] = a(f[i+1] - f[i-1])/dx

With alpha=1/4, a=3/4

Along with the following implicit equation at the boundaries:

f'[1] + 2f'[2] =  (-5f[1] + 4f[2] + f[3])/2dx

The tridiagonal system is then of the form:

1       2       .       .       .       .
1/ 4    1       1/4     .       .       .
.       1/4     1       1/4     .       .
.       .       1/4     1       1/4     .
.       .       .       1/4     1       1/4
.       .       .       .       2       1
'''

class NearToeplitzSolver:

    def __init__(self, n, nrhs, coeffs):
        '''
        Parameters
        ----------
        n: The size of the tridiagonal system.
        nrhs: The number of right hand sides
        coeffs: A list of coefficients that make up the tridiagonal matrix:
            (b1, c1, ai, bi, ci, an, bn)
        '''
        self.n = n
        self.nrhs = nrhs
        self.coeffs = coeffs

        # check that system_size is a power of 2:
        assert np.int(np.log2(self.n)) == np.log2(self.n)

        # compute coefficients a, b, etc.,
        a, b, c, k1, k2, b_first, k1_first, k1_last = _precompute_coefficients(self.n, self.coeffs)

        # copy coefficients to buffers:
        self.a_d = gpuarray.to_gpu(a)
        self.b_d = gpuarray.to_gpu(b)
        self.c_d = gpuarray.to_gpu(c)
        self.k1_d = gpuarray.to_gpu(k1)
        self.k2_d = gpuarray.to_gpu(k2)
        self.b_first_d = gpuarray.to_gpu(b_first)
        self.k1_first_d = gpuarray.to_gpu(k1_first)
        self.k1_last_d = gpuarray.to_gpu(k1_last)
        
        kernels.render_kernel('kernels.jinja2', 'kernels.cu', n=self.n, shared_size=self.n/2)
        self.cyclic_reduction, = kernels.get_funcs('kernels.cu', 'sharedMemCyclicReduction') 
        self.cyclic_reduction.prepare('PPPPPPPPPddddd')
        
    def solve(self, x_d):

        '''
            Solve the tridiagonal system
            for rhs d, given storage for the solution
            vector in x.
        '''
        [b1, c1,
            ai, bi, ci,
                an, bn] = self.coeffs

        # CR algorithm
        # ============================================
        self.cyclic_reduction.prepared_call(
                 (self.nrhs, 1, 1),
                 (self.n/2, 1, 1),
                 self.a_d.gpudata,
                 self.b_d.gpudata,
                 self.c_d.gpudata,
                 x_d.gpudata,
                 self.k1_d.gpudata,
                 self.k2_d.gpudata,
                 self.b_first_d.gpudata,
                 self.k1_first_d.gpudata,
                 self.k1_last_d.gpudata,
                 b1,
                 c1,
                 ai,
                 bi,
                 ci)

def _precompute_coefficients(system_size, coeffs):
    '''
    The a, b, c, k1, k2
    used in the Cyclic Reduction Algorithm can be
    *pre-computed*.
    Further, for the special case
    of constant coefficients,
    they are the same at (almost) each step of reduction,
    with the exception, of course of the boundary conditions.

    Thus, the information can be stored in arrays
    sized log2(system_size)-1,
    as opposed to arrays sized system_size.

    Values at the first and last point at each step
    need to be stored seperately.

    The last values for a and b are required only at
    the final stage of forward reduction (the 2-by-2 solve),
    so for convenience, these two scalar values are stored
    at the end of arrays a and b.

    -- See the paper
    "Fast Tridiagonal Solvers on the GPU"
    '''
    # these arrays technically have length 1 more than required:
    log2_system_size = int(np.log2(system_size))

    a = np.zeros(log2_system_size, np.float64)
    b = np.zeros(log2_system_size, np.float64)
    c = np.zeros(log2_system_size, np.float64)
    k1 = np.zeros(log2_system_size, np.float64)
    k2 = np.zeros(log2_system_size, np.float64)

    b_first = np.zeros(log2_system_size, np.float64)
    k1_first = np.zeros(log2_system_size, np.float64)
    k1_last = np.zeros(log2_system_size, np.float64)

    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs

    num_reductions = log2_system_size - 1
    for i in range(num_reductions):
        if i == 0:
            k1[i] = ai/bi
            k2[i] = ci/bi
            a[i] = -ai*k1[i]
            b[i] = bi - ci*k1[i] - ai*k2[i]
            c[i] = -ci*k2[i]

            k1_first[i] = ai/b1
            b_first[i] = bi - c1*k1_first[i] - ai*k2[i]

            k1_last[i] = an/bi
            a_last = -(ai)*k1_last[i]
            b_last = bn - (ci)*k1_last[i]
        else:
            k1[i] = a[i-1]/b[i-1]
            k2[i] = c[i-1]/b[i-1]
            a[i] = -a[i-1]*k1[i]
            b[i] = b[i-1] - c[i-1]*k1[i] - a[i-1]*k2[i]
            c[i] = -c[i-1]*k2[i]

            k1_first[i] = a[i-1]/b_first[i-1]
            b_first[i] = b[i-1] - c[i-1]*k1_first[i] - a[i-1]*k2[i]

            k1_last[i] = a_last/b[i-1]
            a_last = -a[i-1]*k1_last[i]
            b_last = b_last - c[i-1]*k1_last[i]

    # put the last values for a and b at the end of the arrays:
    a[-1] = a_last
    b[-1] = b_last

    return a, b, c, k1, k2, b_first, k1_first, k1_last
