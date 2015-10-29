import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
import jinja2
import pycuda.compiler as compiler

kernel_template = """
__global__ void sharedMemCyclicReduction( double *a_d,
                                double *b_d,
                                double *c_d,
                                double *d_d,
                                double *k1_d,
                                double *k2_d,
                                double *b_first_d,
                                double *k1_first_d,
                                double *k1_last_d,
                                const double b1,
                                const double c1,
                                const double ai,
                                const double bi,
                                const double ci)
                                {
    /*

    */
    __shared__ double d_l[{{shared_size | int}}];

    int tix = threadIdx.x; 
    int offset = blockIdx.x*{{n}};
    int i, j, k;
    int idx;
    double d_j, d_k;

    /* When loading to shared memory, perform the first
       reduction step */
    idx = 0;
    if (tix == 0) {
        d_l[tix] = d_d[offset+2*tix+1] - \
                    d_d[offset+2*tix]*k1_first_d[idx] - \
                    d_d[offset+2*tix+2]*k2_d[idx];
    }
    else if (tix == ({{(n/2) | int}}-1)) {
        d_l[tix] = d_d[offset+2*tix+1] - \
                    d_d[offset+2*tix]*k1_last_d[idx];
    }
    else {
        d_l[tix] = d_d[offset+2*tix+1] - \
                    d_d[offset+2*tix]*k1_d[idx] - \
                    d_d[offset+2*tix+2]*k2_d[idx];
    }
    __syncthreads();
    
    /* First step of reduction is complete and 
       the coefficients are in shared memory */
    
    /* Do the remaining forward reduction steps: */
    for (int stride=2; stride<{{(n/2) | int}}; stride=stride*2) {
        idx = idx + 1;
        i = (stride-1) + tix*stride;
        if (tix < {{n}}/(2*stride)) {
            if (tix == 0) {
                d_l[i] = d_l[i] - \
                            d_l[i-stride/2]*k1_first_d[idx] - \
                            d_l[i+stride/2]*k2_d[idx];
            }
            else if (i == ({{n}}/2-1)) {
                d_l[i] = d_l[i] - \
                             d_l[i-stride/2]*k1_last_d[idx];
            }
            else {
                d_l[i] = d_l[i] - d_l[i-stride/2]*k1_d[idx] - \
                            d_l[i+stride/2]*k2_d[idx];
            }
        }
        __syncthreads();
    }

    if (tix == 0) {
        j = rint(log2((float) {{(n/2) | int}})) - 1;
        k = rint(log2((float) {{(n/2) | int}}));

        d_j = (d_l[{{n}}/4-1]*b_d[k] - \
               c_d[j]*d_l[{{n}}/2-1])/ \
            (b_first_d[j]*b_d[k] - c_d[j]*a_d[k]);

        d_k = (b_first_d[j]*d_l[{{n}}/2-1] - \
               d_l[{{n}}/4-1]*a_d[k])/ \
            (b_first_d[j]*b_d[k] - c_d[j]*a_d[k]);

        d_l[{{n}}/4-1] = d_j;
        d_l[{{n}}/2-1] = d_k;
    }
    __syncthreads();
    
    idx = rint(log2((float) {{n}}))-2;
    for (int stride={{n}}/4; stride>1; stride=stride/2) {
        idx = idx - 1;
        i = (stride/2-1) + tix*stride;
        if (tix < {{n}}/(2*stride)){
            if (tix == 0) {
                d_l[i] = (d_l[i] - c_d[idx]*d_l[i+stride/2])/\
                            b_first_d[idx];
            }
            else {
                d_l[i] = (d_l[i] - a_d[idx]*d_l[i-stride/2] -\
                            c_d[idx]*d_l[i+stride/2])/b_d[idx];
            }
        }
        __syncthreads();
    }

    //When writing from shared memory, perform the last
    //substitution step
    if (tix == 0) {
        d_d[offset+2*tix] = (d_d[offset+2*tix] - c1*d_l[tix])/b1;
        d_d[offset+2*tix+1] = d_l[tix];
    }
    else {
        d_d[offset+2*tix] = (d_d[offset+2*tix] - \
                                ai*d_l[tix-1] - ci*d_l[tix])/bi;
        d_d[offset+2*tix+1] = d_l[tix];
    } 
    
    __syncthreads();
}



"""



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
        
        tpl = jinja2.Template(kernel_template)
        rendered_kernel = tpl.render(n=self.n, shared_size=self.n/2)
        module = compiler.SourceModule(rendered_kernel,
                options=['-O2'])
        self.cyclic_reduction = module.get_function(
            'sharedMemCyclicReduction')
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
