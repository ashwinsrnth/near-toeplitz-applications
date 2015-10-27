import numpy as np
np.random.seed(2359230)
def solve_two_by_two(A, b):
    # solve the 2-by-2 system Ax=b
    x = 1./(A[0,0]*A[1,1] - A[0,1]*A[1,0])*np.asmatrix([[A[1,1], -A[0,1]],[-A[1,0], A[0,0]]])*np.asmatrix(b).transpose()
    return x[0], x[1]

class NearToeplitzSolver:
    def __init__(self, N, nrhs, coeffs):
        '''
        Solve several near Toeplitz tridiagonal systems.

        :param N: size of each tridiagonal systems
        :type N: int
        :param nrhs: number of rhs to solve for
        :type nrhs: int
        :param coeffs: coefficients (b1, c1, ai, bi ci, an, bn)
            that define the tridiagonal system
        :type coeffs: tuple
        '''
        self.N = N
        self.nrhs = nrhs
        self.coeffs = coeffs
        (self.a, self.b, self.c, self.k1, self.k2,
            self.b_first,
            self.k1_first,
            self.k1_last) = _precompute_coefficients(
            self.N, self.coeffs)
    
    def solve(self, d):
        '''
        Solve the system for rhs `d`
        :param d: array of size = N
        :type d: numpy.ndarray
        '''
        a, b, c, k1, k2, b_first, k1_first, k1_last = (
                self.a, self.b, self.c, self.k1, self.k2,
                self.b_first, self.k1_first, self.k1_last)
        b1, c1, ai, bi, ci, an, bn = self.coeffs

        for irhs in range(self.nrhs):
            offset = irhs*self.N
            stride = 1
            for step in np.arange(int(np.log2(self.N))-1):
                stride *= 2
                idx = int(np.log2(stride)) - 1
                for i in range(stride-1, self.N, stride):
                    if i == stride-1:
                        d[offset+i] = (d[offset+i] - 
                            d[offset+i-stride/2]*k1_first[idx] -
                            d[offset+i+stride/2]*k2[idx])
                    elif i == self.N-1:
                        d[offset+i] = (d[offset+i] - 
                            d[offset+i-stride/2]*k1_last[idx])
                    else:
                        d[offset+i] = (d[offset+i] -
                            d[offset+i-stride/2]*k1[idx] -
                            d[offset+i+stride/2]*k2[idx])
            stride *= 2
            m = int(np.log2(stride/2)) - 1
            n = int(np.log2(stride/2))
            d[offset+stride/2-1], d[offset+stride-1] = solve_two_by_two(
                np.array([[b_first[m], c[m]],
                          [a[n], b[n]]]),
                np.array([d[offset+stride/2-1], d[offset+stride-1]]))

            for i in np.arange(int(np.log2(self.N))-1):
                stride /= 2
                for i in range(self.N - stride/2-1, -1, -stride):
                    if stride == 2:
                        if i == 0:
                            d[offset+i] = (d[offset+i] - c1*d[offset+i+stride/2])/b1
                        else:
                            d[offset+i] = (d[offset+i] - ai*d[offset+i-stride/2] - 
                                ci*d[offset+i+stride/2])/bi
                    else:
                        idx = int(np.log2(stride)) - 2
                        if i == stride/2-1:
                            d[offset+i] = (d[offset+i] -
                              c[idx]*d[offset+i+stride/2])/b_first[idx]
                        else:
                            d[offset+i] = (d[offset+i] - 
                              a[idx]*d[offset+i-stride/2] -
                              c[idx]*d[offset+i+stride/2])/b[idx]

def _precompute_coefficients(N, coeffs):
    '''
    The a, b, c, k1, k2
    used in the Cyclic Reduction Algorithm can be
    *pre-computed*.
    Further, for the special case
    of near-Toeplitz matrices,
    they are the same at (almost) each step of reduction,
    with the exception, of course of the boundary conditions.

    Thus, the information can be stored in arrays
    sized log2(N)-1,
    as opposed to arrays sized N.

    Values at the first and last point at each step
    need to be stored seperately.

    The last values for a and b are required only at
    the final stage of forward reduction (the 2-by-2 solve),
    so for convenience, these two scalar values are stored
    at the end of arrays a and b.

    -- See the paper
    "Fast Tridiagonal Solvers on the GPU"

    :param N: size of each tridiagonal systems
    :type N: int
    :param coeffs: coefficients (b1, c1, ai, bi ci, an, bn)
        that define the tridiagonal system
    :type coeffs: tuple
    '''
    log2_N = int(np.log2(N))
    
    # a and b are larger in size by 1:
    a = np.zeros(log2_N, np.float64)
    b = np.zeros(log2_N, np.float64)
    # other coefficient arrays are log2_N-1:
    c = np.zeros(log2_N-1, np.float64)
    k1 = np.zeros(log2_N-1, np.float64)
    k2 = np.zeros(log2_N-1, np.float64)
    b_first = np.zeros(log2_N-1, np.float64)
    k1_first = np.zeros(log2_N-1, np.float64)
    k1_last = np.zeros(log2_N-1, np.float64)

    [b1, c1,
        ai, bi, ci,
            an, bn] = coeffs

    num_reductions = log2_N - 1
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

    # put the last values for a and b at the end of the arrays,
    # these are needed for the 2-by-2 solve
    a[-1] = a_last
    b[-1] = b_last

    return a, b, c, k1, k2, b_first, k1_first, k1_last
