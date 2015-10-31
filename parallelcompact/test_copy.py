from reduced import negate_and_copy_faces
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy.linalg import solve_banded
from numpy.testing import assert_allclose


x = np.random.rand(32, 32, 32)
x_faces = np.random.rand(2, 32, 32)

x_d = gpuarray.to_gpu(x)
x_faces_d = gpuarray.to_gpu(x_faces)
negate_and_copy_faces(x_d, x_faces_d, (32, 32, 32), 1, 3)

print x_d.get()[:, :, -1]
print x_faces_d.get()[1, :, :]
