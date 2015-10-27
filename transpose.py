import numpy as np
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
import jinja2

kernel_template = """
__global__ void transposeCoalesced(double *odata, const double *idata)
{
  __shared__ double tile[{{TILE_DIM}}][{{TILE_DIM}}];

  int x = blockIdx.x * {{TILE_DIM}} + threadIdx.x;
  int y = blockIdx.y * {{TILE_DIM}} + threadIdx.y;
  int width = gridDim.x * {{TILE_DIM}};

  for (int j = 0; j < {{TILE_DIM}}; j += {{BLOCK_ROWS}})
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * {{TILE_DIM}} + threadIdx.x;  // transpose block offset
  y = blockIdx.x * {{TILE_DIM}} + threadIdx.y;

  for (int j = 0; j < {{TILE_DIM}}; j += {{BLOCK_ROWS}})
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}
"""

tpl = jinja2.Template(kernel_template)
rendered_kernel = tpl.render(TILE_DIM=32, BLOCK_ROWS=8)
module = compiler.SourceModule(rendered_kernel, options=['-O2'])
transposeFunc = module.get_function('transposeCoalesced')
transposeFunc.prepare('PP')

a = np.random.rand(64, 64)
b = np.zeros((64, 64), dtype=np.float64)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
transposeFunc.prepared_call((2, 2, 1), (32, 8, 1), b_gpu.gpudata, a_gpu.gpudata)

from numpy.testing import *
assert_allclose(a.transpose(), b_gpu.get())
