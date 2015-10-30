import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize
import jinja2

kernel_template = """
__global__ void transposeCoalesced(double *odata, const double *idata)
{
  __shared__ double tile[{{TILE_DIM}}][{{TILE_DIM+1}}];

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
@context_dependent_memoize
def _get_transpose_kernel(block_size):
    TILE_DIM = block_size[0]
    BLOCK_ROWS = block_size[1]
    tpl = jinja2.Template(kernel_template)
    rendered_kernel = tpl.render(TILE_DIM=TILE_DIM, BLOCK_ROWS=BLOCK_ROWS)
    module = compiler.SourceModule(rendered_kernel, options=['-O2'])
    transposeFunc = module.get_function('transposeCoalesced')
    transposeFunc.prepare('PP')
    return transposeFunc

def transpose(a, b, N):
    '''
    Transpose the array a,
    putting the result in b.
    a is assumed square with
    width N---multiple of 32
    '''
    block_size = (32, 32)
    f = _get_transpose_kernel(block_size)
    f.prepared_call((N/block_size[0], N/block_size[0], 1), (block_size[0], block_size[1], 1),
                b.gpudata, a.gpudata)

