import pycuda.driver as cuda
from mpi4py import MPI
import atexit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

cuda.init()
device = cuda.Device(rank%2)
context = device.make_context()
context.push()
atexit.register(context.pop)
atexit.register(cuda.Context.pop)
atexit.register(MPI.Finalize)
