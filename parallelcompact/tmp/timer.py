import pycuda.driver as cuda
from mpi4py import MPI

def timeit(func):
    def func_wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm.Barrier()
        t1 = MPI.Wtime()
        result = func(*args, **kwargs)
        cuda.Context.synchronize()
        comm.Barrier()
        t2 = MPI.Wtime()
        if rank == 0: print func.__name__, ': ',t2-t1
        return result 
    return func_wrapper
