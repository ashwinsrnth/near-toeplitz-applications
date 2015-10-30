from mpi4py import MPI
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import time

class DA:

    def __init__(self, comm, local_dims, proc_sizes, stencil_width):        
        """
        DA: a class for handling structured grid information

        :param comm: The communicator for all
                processes in the group
        :type comm: mpi4py communicator
        :param local_dims: Dimensions (nz, ny, nx) of the
                portion of the problem belonging to each process
        :type local_dims: tuple 
        :param proc_sizes: The number of processes (npz, npy, npx)
                in each direction
        :type proc_sizes: tuple
        :param stencil_width: The width of boundary information
                that may be exchanged between processes
        :type stencil_width: int
        """
        comm = comm.Create_cart(proc_sizes)
        self.comm = comm
        self.local_dims = local_dims
        self.proc_sizes = proc_sizes
        self.stencil_width = stencil_width
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        self.nz, self.ny, self.nx = local_dims
        self.npz, self.npy, self.npx = proc_sizes
        self.mz, self.my, self.mx = self.comm.Get_topo()[2]
        
        assert(self.size == reduce(lambda a,b: a*b, proc_sizes))
        self._create_halo_arrays()
   
    def create_global_vector(self):
        """
        Returns:
            out (pycuda.gpuarray.GPUArray):
                an array sized (nz, ny, nx)
        """
        return gpuarray.zeros((self.nz,
            self.ny,
            self.nx), dtype=np.float64)

    def create_local_vector(self):
        """
        Returns:
            out (pycuda.gpuarray.GPUArray):
                an array sized (nz+2*sw, ny+2*sw, nx+2*sw)
        """
        return gpuarray.zeros([self.nz+2*self.stencil_width,
            self.ny+2*self.stencil_width,
            self.nx+2*self.stencil_width], dtype=np.float64)

    def global_to_local(self, global_array, local_array):
        """
        Transfer from global portion of an array on each process
        to the local portion (involves communication of boundary info)
        """

        npz, npy, npx = self.npz, self.npy, self.npx
        nz, ny, nx = self.nz, self.ny, self.nx
        mz, my, mx = self.mz, self.my, self.mx
        sw = self.stencil_width

        # copy inner elements:
        self._copy_global_to_local(global_array, local_array)

        # copy from arrays to send halos:
        self._copy_array_to_halo(global_array, self.left_send_halo, [nz, ny, sw], [0, 0, 0])
        self._copy_array_to_halo(global_array, self.right_send_halo, [nz, ny, sw], [0, 0, nx-sw])

        self._copy_array_to_halo(global_array, self.bottom_send_halo, [nz, sw, nx], [0, 0, 0])
        self._copy_array_to_halo(global_array, self.top_send_halo, [nz, sw, nx], [0, ny-sw, 0])
        
        self._copy_array_to_halo(global_array, self.front_send_halo, [sw, ny, nx], [0, 0, 0])
        self._copy_array_to_halo(global_array, self.back_send_halo, [sw, ny, nx], [nz-sw, 0, 0])

        # perform swaps in x-direction
        sendbuf = [self.right_send_halo.gpudata.as_buffer(self.right_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.left_recv_halo.gpudata.as_buffer(self.left_recv_halo.nbytes), MPI.DOUBLE]
        req1 = self._forward_swap(sendbuf, recvbuf, self.rank-1, self.rank+1, mx, npx, 10)

        sendbuf = [self.left_send_halo.gpudata.as_buffer(self.left_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.right_recv_halo.gpudata.as_buffer(self.right_recv_halo.nbytes), MPI.DOUBLE]
        req2 = self._backward_swap(sendbuf, recvbuf, self.rank+1, self.rank-1, mx, npx, 20)

        # perform swaps in y-direction:
        sendbuf = [self.top_send_halo.gpudata.as_buffer(self.top_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.bottom_recv_halo.gpudata.as_buffer(self.bottom_recv_halo.nbytes), MPI.DOUBLE]
        req3 = self._forward_swap(sendbuf, recvbuf, self.rank-npx, self.rank+npx, my, npy, 30)
       
        sendbuf = [self.bottom_send_halo.gpudata.as_buffer(self.bottom_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.top_recv_halo.gpudata.as_buffer(self.top_recv_halo.nbytes), MPI.DOUBLE]
        req4 = self._backward_swap(sendbuf, recvbuf, self.rank+npx, self.rank-npx, my, npy, 40)

        # perform swaps in z-direction:
        sendbuf = [self.back_send_halo.gpudata.as_buffer(self.back_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.front_recv_halo.gpudata.as_buffer(self.front_recv_halo.nbytes), MPI.DOUBLE]
        req5 = self._forward_swap(sendbuf, recvbuf, self.rank-npx*npy, self.rank+npx*npy, mz, npz, 50)
       
        sendbuf = [self.front_send_halo.gpudata.as_buffer(self.front_send_halo.nbytes), MPI.DOUBLE]
        recvbuf = [self.back_recv_halo.gpudata.as_buffer(self.back_recv_halo.nbytes), MPI.DOUBLE]
        req6 = self._backward_swap(sendbuf, recvbuf, self.rank+npx*npy, self.rank-npx*npy, mz, npz, 60)

        requests = [req for req in  [req1, req2, req3, req4, req5, req6] if req != None]
        MPI.Request.Waitall(requests, [MPI.Status()]*len(requests))

        # copy from recv halos to local_array:
        if self.has_neighbour('left'):
            self._copy_halo_to_array(self.left_recv_halo, local_array, [nz, ny, sw], [sw, sw, 0])
        
        if self.has_neighbour('right'):
            self._copy_halo_to_array(self.right_recv_halo, local_array, [nz, ny, sw], [sw, sw, sw+nx])

        if self.has_neighbour('bottom'):
            self._copy_halo_to_array(self.bottom_recv_halo, local_array, [nz, sw, nx], [sw, 0, sw])
        
        if self.has_neighbour('top'):
            self._copy_halo_to_array(self.top_recv_halo, local_array, [nz, sw, nx], [sw, sw+ny, sw])

        if self.has_neighbour('front'):
            self._copy_halo_to_array(self.front_recv_halo, local_array, [sw, ny, nx], [0, sw, sw])
        
        if self.has_neighbour('back'):
            self._copy_halo_to_array(self.back_recv_halo, local_array, [sw, ny, nx], [sw+nz, sw, sw])
        
    def local_to_global(self, local_array, global_array):
        """
        Transfer from the local portion of an array on each process
        to the global portion (involves no communication)
        """

        self._copy_local_to_global(local_array, global_array)
    
    def gather(self, sendbuf, recvbuf, root=0):
        self.comm.Gather(sendbuf, recvbuf, root=root)

    def gatherv(self, sendbuf, recvbuf, root=0):
        self.comm.Gatherv(sendbuf, recvbuf, root=root)

    def scatterv(self, sendbuf, recvbuf, root=0):
        self.comm.Scatterv(sendbuf, recvbuf, root=root)
   
    def scatter(self, sendbuf, recvbuf, root=0):
        self.comm.Scatter(sendbuf, recvbuf, root=root)

    def get_line_DA(self, direction):
        """
        Return a one dimensional DA
        composed of all processes in the specified direction.
        For example, for direction=1,
        the DA has proc_size ``(1, 1, npy)``
        and local_dims ``(nz, nx, ny)``.

        :parameter direction: Indicates x- (0), y- (1) or z (2)- direction.
        :type direction: int
        """
        ranks_matrix = np.arange(self.npz*self.npy*self.npx).reshape([self.npz, self.npy, self.npx])
        global_group = self.comm.Get_group()
        if direction == 0:
            line_group = global_group.Incl(ranks_matrix[self.mz, self.my, :])
            line_proc_sizes = [1, 1, self.npx]
            line_local_dims = [self.nz, self.ny, self.nx]
        elif direction == 1:
            line_group = global_group.Incl(ranks_matrix[self.mz, :, self.mx])
            line_proc_sizes = [1, 1, self.npy]
            line_local_dims = [self.nz, self.nx, self.ny]
        else:
            line_group = global_group.Incl(ranks_matrix[:, self.my, self.mx])
            line_proc_sizes = [1, 1, self.npz]
            line_local_dims = [self.ny, self.nx, self.nz]
        line_comm = self.comm.Create(line_group)
        return self.__class__(line_comm, line_local_dims, line_proc_sizes, self.stencil_width)

    def _forward_swap(self, sendbuf, recvbuf, src, dest, loc, dimprocs, tag):
        
        # Perform swap in the +x, +y or +z direction
        req = None

        if loc > 0 and loc < dimprocs-1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)

        elif loc == 0 and dimprocs > 1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = None

        elif loc == dimprocs-1 and dimprocs > 1:
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)
            
        return req

    def _backward_swap(self, sendbuf, recvbuf, src, dest, loc, dimprocs, tag):
        
        # Perform swap in the -x, -y or -z direction
        req = None

        if loc > 0 and loc < dimprocs-1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)
        
        elif loc == 0 and dimprocs > 1:
            req = self.comm.Irecv(recvbuf, source=src, tag=tag)

        elif loc == dimprocs-1 and dimprocs > 1:
            self.comm.Isend(sendbuf, dest=dest, tag=tag)
            req = None

        return req

    def _create_halo_arrays(self):

        # Allocate space for the halos: two per face,
        # one for sending and one for receiving.

        nz, ny, nx = self.local_dims
        sw = self.stencil_width
        # create two halo regions for each face, one holding
        # the halo values to send, and the other holding
        # the halo values to receive.

        self.left_recv_halo = gpuarray.empty([nz, ny, sw], dtype=np.float64)
        self.left_send_halo = self.left_recv_halo.copy()
        self.right_recv_halo = self.left_recv_halo.copy()
        self.right_send_halo = self.left_recv_halo.copy()
    
        self.bottom_recv_halo = gpuarray.empty([nz, sw, nx], dtype=np.float64)
        self.bottom_send_halo = self.bottom_recv_halo.copy()
        self.top_recv_halo = self.bottom_recv_halo.copy()
        self.top_send_halo = self.bottom_recv_halo.copy()

        self.back_recv_halo = gpuarray.empty([sw, ny, nx], dtype=np.float64)
        self.back_send_halo = self.back_recv_halo.copy()
        self.front_recv_halo = self.back_recv_halo.copy()
        self.front_send_halo = self.back_recv_halo.copy()

    def _copy_array_to_halo(self, array, halo, copy_dims, copy_offsets, dtype=np.float64):

        # copy from 3-d array to 2-d halo
        #
        # Paramters:
        # array, halo:  gpuarrays involved in the copy.
        # copy_dims: number of elements to copy in (z, y, x) directions
        # copy_offsets: offsets at the source in (z, y, x) directions
        
        nz, ny, nx = self.local_dims 
        d, h, w  = copy_dims
        z_offs, y_offs, x_offs = copy_offsets
        
        typesize = array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(array.gpudata)
        copier.set_dst_device(halo.gpudata)

        copier.src_x_in_bytes = x_offs*typesize
        copier.src_y = y_offs
        copier.src_z = z_offs

        copier.src_pitch = array.strides[1]
        copier.dst_pitch = halo.strides[1]
        copier.src_height = ny
        copier.dst_height = h


        copier.width_in_bytes = w*typesize
        copier.height = h
        copier.depth = d

        # perform the copy:
        copier()

    def _copy_halo_to_array(self, halo, array, copy_dims, copy_offsets, dtype=np.float64):
        
        # copy from 2-d halo to 3-d array
        #
        # Parameters:
        # halo, array:  gpuarrays involved in the copy
        # copy_dims: number of elements to copy in (z, y, x) directions
        # copy_offsets: offsets at the destination in (z, y, x) directions
        
        nz, ny, nx = self.local_dims
        sw = self.stencil_width
        d, h, w = copy_dims
        z_offs, y_offs, x_offs = copy_offsets

        typesize = array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(halo.gpudata)
        copier.set_dst_device(array.gpudata)

        # this time, offsets are at the destination:
        copier.dst_x_in_bytes = x_offs*typesize
        copier.dst_y = y_offs
        copier.dst_z = z_offs

        copier.src_pitch = halo.strides[1]
        copier.dst_pitch = array.strides[1]
        copier.src_height = h
        copier.dst_height = ny+2*sw

        copier.width_in_bytes = w*typesize
        copier.height = h
        copier.depth = d
        
        # perform the copy:
        copier()

    def _copy_global_to_local(self, global_array, local_array, dtype=np.float64):

        nz, ny, nx = self.local_dims
        sw = self.stencil_width
      
        typesize = global_array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(global_array.gpudata)
        copier.set_dst_device(local_array.gpudata)

        # offsets 
        copier.dst_x_in_bytes = sw*typesize
        copier.dst_y = sw
        copier.dst_z = sw

        copier.src_pitch = global_array.strides[1] 
        copier.dst_pitch = local_array.strides[1]
        copier.src_height = ny
        copier.dst_height = ny+2*sw

        copier.width_in_bytes = nx*typesize
        copier.height = ny
        copier.depth = nz

        copier()

    def _copy_local_to_global(self, local_array, global_array, dtype=np.float64):

        nz, ny, nx = self.local_dims
        sw = self.stencil_width

        typesize = global_array.dtype.itemsize

        copier = cuda.Memcpy3D()
        copier.set_src_device(local_array.gpudata)
        copier.set_dst_device(global_array.gpudata)

        # offsets
        copier.src_x_in_bytes = sw*typesize
        copier.src_y = sw
        copier.src_z = sw

        copier.src_pitch = local_array.strides[1]
        copier.dst_pitch = global_array.strides[1]
        copier.src_height = ny+2*sw
        copier.dst_height = ny

        copier.width_in_bytes = nx*typesize
        copier.height = ny
        copier.depth = nz

        copier()


    def has_neighbour(self, side):
        
        # Check that the processor has a
        # neighbour on a specified side
        # side can be 'left', 'right', 'top' or 'bottom'
        
        npz, npy, npx = self.comm.Get_topo()[0]
        mz, my, mx = self.comm.Get_topo()[2]
        
        if side == 'left' and mx > 0:
            return True
        
        elif side == 'right' and mx < npx-1:
            return True

        elif side == 'bottom' and my > 0:
            return True
        
        elif side == 'top' and my < npy-1:
            return True

        elif side == 'front' and mz > 0:
            return True

        elif side == 'back' and mz < npz-1:
            return True

        else:
            return False

def DA_arange(da, x_range, y_range, z_range):
    '''
    Return x, y and z arrays
    representing coordinate values
    for a grid with specified ranges
    
    Args:
        x_range (tuple): (xmin, xmax)
        y_range (tuple): (ymin, ymax)
        z_range (tuple): (zmin, zmax)

    Returns:
        x, y, z (np.ndarrays): coordinate arrays,
            all sized (nz, ny, nx)
    '''
    nz, ny, nx = da.nz, da.ny, da.nx
    npz, npy, npx = da.npz, da.npy, da.npx
    mz, my, mx = da.mz, da.my, da.mx
    NZ, NY, NX = nz*npz, ny*npy, nx*npx 
    dx = float(x_range[-1]-x_range[0])/(NX-1)
    dy = float(y_range[-1]-y_range[0])/(NY-1)
    dz = float(z_range[-1]-z_range[0])/(NZ-1)
    x_start, y_start, z_start = (x_range[0] + mx*nx*dx,
            y_range[0] + my*ny*dy,
            z_range[0] + mz*nz*dz)
    z, y, x = np.meshgrid(
            np.linspace(z_start, z_start+(nz-1)*dz, nz),
            np.linspace(y_start, y_start+(ny-1)*dy, ny),
            np.linspace(x_start, x_start+(nx-1)*dx, nx),
            indexing='ij')
    return x, y, z

def DA_scatter_blocks(da, x_global, x_local):

    mz, my, mx = da.mz, da.my, da.mx 
    npz, npy, npx = da.npz, da.npy, da.npx
    nz, ny, nx = da.nz, da.ny, da.nx
    assert((nz, ny, nx) == x_local.shape)
    NZ, NY, NX = npz*nz, npy*ny, npx*nx
    size = da.size
    rank = da.rank

    start_z, start_y, start_x = mz*nz, my*ny, mx*nx
    subarray_aux = MPI.DOUBLE.Create_subarray([NZ, NY, NX],
                        [nz, ny, nx], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
    sendbuf = [start_index, MPI.INT]
    displs = np.zeros(size, dtype=np.int)
    recvbuf = [displs, MPI.INT]
    da.comm.Gather(sendbuf, recvbuf, root=0)
    da.comm.Barrier()

    da.comm.Scatterv([x_global, np.ones(size, dtype=np.int), displs, subarray],
        [x_local, MPI.DOUBLE], root=0)

    subarray.Free()

def DA_gather_blocks(da, x_local, x_global):

    mz, my, mx = da.mz, da.my, da.mx 
    npz, npy, npx = da.npz, da.npy, da.npx
    nz, ny, nx = da.nz, da.ny, da.nx
    assert((nz, ny, nx) == x_local.shape)
    NZ, NY, NX = npz*nz, npy*ny, npx*nx
    size = da.size
    rank = da.rank 

    start_z, start_y, start_x = mz*nz, my*ny, mx*nx
    subarray_aux = MPI.DOUBLE.Create_subarray([NZ, NY, NX],
                        [nz, ny, nx], [start_z, start_y, start_x])
    subarray = subarray_aux.Create_resized(0, 8)
    subarray.Commit()

    start_index = np.array(start_z*(NX*NY) + start_y*(NX) + start_x, dtype=np.int)
    sendbuf = [start_index, MPI.INT]
    displs = np.zeros(size, dtype=np.int)
    recvbuf = [displs, MPI.INT]
    da.comm.Gather(sendbuf, recvbuf, root=0)
    da.comm.Barrier()

    da.comm.Gatherv([x_local, MPI.DOUBLE],
        [x_global, np.ones(size, dtype=np.int), displs, subarray], root=0)

    subarray.Free()


