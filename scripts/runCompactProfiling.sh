cd ../experiments
MPIEXECFLAGS='--mca pml ob1 --mca mpi_warn_on_fork 0'

for size in 32 64 128 256 512
do
    for procs in 2 4
    do
        echo mpiexec $MPIEXECFLAGS -n $(($procs*$procs*$procs)) python run-parallel-compact-profile.py $size $procs $procs $procs
        mpiexec $MPIEXECFLAGS -n $(($procs*$procs*$procs)) python run-parallel-compact-profile.py $size $procs $procs $procs
    done
done
