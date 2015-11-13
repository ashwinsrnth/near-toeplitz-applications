cd ../experiments
MPIEXECFLAGS='--mca pml ob1 --mca mpi_warn_on_fork 0'

for size in 128 256 512
do
    echo Weak scaling for problem sized $size
    echo python run-compact.py $size
    python run-compact.py $size
    for procs in 2 4
    do
    config="$size $procs $procs $procs"
        echo mpiexec $MPIEXECFLAGS -n $(($procs*$procs*$procs)) python run-parallel-compact.py $config
        mpiexec $MPIEXECFLAGS -n $(($procs*$procs*$procs)) python run-parallel-compact.py $config
    done
done
