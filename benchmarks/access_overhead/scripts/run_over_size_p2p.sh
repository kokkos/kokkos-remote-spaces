#/bin/bash
BENCHMARK=$1
HOST=$2
DEFAULT_SIZE=100

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=1

HASH=`date|md5sum|head -c 5`
DEVS="0,1"
FILENAME="${BENCHMARK}_${HASH}_p2p.res"
echo $FILENAME
echo "name,type,N,size,iters,time,gups,bw" | tee $FILENAME 
VARS0="--bind-to core --map-by socket"
VARS1=" -x UCX_WARN_UNUSED_ENV_VARS=n  -x HCOLL_RCACHE=^ucs -x \
LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/12.0.0/gcc/12.2.0/base/rantbbm/lib64/:$LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE=10730741824"

#Kokkos Remote Spaces + LDC
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 1); do 
   for reps in $(seq 1 1); do
      CUDA_VISIBLE_DEVICES=$DEVS mpirun -np 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -N $SIZE -I $ITERS -M 3 | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done

# #Kokkos Remote Spaces
# let SIZE=$DEFAULT_SIZE
# for S in $(seq 1 20); do 
#   for reps in $(seq 1 3); do
#    CUDA_VISIBLE_DEVICES=$DEVS mpirun -np 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -N $SIZE -I $ITERS -M 2 | tee -a $FILENAME
#  done
#  let SIZE=$SIZE*2
# done

# #Cuda-ware MPI + Kokkos
# let SIZE=$DEFAULT_SIZE
# for S in $(seq 1 20); do 
#   for reps in $(seq 1 3); do
#     CUDA_VISIBLE_DEVICES=$DEVS mpirun -np 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -N $SIZE -I $ITERS -M 1 | tee -a $FILENAME
#  done
#  let SIZE=$SIZE*2
# done

# #MPI + Kokkos
# let SIZE=$DEFAULT_SIZE
# for S in $(seq 1 20); do 
#  for reps in $(seq 1 3); do
#      CUDA_VISIBLE_DEVICES=$DEVS mpirun -np 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -N $SIZE -I $ITERS -M 0 | tee -a $FILENAME
#  done
#  let SIZE=$SIZE*2
# done