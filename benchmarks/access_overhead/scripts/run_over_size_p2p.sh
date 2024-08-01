#/bin/bash
BENCHMARK=$1
HOST1=$2
HOST2=$3
DEFAULT_SIZE=33554432 #128

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=30

#NVLInk (=||=)
DEVICE_ID_1=0
DEVICE_ID_2=1

#XBus (Summit-like systems)
#DEVICE_ID_1=0
#DEVICE_ID_2=2

#IB
#DEVICE_ID_1=0
#DEVICE_ID_2=0

HASH=`date|md5sum|head -c 5`
FILENAME="${BENCHMARK}_${HASH}_p2p.res"
echo $FILENAME
echo "name,type,N,size,iters,time,gups,bw" | tee $FILENAME 
VARS0="--bind-to core --map-by socket"
VARS1="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/11.8.0/gcc/9.3.0/base/c3ajoqf/lib64/:$LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE=12884901888"

# Some more potential optimizations
#VARS1="" #-x UCX_WARN_UNUSED_ENV_VARS=n  -x HCOLL_RCACHE=^ucs -x \

# #Kokkos Remote Spaces + LDC
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 23); do 
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 $VARS2 -host $HOST1  ./$BENCHMARK -N $SIZE -I $ITERS -M 3 : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_2 -np 1 $VARS0 $VARS1 $VARS2 -host $HOST2  ./$BENCHMARK -N $SIZE -I $ITERS -M 3 | tee -a $FILENAME 
   done
   let SIZE=$SIZE*2
done

# #Kokkos Remote Spaces
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 23); do 
  for reps in $(seq 1 3); do
        CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 $VARS2 -host $HOST1  ./$BENCHMARK -N $SIZE -I $ITERS -M 2 : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_2 -np 1 $VARS0 $VARS1 $VARS2 -host $HOST2  ./$BENCHMARK -N $SIZE -I $ITERS -M 2 | tee -a $FILENAME 
 done
 let SIZE=$SIZE*2
done

#MPI + Kokkos
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 23); do 
 for reps in $(seq 1 3); do
         CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 $VARS2 -host $HOST1  ./$BENCHMARK -N $SIZE -I $ITERS -M 0 : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_2 -np 1 $VARS0 $VARS1 $VARS2 -host $HOST2  ./$BENCHMARK -N $SIZE -I $ITERS -M 0 | tee -a $FILENAME 
 done
 let SIZE=$SIZE*2
done

# #Cuda-ware MPI + Kokkos
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 23); do 
  for reps in $(seq 1 3); do
        CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 $VARS2 -host $HOST1  ./$BENCHMARK -N $SIZE -I $ITERS -M 1 : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_2 -np 1 $VARS0 $VARS1 $VARS2 -host $HOST2  ./$BENCHMARK -N $SIZE -I $ITERS -M 1 | tee -a $FILENAME 
 done
 let SIZE=$SIZE*2
done

