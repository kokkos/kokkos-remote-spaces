#!/bin/bash

BENCHMARK=$1
DEFAULT_SIZE=128 #num elems

ITERS=10

#print header
HASH=`date|md5sum|head -c 5`
FILENAME="${BENCHMARK}_${HASH}.res"
echo $FILENAME
echo "name,elems,size_array(KB),bw_set,bw_copy,bw_scale,bw_add,bw_triad" | tee $FILENAME 

VARS="NVSHMEM_SYMMETRIC_SIZE=12884901888"

#run test over size
SIZE=$DEFAULT_SIZE
for S in $(seq 1 23); do 
   for reps in $(seq 1 3); do
      $VARS mpirun -np 1 ./$BENCHMARK $SIZE $ITERS | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done
