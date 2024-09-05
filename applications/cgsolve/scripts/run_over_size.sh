#/bin/bash
BENCHMARK=$1
HOST1=$2
HOST2=$3
HOST3=$4
HOST4=$5

DEVICE_ID_1=0
DEVICE_ID_2=1
DEVICE_ID_3=2
DEVICE_ID_4=3

HASH=`date|md5sum|head -c 5`
FILENAME="${BENCHMARK}_${HASH}"
echo $FILENAME
#VARS0="--bind-to core --map-by socket"
VARS1="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/12.0.0/gcc/12.2.0/base/rantbbm/lib64/:$LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE=8589934592"


#One rank
FILENAME_ACTUAL=$FILENAME"_1x1x1.res"
echo "ranks,N,num_iters,total_flops,time,GFlops,BW(GB/sec)" | tee $FILENAME_ACTUAL
for S in 10 20 40 80 160 300; do
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 -host $HOST1 ./$BENCHMARK $S 10 | tee -a $FILENAME_ACTUAL
   done
done 

#Two ranks 
FILENAME_ACTUAL=$FILENAME"_1x1x2.res"
echo "ranks,N,num_iters,total_flops,time,GFlops,BW(GB/sec)" | tee $FILENAME_ACTUAL
for S in 10 20 40 80 160 300; do
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 -host $HOST1 ./$BENCHMARK $S 10  : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_2 -np 1 $VARS0 $VARS1 -host $HOST1 ./$BENCHMARK $S 10 | tee -a $FILENAME_ACTUAL
   done
done

#Two ranks 
FILENAME_ACTUAL=$FILENAME"_1x2x1.res"
echo "ranks,N,num_iters,total_flops,time,GFlops,BW(GB/sec)" | tee $FILENAME_ACTUAL
for S in 10 20 40 80 160 300; do
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 -host $HOST1 ./$BENCHMARK $S 10  : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_3 -np 1 $VARS0 $VARS1 -host $HOST1 ./$BENCHMARK $S 10 | tee -a $FILENAME_ACTUAL 
   done
done

#Two ranks 
FILENAME_ACTUAL=$FILENAME"_2x1x1.res"
echo "ranks,N,num_iters,total_flops,time,GFlops,BW(GB/sec)" | tee $FILENAME_ACTUAL
for S in 10 20 40 80 160 300; do
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 2 $VARS0 $VARS1 -host $HOST1,$HOST2 ./$BENCHMARK $S 10 | tee -a $FILENAME_ACTUAL 
   done
done

#Four ranks 
# FILENAME_ACTUAL=$FILENAME"_4x1x1.res"
# echo "ranks,N,num_iters,total_flops,time,GFlops,BW(GB/sec" | tee $FILENAME_ACTUAL
# for S in 10 20 40 80 160 300; do
#    for reps in $(seq 1 3); do
#       CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 4 $VARS0 $VARS1 -host $HOST1,$HOST2,$HOST3,$HOST4  ./$BENCHMARK $S 10 | tee -a $FILENAME_ACTUAL 
#    done
# done
