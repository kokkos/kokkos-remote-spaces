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
VARS0="--bind-to core --map-by socket"
VARS1="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/12.0.0/gcc/12.2.0/base/rantbbm/lib64/:$LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE=12884901888"

SIZE_DEF=40

#One rank
let SIZE=$SIZE_DEF
FILENAME_ACTUAL=$FILENAME"_1x1x1.res"
echo "name,type,ranks,t1,t2,t3,X,size(MB),t_all_iter,t_all" | tee $FILENAME_ACTUAL 
for S in $(seq 1 4); do 
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 -host $HOST1  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 | tee -a $FILENAME_ACTUAL
   done
   let SIZE=$SIZE*2
done

#Two ranks 
let SIZE=$SIZE_DEF
FILENAME_ACTUAL=$FILENAME"_1x1x2.res"
echo "name,type,ranks,t1,t2,t3,X,size(MB),t_all_iter,t_all" | tee $FILENAME_ACTUAL 
for S in $(seq 1 4); do 
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 -host $HOST1  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_2 -np 1 $VARS0 $VARS1 -host $HOST1  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 | tee -a $FILENAME_ACTUAL
   done
   let SIZE=$SIZE*2
done

#Two ranks 
let SIZE=$SIZE_DEF
FILENAME_ACTUAL=$FILENAME"_1x2x1.res"
echo "name,type,ranks,t1,t2,t3,X,size(MB),t_all_iter,t_all" | tee $FILENAME_ACTUAL 
for S in $(seq 1 4); do 
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 1 $VARS0 $VARS1 -host $HOST1  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 : \
      -x CUDA_VISIBLE_DEVICES=$DEVICE_ID_3 -np 1 $VARS0 $VARS1 -host $HOST1  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 | tee -a $FILENAME_ACTUAL 
   done
   let SIZE=$SIZE*2
done


#Two ranks 
let SIZE=$SIZE_DEF
FILENAME_ACTUAL=$FILENAME"_2x1x1.res"
echo "name,type,ranks,t1,t2,t3,X,size(MB),t_all_iter,t_all" | tee $FILENAME_ACTUAL 
for S in $(seq 1 4); do 
   for reps in $(seq 1 3); do
      CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 2 $VARS0 $VARS1 -host $HOST1,$HOST2  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 | tee -a $FILENAME_ACTUAL 
   done
   let SIZE=$SIZE*2
done

# #Four ranks 
# let SIZE=$SIZE_DEF
# FILENAME_ACTUAL=$FILENAME"_4x1x1.res"
# echo "name,type,ranks,t1,t2,t3,X,size(MB),t_all_iter,t_all" | tee $FILENAME_ACTUAL 
# for S in $(seq 1 4); do 
#    for reps in $(seq 1 3); do
#       CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np 4 $VARS0 $VARS1 -host $HOST1,$HOST2,$HOST3,$HOST4  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N 100 | tee -a $FILENAME_ACTUAL 
#    done
#    let SIZE=$SIZE*2
# done
