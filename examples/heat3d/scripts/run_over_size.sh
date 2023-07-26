#/bin/bash
BENCHMARK=$1
HOST=$2
DEFAULT_SIZE=10

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=10000

DS=$DATA_SIZE

#VARS0="-x NVSHMEM_SYMMETRIC_SIZE=10737418240 -x NVSHMEMTEST_USE_MPI_LAUNCHER=1"
VARS0="-x NVSHMEM_SYMMETRIC_SIZE=10737418240 -x NVSHMEMTEST_USE_MPI_LAUNCHER=1"
VARS1="-x UCX_WARN_UNUSED_ENV_VARS=n"
VARS2="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/11.8.0/gcc/11.3.0/base/ztdfrze/lib64/:$LD_LIBRARY_PATH"

#print header
HASH=`date|md5sum|head -c 5`

# TYPE="1x1x1"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 40); do 
#    for reps in $(seq 1 3); do
#      mpirun -x CUDA_VISIBLE_DEVICES=0 -np 1 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE+10
# done

TYPE="1x1x2"
FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
echo $FILENAME
echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# #run test over size
SIZE=$DEFAULT_SIZE
for S in $(seq 1 40); do 
   for reps in $(seq 1 3); do
      mpirun -x CUDA_VISIBLE_DEVICES=0,1 -np 2 -npernode 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
   done
   let SIZE=$SIZE+10
done


# TYPE="1x2x1"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 40); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,2 -np 2 -npernode 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE+10
# done


# TYPE="2x1x1"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 40); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 -np 2 -npernode 1 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE+10
# done


# TYPE="4x1x1"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 40); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 -np 4 -npernode 1 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE+10
# done