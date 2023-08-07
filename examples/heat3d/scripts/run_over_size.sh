#/bin/bash
BENCHMARK=$1
HOST1=$2 
HOST2=$3 
HOST3=$4 
HOST4=$5 

DEFAULT_SIZE=10

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=2500

DS=$DATA_SIZE

VARS0="--bind-to core  -x NVSHMEM_SYMMETRIC_SIZE=10737418240"
VARS1="-x UCX_WARN_UNUSED_ENV_VARS=n  -x HCOLL_RCACHE=^ucs -x LD_LIBRARY_PATH=/g/g92/ciesko1/software/nvshmem_src_2.9.0-2/install/lib:$LD_LIBRARY_PATH"
HASH=`date|md5sum|head -c 5`

#=====================================
#=====================================
#=====================================
#=====================================

# TYPE="1x1"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 7); do 
#    for reps in $(seq 1 3); do
#      mpirun -x CUDA_VISIBLE_DEVICES=0 -np 1 -npernode 1 $VARS0 $VARS1 $VARS2 -host "$HOST1:1"  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE*2
# done

# TYPE="1x2"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# # #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 7); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,1 -np 2 -npernode 2  $VARS0 $VARS1 $VARS2 -host "$HOST1:2" ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE*2
# done

# TYPE="1x4"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
# # #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 7); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 -np 4 -npernode 4  $VARS0 $VARS1 $VARS2 -host "$HOST1:4"  ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE*2
# done

#=====================================
#=====================================
#=====================================
#=====================================

TYPE="2x1"
FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
echo $FILENAME
echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
SIZE=$DEFAULT_SIZE
for S in $(seq 1 7); do 
   for reps in $(seq 1 3); do
     mpirun -x CUDA_VISIBLE_DEVICES=0 -np 2 -npernode 1 $VARS0 $VARS1 $VARS2 -host "$HOST1:1,$HOST2:1" ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done

TYPE="2x2"
FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
echo $FILENAME
echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# #run test over size
SIZE=$DEFAULT_SIZE
for S in $(seq 1 7); do 
   for reps in $(seq 1 3); do
      mpirun -x CUDA_VISIBLE_DEVICES=0,1 -np 4 -npernode 2 $VARS0 $VARS1 $VARS2 -host "$HOST1:2,$HOST2:2" ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done

TYPE="2x4"
FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
echo $FILENAME
echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
# #run test over size
SIZE=$DEFAULT_SIZE
for S in $(seq 1 7); do 
   for reps in $(seq 1 3); do
      mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 -np 8 -npernode 4  $VARS0 $VARS1 $VARS2 -host "$HOST1:4,$HOST2:4" ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done


#=====================================
#=====================================
#=====================================
#=====================================

# TYPE="4x1"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 7); do 
#    for reps in $(seq 1 3); do
#      mpirun -x CUDA_VISIBLE_DEVICES=0 -np 4 -npernode 1 $VARS0 $VARS1 $VARS2 -host $HOST1:1,$HOST2:1,$HOST3:1,$HOST4:1 ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE*2
# done

# TYPE="4x2"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 

# # #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 7); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,1 -np 8 --map-by ppr:2:node  $VARS0 $VARS1 $VARS2 -host $HOST1,$HOST2,$HOST3,$HOST4 ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE*2
# done

# TYPE="4x4"
# FILENAME="${BENCHMARK}_${HASH}_${TYPE}_p2p.res"
# echo $FILENAME
# echo "name,type,ranks,step,t_avg,time_inner,time_surface,time_update,time_last_iter,time_all,GUPs,view_size_elems,view_size(MB)" | tee $FILENAME 
# # #run test over size
# SIZE=$DEFAULT_SIZE
# for S in $(seq 1 7); do 
#    for reps in $(seq 1 3); do
#       mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 -np 16 --map-by ppr:4:node  $VARS0 $VARS1 $VARS2 -host $HOST1,$HOST2,$HOST3,$HOST4 ./$BENCHMARK -X $SIZE -Y $SIZE -Z $SIZE -N $ITERS | tee -a $FILENAME
#    done
#    let SIZE=$SIZE*2
# done
