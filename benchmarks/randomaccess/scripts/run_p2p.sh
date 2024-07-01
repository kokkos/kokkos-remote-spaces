# Runscript for RandomAccess

BINARY=$1
HOSTNAME1=$2
HOSTNAME2=$3

HASH=`date|md5sum|head -c 5`
FILENAME="${BINARY}_${HASH}.res"
echo $FILENAME
echo "name,ranks,LS,TS,VL,num_elems,start_idx,idx_range,size(MB),spread,lat,time,MUPs,BW(MB)" | tee $FILENAME 
VAR="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/12.0.0/gcc/12.2.0/base/rantbbm/lib64/:$LD_LIBRARY_PATH "
VAR_HOST="--mca osc ^ucx"

DEFAULT_SIZE=1048576 #1GB
#DEFAULT_SIZE=4194304 #4GB

let SIZE=$DEFAULT_SIZE

# 2 ranks, overhead eval over spread 
# Requires USE_GLOBAL_LAYOUT or USE_PARTITIONED_LAYOUT 
# Can sufficient ready work help to hide latency (and overhead)?

for LS in 128; do
    for TS in 64; do
        for SPREAD in {0..100}; do
            for reps in  $(seq 1 3); do
             mpirun -x CUDA_VISIBLE_DEVICES=0 $VAR -np 1 -npernode 2 -host $HOSTNAME1 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE : -x CUDA_VISIBLE_DEVICES=1 $VAR -np 1 -npernode 2 -host $HOSTNAME2 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE | tee -a $FILENAME 
            done
        done
    done
done


# 2 ranks, constant spread, runs over LS and TS
# Requires USE_GLOBAL_LAYOUT or USE_PARTITIONED_LAYOUT 

SPREAD=100

#NVLink
for LS in 1 4 32 128 512; do
    for TS in 1 2 4 8 16 32 64; do
        for reps in  $(seq 1 3); do
            mpirun -x CUDA_VISIBLE_DEVICES=0 $VAR -np 1 -npernode 2 -host $HOSTNAME1 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE : -x CUDA_VISIBLE_DEVICES=1 $VAR -np 1 -npernode 2 -host $HOSTNAME1 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE | tee -a $FILENAME 
        done
    done
done

#XBus
for LS in 1 4 32 128 512; do
    for TS in 1 2 4 8 16 32 64; do
        for reps in  $(seq 1 3); do
            mpirun -x CUDA_VISIBLE_DEVICES=0 $VAR -np 1 -npernode 2 -host $HOSTNAME1 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE : -x CUDA_VISIBLE_DEVICES=2 $VAR -np 1 -npernode 2 -host $HOSTNAME1 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE | tee -a $FILENAME 
        done
    done
done

#IB
for LS in 1 4 32 128 512; do
    for TS in 1 2 4 8 16 32 64; do
        for reps in  $(seq 1 3); do
            mpirun -x CUDA_VISIBLE_DEVICES=0 $VAR -np 1 -npernode 2 -host $HOSTNAME1 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE : -x CUDA_VISIBLE_DEVICES=1 $VAR -np 1 -npernode 2 -host $HOSTNAME2 ./$BINARY -p $SPREAD -l $LS -t $TS -s $SIZE | tee -a $FILENAME 
        done
    done
done
