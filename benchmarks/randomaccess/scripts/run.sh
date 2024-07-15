# Runscript for RandomAccess
BINARY=$1

HASH=`date|md5sum|head -c 5`
FILENAME="${BINARY}_${HASH}.res"
echo $FILENAME
echo "name,ranks,LS,TS,VL,num_elems,start_idx,idx_range,size(MB),spread,lat,time,MUPs,BW(MB)" | tee $FILENAME 

VAR_HOST="--mca osc ^ucx"

DEFAULT_SIZE=1048576 #1GB
#DEFAULT_SIZE=4194304 #4GB

let SIZE=$DEFAULT_SIZE

# 1 rank, overhead eval
# Can sufficient ready work help to hide latency (and overhead)?

for LS in 1 4 32 128 512; do
    for TS in 1 2 4 8 16 32 64; do
        for reps in  $(seq 1 3); do
            CUDA_VISIBLE_DEVICES=3 mpirun $VAR_HOST -np 1 -npernode 1 ./$BINARY -p 0 -l $LS -t $TS -s $SIZE | tee -a $FILENAME
        done
    done
done

