#/bin/bash
BENCHMARK=$1
DEFAULT_SIZE=134217728

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=10
DS=$DEFAULT_SIZE
#print header
HASH=`date | md5sum | head -c 5`
FILENAME=$BENCHMARK_$HASH_kernel_conf.res
echo "name,type,N,size,iters,time,ls,ts,gups,bw" | tee $FILENAME 

#run test over kernel params
for LS in 4 8 16 32 64 128 256 512 1024; do
    for TS in 32 64 128 256 512 1024; do 
      for reps in $(seq 1 3); do
	          ./$BENCHMARK -N $DS -I $ITERS -M 0 -LS $LS -TS $TS | tee -a $FILENAME
       done	
    done
done

for LS in 4 8 16 32 64 128 256 512 1024; do
    for TS in 32 64 128 256 512 1024; do 
      for reps in $(seq 1 3); do
	          ./$BENCHMARK -N $DS -I $ITERS -M 1 -LS $LS -TS $TS | tee -a $FILENAME
       done	
    done
done

for LS in 4 8 16 32 64 128 256 512 1024; do
    for TS in 32 64 128 256 512 1024; do  
      for reps in $(seq 1 3); do
	          ./$BENCHMARK -N $DS -I $ITERS -M 2 -LS $LS -TS $TS | tee -a $FILENAME
       done	
    done
done