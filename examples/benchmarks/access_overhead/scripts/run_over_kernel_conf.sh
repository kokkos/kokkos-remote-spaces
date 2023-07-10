#/bin/bash
BENCHMARK=$1
SIZE=1000000

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

DS=$DATA_SIZE
#print header
HASH=`date | md5sum | head -c 5`
FILENAME=$BENCHMARK_$HASH.res
echo "ls,ts,vs,elems,size,size_total,setTime,copyTime,scaleTime,addTime,triadTime" | tee $FILENAME 

#run test over kernel params
for LS in 1 2 4 8 16 32 64; do
    for TS in 1 2 4 8 16 32; do 
       for VS in 1 2 4 8; do 
          for reps in $(seq 1 3); do
	          ./$BENCHMARK -N $DS -I 10 -M 0 | tee -a $FILENAME
         done
       done	
    done
done