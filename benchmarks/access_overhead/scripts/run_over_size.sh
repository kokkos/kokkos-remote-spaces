#/bin/bash
BENCHMARK=$1
DEFAULT_SIZE=1000

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=200

#print header
HASH=`date|md5sum|head -c 5`
FILENAME="${BENCHMARK}_${HASH}.res"
echo $FILENAME
echo "name,type,N,size,iters,time,gups,bw" | tee $FILENAME 

#run test over size
SIZE=$DEFAULT_SIZE
for S in $(seq 1 21); do 
   for reps in $(seq 1 3); do
      ./$BENCHMARK -N $SIZE -I $ITERS -M 0 | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done

#run test over size
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 21); do 
   for reps in $(seq 1 3); do
      ./$BENCHMARK -N $SIZE -I $ITERS -M 1 | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done

#run test over size
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 21); do 
   for reps in $(seq 1 3); do
      ./$BENCHMARK -N $SIZE -I $ITERS -M 2 | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done
