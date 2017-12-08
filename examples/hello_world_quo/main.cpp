#include<Kokkos_Core.hpp>
#include<Kokkos_QUOSpace.hpp>
#include<mpi.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize(argc,argv);

  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);

  Kokkos::QUOSpace  space;
  {
    Kokkos::View<int*, Kokkos::QUOSpace> a("A",num_ranks * 10 * sizeof(int));  
    int* data = a.data(); 
    //space.allocate(num_ranks * 10 * sizeof(int)); 
    printf("Ptr: %i %p\n",my_rank,data);
    {
      Kokkos::View<int*, Kokkos::QUOSpace> a_1 = a;
      printf("Refcount a_1: %i Rank: %i\n",a_1.use_count(),my_rank);
    }
    for(int i = my_rank * 10; i < (my_rank+1) * 10; i++)
      a(i) = (my_rank + 1) * 100 + i%10;
    printf("Refcount a: %i Rank: %i\n",a.use_count(),my_rank);
    space.fence();

    for(int i=0; i<num_ranks * 10; i++)
      printf("%i %i %i %i Data\n",my_rank*10000+i,my_rank,i,a(i));  
  }
  Kokkos::finalize();
  MPI_Finalize();
}

