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
    // This should not be needed 
    space.fence();
    
    {
      Kokkos::View<int*, Kokkos::QUOSpace> a_1 = a;
      printf("Refcount a_1: %i Rank: %i\n",a_1.use_count(),my_rank);
      if(a_1.use_count()!=2) printf("Error RefCount %i %i\n",my_rank,a_1.use_count());
    }
    if(a.use_count()!=1) printf("Error RefCount %i %i\n",my_rank,a.use_count());
    
    for(int i = my_rank * 10; i < (my_rank+1) * 10; i++)
      a(i) = (my_rank + 1) * 100 + i%10;

    space.fence();

    for(int i=0; i<num_ranks * 10; i++) {
      if(i%10==0)
        printf("%i %i %i %i Data\n",my_rank*10000+i,my_rank,i,a(i));  
      if(a(i) != ((i/10 + 1)*100 + i%10)) printf("Error Data: %i %i %i %i\n",my_rank,i,a(i),((i/10 + 1)*100 + i%10));
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
}

