#include<Kokkos_Core.hpp>
#include<Kokkos_RemoteSpaces.hpp>
#include<Kokkos_QUOSpace.hpp>

#include<mpi.h>

typedef Kokkos::View<int**, Kokkos::QUOSpace> view_type;
int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);
  Kokkos::initialize(argc,argv);

  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);

  Kokkos::QUOSpace  space;
  {
    int rank_list[2];
    rank_list[0] = 0;
    rank_list[1] = 1;
    view_type a = 
      Kokkos::allocate_symmetric_remote_view<view_type>("A",num_ranks,rank_list,10);
    int* data = a.data(); 
    printf("Ptr: %i %p\n",my_rank,data);
    {
      Kokkos::View<int**, Kokkos::QUOSpace> a_1 = a;
      printf("Refcount a_1: %i Rank: %i\n",a_1.use_count(),my_rank);
    }
    for(int i = 0; i < 10; i++)
      a(my_rank,i) = (my_rank + 1) * 100 + i%10;
    printf("Refcount a: %i Rank: %i\n",a.use_count(),my_rank);
    space.fence();

    for(int rank=0; rank<num_ranks ; rank++)
      for(int i=0; i<10; i++)
        printf("%i %i %i %i Data\n",my_rank*10000+rank*100+i,my_rank,i,a(rank,i));  
    
  }
  Kokkos::finalize();
  MPI_Finalize();
}

