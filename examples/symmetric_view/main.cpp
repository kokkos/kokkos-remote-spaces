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
    space.fence();

    {
      Kokkos::View<int**, Kokkos::QUOSpace> a_1 = a;
      printf("Refcount a_1: %i Rank: %i\n",a_1.use_count(),my_rank);
      if(a_1.use_count()!=2) printf("Error RefCount %i %i\n",my_rank,a_1.use_count());
    }
    if(a.use_count()!=1) printf("Error RefCount %i %i\n",my_rank,a.use_count());

    for(int i = 0; i < 10; i++)
      a(my_rank,i) = (my_rank + 1) * 100 + i%10;
    space.fence();

    for(int rank=0; rank<num_ranks ; rank++)
      for(int i=0; i<10; i++) {
        if(i==0)
          printf("%i %i %i %i %i Data\n",my_rank*10000+rank*100+i,my_rank,rank,i,a(rank,i));  
        if(a(rank,i) != ((rank+1)*100 + i%10))     
          printf("Error Data: %i %i %i %i %i\n",my_rank,rank,i,a(rank,i),((rank + 1)*100 + i%10));
      }
  }
  Kokkos::finalize();
  MPI_Finalize();
}

