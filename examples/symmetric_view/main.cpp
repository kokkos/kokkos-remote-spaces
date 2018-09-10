#include<Kokkos_Core.hpp>
#include<Kokkos_RemoteSpaces.hpp>

#include<mpi.h>

#define SPACE Kokkos::DefaultRemoteMemorySpace

#define SCALAR int

typedef Kokkos::View<SCALAR**, SPACE> view_type;
int main(int argc, char* argv[]) {
  MPI_Init(&argc,&argv);
  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_init();
  #endif
  #ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  shmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  shmemx_init_attr (SHMEMX_INIT_WITH_MPI_COMM, &attr);
  #endif
  Kokkos::initialize(argc,argv);
  typedef Kokkos::ViewTraits< void , SPACE >  prop ;
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);
  SPACE  space;
  {
    int rank_list[2];
    rank_list[0] = 0;
    rank_list[1] = 1;
    view_type a = 
      Kokkos::allocate_symmetric_remote_view<view_type>("MyView",num_ranks,rank_list,10);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int) {
      a(0,0) = 0;
    });
    space.fence();

    {
      view_type b = 
       Kokkos::allocate_symmetric_remote_view<view_type>("B",num_ranks,rank_list,10);
      MPI_Win win; 
      int* ptr;
      MPI_Win_allocate(100, sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &ptr, &win);
      Kokkos::View<SCALAR**, SPACE> a_1 = a;
      MPI_Win_free(&win);
      printf("Refcount a_1: %i Rank: %i %s\n",a_1.use_count(),my_rank,a_1.label().c_str());
      if(a_1.use_count()!=2) printf("Error RefCount %i %i\n",my_rank,a_1.use_count());
    }
    if(a.use_count()!=1) printf("Error RefCount %i %i\n",my_rank,a.use_count());

    for(int i = 0; i < 10; i++)
    Kokkos::parallel_for(10, KOKKOS_LAMBDA(const int& i) {
      a(my_rank,i) = (my_rank + 1) * 100 + i%10;
    });
    space.fence();
    printf("Label after write: %s\n",a.label().c_str());
    for(int rank=0; rank<num_ranks ; rank++)
    Kokkos::parallel_for(10, KOKKOS_LAMBDA(const int& i) {
        int val = a(rank,i);
        ////if(i==1)
        //  printf("%i %i %i %i %i %i Data\n",my_rank*10000+rank*100+i,my_rank,rank,i,int(a(rank,i)),val);  
        if(val != ((rank+1)*100 + i%10))     
          printf("Error Data: %i %i %i %i %i\n",my_rank,rank,i,int(a(rank,i)),((rank + 1)*100 + i%10));
      }
    );
    space.fence();    
  }
  Kokkos::finalize();
  MPI_Finalize();
}

