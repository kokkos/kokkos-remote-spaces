#include<Kokkos_Core.hpp>
#include<Kokkos_RemoteSpaces.hpp>

#include<mpi.h>

typedef Kokkos::DefaultRemoteMemorySpace RemoteMemSpace
typedef Kokkos::DefaultExecutionSpace Device;
typedef Kokkos::HostSpace::execution_space Host;
typedef Kokkos::TeamPolicy<Device> team_policy;
typedef Kokkos::View<SCALAR*, RemoteMemSpace> view_type;

#define SCALAR double
#define N 1000

KokkosGenerator

int64_t f(int64_t i)
{
  typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
  return rand_gen.urand64() + i;
}

int main(int argc, char* argv[]) {

  int param_size = argc > 1 ? atoi(argv[1]) : N;
  int param_nnodes = argc > 2 ? atoi(argv[2]) : 1;
  int P3 = argc > 3 ? atoi(argv[3]) : 1;
  int P4 = argc > 4 ? atoi(argv[4]) : 1;
  int P5 = argc > 5 ? atoi(argv[5]) : 1;

  // Init

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
  
  // Vars

  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);
  RemoteMemSpace  remoteMemSpace;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(5374857);
  
  
  // Kernel

  {
    int rank_list[P];
    for (int r = 0; r < numRanks; r++)
      rank_list[r] = r;

    view_type a = 
      Kokkos::allocate_symmetric_remote_view<view_type>
      ("MyView",num_ranks,rank_list,param_size);

      Kokkos::parallel_for("Outer", team_policy(N), 
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {

        Kokkos::parallel_for("Inner",TeamThreadRange(team, N),
        [&](const int64_t i){
          a(f(index)) ^ = 0xCOFFEE;  
        }
    });
    remoteMemSpace.fence();
    Kokkos::fence();

    // Check

    for(int64_t i = 0; i < N; ++i)
      ASSERT(a(f(i)) ^ 0xCOFFEE == 0);  
  }
  
  Kokkos::finalize();
  MPI_Finalize();
}

