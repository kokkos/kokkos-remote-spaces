/* A micro benchmark ported mainly from Heat3D to test overhead of RMA */

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double*, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double*>;
using UnmanagedView_t =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView_t = typename RemoteView_t::HostMirror;
using policy_t = Kokkos::RangePolicy<int>;

#define default_N 800000
#define default_iters 3

struct Args_t{
  int mode = 0;
  int N = default_N;
  int iters = default_iters;
};

void print_help() {
  printf("Options (default):\n");
  printf("  -N IARG: (%i) num elements in the vector\n",default_N);
  printf("  -I IARG: (%i) num repititions\n",default_iters);
  printf("  -M IARG: (%i) mode (view type)\n",0);
  printf("     modes:\n");
  printf("       0: Kokkos (Normal)  View\n");
  printf("       1: Kokkos Remote    View\n");
  printf("       2: Kokkos Unmanaged View\n");
}

// read command line args
bool read_args(int argc, char* argv[], Args_t & args) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      print_help();
      return false;
    }
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-N") == 0) args.N = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-I") == 0) args.iters = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-M") == 0) args.mode = atoi(argv[i + 1]);
  }
  return true;
}

template <typename ViewType_t>
struct Stream {
  int N; /* size of vector */
  int iters;   /* number of iterations */

  ViewType_t v;

  Stream(Args_t args):N(args.N),iters(args.iters){};

  KOKKOS_FUNCTION
  void operator()(int i) const { v(i) += 1; }

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b = 0;
    double time = 0;
    for (int i = 0; i <= iters; i++) {
      time_a = timer.seconds();
      Kokkos::parallel_for("stream", policy_t({0}, {N}), *this);
      RemoteSpace_t().fence();
      time_b = timer.seconds();
      time += time_b - time_a;
    }
    double gups =  time / (N * iters) / 1000 / 1000;
    printf("Stream,%lu,%lu,%lf,%lf",
      N,
      iters,
      time,
      gups);
  }
};

int main(int argc, char* argv[]) {
  int mpi_thread_level_available;
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;

#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
  mpi_thread_level_required = MPI_THREAD_SINGLE;
#endif

  MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);

#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

#ifdef KRS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  Kokkos::initialize(argc, argv);

  do{
    Args_t args;
    if(!read_args(argc,argv, args)){
      printf("Wrong args\n");
      break;
    };
       
    {
      if (args.mode == 0) {
        Stream<PlainView_t> s(args);
        s.run();
      } else if (args.mode == 1) {
        Stream<RemoteView_t> s(args);
        s.run();
      } else if (args.mode == 2) {
        printf("unmanaged views not handled yet.");
        // Stream<UnmanagedView_t> s(args);
        // s.run();
      } else {
        printf("invalid mode selected (%d)\n", args.mode);
      }
    }
  }while(false);

  Kokkos::finalize();
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#endif
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
  MPI_Finalize();
  return 0;
}
