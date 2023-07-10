/* A micro benchmark ported mainly from Heat3D to test overhead of RMA */

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace&) {}
  static bool overlap() { return false; }
};

#ifndef KOKKOS_ENABLE_DEBUG
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(Kokkos::Cuda& space) {
    cudaStream_t stream = space.cuda_stream();
    cudaStreamDestroy(stream);
  }
  static bool overlap() { /* returns true if you can overlap */
    bool value          = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::stoi(local_rank_str) == 0);
    }
    return value;
  }
};
#endif /* KOKKOS_ENABLE_CUDA */
#endif /* KOKKOS_ENABLE_DEBUG */

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double*, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double*>;
using UnmanagedView_t =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView_t = typename RemoteView_t::HostMirror;

template <typename ViewType_t>
struct Stream_Manager {
  int len; /* size of vector */
  int N;   /* number of iterations */
  int indication_of_lack_of_cpp_knowledge;

  // Temperature and delta Temperature
  HostView_t V_h;
  ViewType_t V;

  Stream_Manager(int a) : indication_of_lack_of_cpp_knowledge(a) {
    // populate with defaults, set the rest in initialize_views.
    len = 8000000;
    V_h = HostView_t();
    V   = ViewType_t();
    N   = 10000;
  }

  void initialize_views() {
    /* how to handle unmanaged? */
    // if (std::is_same<ViewType_t, UnmanagedView_t>::value) {
    // R = RemoteView_t("System::Vector", len);
    // V = ViewType_t(R.data(), len);
    // }
    // else {
    V = ViewType_t("System::Vector", len);
    // }
    V_h = HostView_t("Host::Vector", V.extent(0));

    Kokkos::deep_copy(V_h, 0);
    Kokkos::deep_copy(V, V_h);

    printf("My Vector: [%i, %i]\n", 0, len - 1);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -l IARG: (%i) num elements in the V vector\n", len);
    printf("  -N IARG: (%i) num repititions\n", N);
  }

  // check command line args
  bool check_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        print_help();
        return false;
      }
    }
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-l") == 0) len = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-len") == 0) len = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
    }
    initialize_views();
    return true;
  }
  struct stream_benchmark {};
  KOKKOS_FUNCTION
  void operator()(stream_benchmark, int i) const { V(i) += 1; }
  void Stream_Benchmark() {
    using policy_t = Kokkos::RangePolicy<stream_benchmark, int>;
    Kokkos::parallel_for("stream", policy_t({0}, {len}), *this);
  }

  // run copy benchmark
  void timestep() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b    = 0;
    double time_stream = 0;
    double old_time    = 0.0;
    for (int t = 0; t <= N; t++) {
      time_a = timer.seconds();
      Stream_Benchmark();
      RemoteSpace_t().fence();
      time_b = timer.seconds();
      time_stream += time_b - time_a;
      if ((t % 400 == 0 || t == N)) {
        double time = timer.seconds();
        Kokkos::deep_copy(V_h, V);
        printf("%d V_h(0)=%lf Time (%lf %lf)\n", t, V_h(0), time,
               time - old_time);
        printf("    stream: %lf\n", time_stream);
        old_time = time;
      }
    }
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
  {
    /* use 'mode' variable to pack any of three benchmarks into one here */
    int mode = 0;
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        printf("modes:\n");
        printf("  0: Kokkos (Normal)  View\n");
        printf("  1: Kokkos Remote    View\n");
        printf("  2: Kokkos Unmanaged View\n");
        printf("  -m IARG: (%d) which mode to choose\n", mode);
        break;
      }
      if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "-type") == 0) {
        mode = atoi(argv[i + 1]);
      }
    }

    if (mode == 0) {
      Stream_Manager<PlainView_t> sys(0);
      if (sys.check_args(argc, argv)) sys.timestep();
    } else if (mode == 1) {
      Stream_Manager<RemoteView_t> sys(0);
      if (sys.check_args(argc, argv)) sys.timestep();
    } else if (mode == 2) {
      printf("unmanaged views not handled yet.");
      // Stream_Manager<UnmanagedView_t> sys(0);
      // if (sys.check_args(argc, argv))
      // sys.timestep();
    } else {
      printf("invalid mode selected (%d)\n", mode);
    }
  }
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
