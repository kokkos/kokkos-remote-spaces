/* A micro benchmark ported mainly from Heat3D to test overhead of RMA */

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <assert.h>

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace &) {}
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
  static void destroy(Kokkos::Cuda &space) {
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

struct CommHelper {
  MPI_Comm comm;
  int me;
  int nranks;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);
  }
};

struct System {
  // size of system
  int X, Y, Z;
  int dX, dY, dZ;
  int my_lo_x, my_hi_x;
  int N; /* number of timesteps */
  CommHelper comm;

  // Temperature and delta Temperature
  Kokkos::View<double ***> T1, T2, dT;
  Kokkos::View<double ***>::HostMirror T_h;

  Kokkos::DefaultExecutionSpace E_bulk;

  System(MPI_Comm comm_) : comm(comm_) {
    // populate with defaults, set the rest in setup_subdomain.
    X = Y = Z = 200;
    my_lo_x = my_hi_x = -1;
    T1                = Kokkos::View<double ***>();
    T2                = Kokkos::View<double ***>();
    dT                = Kokkos::View<double ***>();
    T_h               = Kokkos::View<double ***>::HostMirror();
    E_bulk            = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();
    N                 = 10000;
  }
  void destroy_exec_spaces() {
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(E_bulk);
  }

  void setup_subdomain() {
    dX = X;
    dY = Y;
    dZ = Z;

    dX      = (X / comm.nranks);
    my_lo_x = dX * comm.me;
    my_hi_x = dX * (comm.me + 1);
    if (my_hi_x > X) my_hi_x = X;

    T1  = Kokkos::View<double ***>("System::T1", my_hi_x, Y, Z);
    T2  = Kokkos::View<double ***>("System::T2", my_hi_x, Y, Z);
    dT  = Kokkos::View<double ***>("System::dT", my_hi_x, Y, Z);
    T_h = Kokkos::View<double ***>::HostMirror("Host::T_h", my_hi_x, Y, Z);
    printf("My Domain: %i (%i %i %i) (%i %i %i)\n", comm.me, my_lo_x, Y, Z,
           my_hi_x, Y, Z);

    Kokkos::deep_copy(T1, 0);
    Kokkos::deep_copy(dT, 1);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in the X direction\n", X);
    printf("  -Y IARG: (%i) num elements in the Y direction\n", Y);
    printf("  -Z IARG: (%i) num elements in the Z direction\n", Z);
    printf("  -N IARG: (%i) num timesteps\n", Z);
  }

  // check command line args
  bool check_args(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        print_help();
        return false;
      }
    }
    for (int i = 1; i < argc; i++) { /* no i=i+1? no else if? */
      if (strcmp(argv[i], "-X") == 0) X = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Y") == 0) Y = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Z") == 0) Z = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
    }
    setup_subdomain();
    return true;
  }
  struct copy_TplusdT_benchmark {};
  KOKKOS_FUNCTION
  void operator()(copy_TplusdT_benchmark, int x, int y, int z) const {
    T2(x, y, z) = T1(x, y, z);
  }
  void Copy_TplusdT_Benchmark() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, copy_TplusdT_benchmark, int>;
    Kokkos::parallel_for("TplusdT",
                         policy_t(E_bulk, {0, 0, 0}, {my_hi_x, Y, Z}), *this);
  }

  // run time loops
  void timestep() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b     = 0;
    double time_TplusdT = 0;
    double old_time     = 0.0;
    for (int t = 0; t <= N; t++) {
      time_a = timer.seconds();
      Copy_TplusdT_Benchmark();
      Kokkos::fence();
      time_b = timer.seconds();
      time_TplusdT += time_b - time_a;
      if ((t % 100 == 0 || t == N) && (comm.me == 0)) {
        double time = timer.seconds();
        printf("%d T_h(0)=%lf Time (%lf %lf)\n", t, T_h(0, 0, 0), time,
               time - old_time);
        printf("    TplusdT: %lf\n", time_TplusdT);
        old_time = time;
      }
    }
  }
};

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    System sys(MPI_COMM_WORLD);

    if (sys.check_args(argc, argv)) sys.timestep();
    sys.destroy_exec_spaces();
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
