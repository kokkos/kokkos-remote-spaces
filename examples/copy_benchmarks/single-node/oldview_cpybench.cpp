/* A micro benchmark ported mainly from Heat3D to test overhead of RMA */

#include <Kokkos_Core.hpp>
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

using PlainView_t = Kokkos::View<double***>;
using HostView_t  = typename PlainView_t::HostMirror;

struct System {
  // size of system
  int X, Y, Z;
  int dX, dY, dZ;
  int my_lo_x, my_hi_x;
  int N; /* number of timesteps */

  // Temperature and delta Temperature
  PlainView_t T1, dT;
  HostView_t T_h;

  Kokkos::DefaultExecutionSpace E_bulk;

  System(int a) : X(a) {
    // populate with defaults, set the rest in setup_subdomain.
    X = Y = Z = 200;
    my_lo_x = my_hi_x = -1;
    T_h               = HostView_t();
    T1                = PlainView_t();
    dT                = PlainView_t();
    E_bulk            = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();
    N                 = 10000;
    // E_bulk  =
    // SpaceInstance<Kokkos::DefaultExecutionSpace::memory_space>::create();
  }
  void destroy_exec_spaces() {
    SpaceInstance<Kokkos::DefaultExecutionSpace>::destroy(E_bulk);
  }

  void setup_subdomain() {
    dX = X;
    dY = Y;
    dZ = Z;

    T1  = PlainView_t("System::T1", X, Y, Z);
    T_h = HostView_t("Host::T", T1.extent(0), Y, Z);
    dT  = PlainView_t("System::dT", X, Y, Z);
    printf("My Domain: (%i %i %i) (%i %i %i)\n", 0, 0, 0, X, Y, Z);

    Kokkos::deep_copy(T_h, 0);
    Kokkos::deep_copy(T1, T_h);
    Kokkos::deep_copy(T_h, 1);
    Kokkos::deep_copy(dT, T_h);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in the X direction\n", X);
    printf("  -Y IARG: (%i) num elements in the Y direction\n", Y);
    printf("  -Z IARG: (%i) num elements in the Z direction\n", Z);
    printf("  -N IARG: (%i) num timesteps\n", Z);
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
    T1(x, y, z) += dT(x, y, z);
  }
  void Copy_TplusdT_Benchmark() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, copy_TplusdT_benchmark, int>;
    Kokkos::parallel_for(
        "TplusdT", policy_t(E_bulk, {my_lo_x, 0, 0}, {my_hi_x, Y, Z}), *this);
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
      E_bulk.fence();  // this fence is not working at the moment.
      time_b = timer.seconds();
      time_TplusdT += time_b - time_a;
      if (t % 100 == 0 || t == N) {
        double time = timer.seconds();
        printf("%d T_h(0)=%lf Time (%lf %lf)\n", t, T_h(0, 0, 0), time,
               time - old_time);
        printf("    TplusdT: %lf\n", time_TplusdT);
        old_time = time;
      }
    }
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    System sys(0);

    if (sys.check_args(argc, argv)) sys.timestep();
    sys.destroy_exec_spaces();
  }
  Kokkos::finalize();
  return 0;
}
