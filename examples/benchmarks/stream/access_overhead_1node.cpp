/* A micro benchmark ported mainly from Heat3D to test overhead of RMA */

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double *, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double *>;
using UnmanagedView_t =
    Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView_t = typename RemoteView_t::HostMirror;
using policy_t   = Kokkos::RangePolicy<int>;

template <typename AViewType_t, typename BViewType_t>
struct Stream_Manager {
  int N;           /* size of vector */
  int iterations;  /* number of iterations */
  bool plusequals; /* by default, do +=. But do ++ if asked. */

  AViewType_t A;
  BViewType_t B;

  Stream_Manager(AViewType_t &a, BViewType_t &b, int n, int i, bool pe)
      : A(a), B(b), N(n), iterations(i), plusequals(pe) {}

  struct stream_plusequals_add {
    AViewType_t A;
    BViewType_t B;
    stream_plusequals_add(AViewType_t A_, BViewType_t B_) : A(A_), B(B_) { ; }
    KOKKOS_FUNCTION
    void operator()(int i) const { A(i) += B(i); }
  };

  double stream_plusequals_benchmark(void) {
    Kokkos::Timer timer;
    double time_begin, time_end;
    time_begin = timer.seconds();

    Kokkos::parallel_for("plusequals", policy_t({0}, {N}),
                         stream_plusequals_add(A, B));

    RemoteSpace_t().fence();
    time_end = timer.seconds();
    return time_end - time_begin;
  }

  struct stream_plusplus_add {
    AViewType_t A;
    stream_plusplus_add(AViewType_t A_) : A(A_) { ; }
    KOKKOS_FUNCTION
    void operator()(int i) const { A(i) += 1; }
  };

  double stream_plusplus_benchmark(void) {
    Kokkos::Timer timer;
    double time_begin, time_end;
    time_begin = timer.seconds();

    Kokkos::parallel_for("plusplus", policy_t({0}, {N}),
                         stream_plusplus_add(A));

    RemoteSpace_t().fence();
    time_end = timer.seconds();
    return time_end - time_begin;
  }

  // run stream benchmark
  void benchmark() {
    Kokkos::Timer timer;
    double time_stream = 0;
    double old_time    = 0.0;
    /* warmup run */
    for (int t = 1; t <= 1; t++) {
      if (plusequals)
        stream_plusequals_benchmark();
      else
        stream_plusplus_benchmark();
    }
    for (int t = 1; t <= iterations; t++) {
      if (plusequals)
        time_stream += stream_plusequals_benchmark();
      else
        time_stream += stream_plusplus_benchmark();

      if ((t % 400 == 0 || t == iterations)) {
        double time = timer.seconds();
        printf("%d Time (%lf %lf)\n", t, time, time - old_time);
        printf("    stream: %lf\n", time_stream);
        old_time = time;
      }
    }
    double elements_updated = 1.0 * iterations * N;
    double gups             = elements_updated * 1e-9 / time_stream;
    printf("GUPs: %lf bandwidth: %lf\n", gups, gups * 8);
  }
};

int main(int argc, char *argv[]) {
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
    /* use 'mode' variables to pack any of twelve benchmarks into one here */
    int a_mode = 0;
    int b_mode = -1;
    int N;
    int iterations;
    iterations = 10000;
    N          = 8000000;
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        printf("Options (default):\n");
        printf("  -i IARG: (%i) num iterations of streams\n", iterations);
        printf("  -N IARG: (%i) number of elements in the V vector\n", N);
        printf("  -a IARG: (%d) which mode to choose for view A\n", a_mode);
        printf("  -b IARG: (%d) which mode to choose for view B\n", b_mode);
        printf("modes:\n");
        printf("  0: Kokkos (Normal)  View\n");
        printf("  1: Kokkos Remote    View\n");
        printf("  2: Kokkos Unmanaged View\n");
        printf("  if nothing passed for view b, then benchmark A(i) ++\n");
        return 0;
      }
      if (strcmp(argv[i], "-a") == 0) a_mode = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-atype") == 0) a_mode = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-b") == 0) b_mode = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-btype") == 0) b_mode = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-iters") == 0) iterations = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-i") == 0) iterations = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
    }

    if (a_mode == 0) {
      using AType_t = PlainView_t;
      AType_t a;
      a = AType_t("System::A", N);
      if (b_mode == 0) {
        PlainView_t b;
        b = PlainView_t("System::B", N);
        Stream_Manager<AType_t, PlainView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else if (b_mode == 1) {
        RemoteView_t b;
        b = RemoteView_t("System::B", N);
        Stream_Manager<AType_t, RemoteView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else if (b_mode == 2) {
        UnmanagedView_t b;
        RemoteView_t c;
        c = RemoteView_t("System::B", N);
        b = UnmanagedView_t(c.data(), N);
        Stream_Manager<AType_t, UnmanagedView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else {
        Stream_Manager<AType_t, AType_t> sys(a, a, N, iterations, false);
        sys.benchmark();
      }
    }
    if (a_mode == 1) {
      using AType_t = RemoteView_t;
      AType_t a;
      a = AType_t("System::A", N);
      if (b_mode == 0) {
        PlainView_t b;
        b = PlainView_t("System::B", N);
        Stream_Manager<AType_t, PlainView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else if (b_mode == 1) {
        RemoteView_t b;
        b = RemoteView_t("System::B", N);
        Stream_Manager<AType_t, RemoteView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else if (b_mode == 2) {
        UnmanagedView_t b;
        RemoteView_t c;
        c = RemoteView_t("System::B", N);
        b = UnmanagedView_t(c.data(), N);
        Stream_Manager<AType_t, UnmanagedView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else {
        Stream_Manager<AType_t, AType_t> sys(a, a, N, iterations, false);
        sys.benchmark();
      }
    }
    if (a_mode == 2) {
      using AType_t = UnmanagedView_t;
      AType_t a;
      RemoteView_t d;
      d = RemoteView_t("System::A", N);
      a = AType_t(d.data(), N);
      if (b_mode == 0) {
        PlainView_t b;
        b = PlainView_t("System::B", N);
        Stream_Manager<AType_t, PlainView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else if (b_mode == 1) {
        RemoteView_t b;
        b = RemoteView_t("System::B", N);
        Stream_Manager<AType_t, RemoteView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      }
      if (b_mode == 2) {
        UnmanagedView_t b;
        RemoteView_t c;
        c = RemoteView_t("System::B", N);
        b = UnmanagedView_t(c.data(), N);
        Stream_Manager<AType_t, UnmanagedView_t> sys(a, b, N, iterations, true);
        sys.benchmark();
      } else {
        Stream_Manager<AType_t, AType_t> sys(a, a, N, iterations, false);
        sys.benchmark();
      }
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
