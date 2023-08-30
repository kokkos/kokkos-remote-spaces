#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
/* allows avoiding of code duplication */
#define LATENCY_BIT 1
#define MESSAGE_RATE_BIT 2
#define BOTH_BITS LATENCY_BIT | MESSAGE_RATE_BIT

template <typename dType>
struct Bench_Manager {
  using RemoteView_t = Kokkos::View<dType *, RemoteSpace_t>;
  RemoteView_t X;
  int len = 0;
  int me;
  int partner;
  int nranks;
  int my_start_i;
  int partner_start_i;
  int interval; /* all processes always have the same interval */
  int iterations;
  int t;
  int dType_size;

  double start_time, end_time, benchmark_time;

  Bench_Manager(int l, int iters) : len(l), iterations(iters) {
    X = RemoteView_t("Device::X", len);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    partner         = (me + (nranks / 2)) % nranks;
    interval        = len / nranks;
    my_start_i      = me * interval;
    partner_start_i = partner * interval;

    dType_size = sizeof(dType);

    printf("I am rank %d with access to [%d,%d) and my partner is %d\n", me,
           my_start_i, my_start_i + interval, partner);
  }

  struct b_put {};
  struct b_get {};
  using policy_put_t = Kokkos::RangePolicy<b_put, size_t>;
  using policy_get_t = Kokkos::RangePolicy<b_get, size_t>;
  KOKKOS_FUNCTION
  void operator()(const b_put &, const size_t i) const {
    X(partner_start_i + i) = X(my_start_i + i);
  }
  KOKKOS_FUNCTION
  void operator()(const b_get &, const size_t i) const {
    X(my_start_i + i) = X(partner_start_i + i);
  }

  template <typename policy_t>
  void benchmark(int btype_num) {
    Kokkos::Timer timer;
    benchmark_time = 0.0;
    if (me < nranks / 2) {
      Kokkos::parallel_for("KOKKOS::bench WARMUP",
                           policy_t(my_start_i, my_start_i + interval), *this);
    }
    RemoteSpace_t().fence();
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
    for (t = 1; t <= iterations; t++) {
      if (t % 100 == 0 && me == 0) printf("iteration %d\n", t);
      start_time = timer.seconds();
      if (me < nranks / 2) {
        Kokkos::parallel_for("KOKKOS::bench",
                             policy_t(my_start_i, my_start_i + interval),
                             *this);
      }
      RemoteSpace_t().fence();
      Kokkos::fence();
      MPI_Barrier(MPI_COMM_WORLD);
      end_time = timer.seconds();
      benchmark_time += (end_time - start_time);
    }
    if (me == 0) {
      if (std::is_same<policy_t, policy_put_t>::value) {
        printf("operation: PUT\n");
      }
      if (std::is_same<policy_t, policy_get_t>::value) {
        printf("operation: GET\n");
      }

      double message_bytes = 1.0 * interval * dType_size * iterations;
      printf("dtype_bytes %d message_bytes %.0lf\n", dType_size, message_bytes);
      printf("total benchmark time: %lf seconds\n", benchmark_time);

      if (btype_num & LATENCY_BIT) {
        double avg_time_us = benchmark_time / iterations * 1e6;
        printf("latency: %lf (microseconds)\n", avg_time_us);
      }
      if (btype_num & MESSAGE_RATE_BIT) {
        double message_gbytes_per_sec = message_bytes * 1e-9 / benchmark_time;
        printf("bandwidth: %lf (GB/s)\n", message_gbytes_per_sec);
      }
    }
  }

  void benchmark_put_lat(void) { benchmark<policy_put_t>(LATENCY_BIT); }
  void benchmark_get_lat(void) { benchmark<policy_get_t>(LATENCY_BIT); }
  void benchmark_put_mr(void) { benchmark<policy_put_t>(MESSAGE_RATE_BIT); }
  void benchmark_get_mr(void) { benchmark<policy_get_t>(MESSAGE_RATE_BIT); }
  void benchmark_put_all(void) { benchmark<policy_put_t>(BOTH_BITS); }
  void benchmark_get_all(void) { benchmark<policy_get_t>(BOTH_BITS); }
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
    int mode  = 4;
    int len   = 1e6;
    int iters = 1e3;
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        printf("  -m IARG: (%d) which mode to choose\n", mode);
        printf("  -l IARG: (%d) what to use for length\n", mode);
        printf("  -i IARG: (%d) how many iterations to run for\n", mode);
        printf("((modes:))\n");
        printf("  0: osu put latency\n");
        printf("  1: osu get latency\n");
        printf("  2: osu put message rate\n");
        printf("  3: osu get message rate\n");
        printf("  4: osu put both benchmarks\n");
        printf("  5: osu get both benchmarks\n");
        break;
      }
      if (strcmp(argv[i], "-m") == 0)
        mode = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-l") == 0)
        len = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-N") == 0)
        len = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-i") == 0)
        iters = atoi(argv[i + 1]);
    }

    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks % 2) {
      printf("error: must have nranks be even.\n");
      exit(1);
    }
    if (len % nranks) {
      printf("error: must have len be a multiple of nranks.\n");
      exit(1);
    }

    /* different dtypes are not supported yet, but easily could be extended. */
    Bench_Manager<double> sys(len, iters);
    if (mode == 0) {
      sys.benchmark_put_lat();
    } else if (mode == 1) {
      sys.benchmark_get_lat();
    } else if (mode == 2) {
      sys.benchmark_put_mr();
    } else if (mode == 3) {
      sys.benchmark_get_mr();
    } else if (mode == 4) {
      sys.benchmark_put_all();
    } else if (mode == 5) {
      sys.benchmark_get_all();
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
