#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>

using RemoteSpace_t      = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteDoubleView_t = Kokkos::View<double *, RemoteSpace_t>;

template <typename ViewType_t>
struct Bench_Manager {
  ViewType_t X;
  int len = 0;
  int me;
  int partner;
  int nranks;
  int my_start_i;
  int partner_start_i;
  int interval;
  int iterations;
  int t;

  double start_time, end_time, benchmark_time;

  Bench_Manager(int l, int iters) : len(l), iterations(iters) {
    X = ViewType_t("Device::X", len);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    partner         = (me + (nranks / 2)) % nranks;
    interval        = len / nranks;
    my_start_i      = me * interval;
    partner_start_i = partner * interval;

    printf("I am rank %d with access to [%d,%d) and my partner is %d\n", me,
           my_start_i, my_start_i + interval, partner);
  }

  struct b_put {};
  struct b_get {};
  using policy_put_t = Kokkos::RangePolicy<b_put, size_t>;
  using policy_get_t = Kokkos::RangePolicy<b_get, size_t>;
  KOKKOS_FUNCTION
  void operator()(const b_put &, const size_t i) const {
    // printf("copying X(%d) to X(%d)\n", (int)(my_start_i+i),
    // (int)(partner_start_i+i));
    X(partner_start_i + i) = X(my_start_i + i);
  }
  KOKKOS_FUNCTION
  void operator()(const b_get &, const size_t i) const {
    X(my_start_i + i) = X(partner_start_i + i);
  }

  void benchmark_put_lat(void) {
    Kokkos::Timer timer;  // TODO: how come this can't go up top?
    benchmark_time = 0.0;
    if (me < nranks / 2) {
      Kokkos::parallel_for("KOKKOS::put_lat WARMUP",
                           policy_put_t(my_start_i, my_start_i + interval),
                           *this);
    }
    RemoteSpace_t().fence();
    Kokkos::fence();
    for (t = 1; t <= iterations; t++) {
      if (t % 100 == 0 && me == 0) printf("iteration %d\n", t);
      start_time = timer.seconds();
      if (me < nranks / 2) {
        Kokkos::parallel_for("KOKKOS::put_lat",
                             policy_put_t(my_start_i, my_start_i + interval),
                             *this);
      }
      RemoteSpace_t().fence();
      Kokkos::fence();
      end_time = timer.seconds();
      benchmark_time += (end_time - start_time);
    }
    if (me == 0) printf("total benchmark time: %lf seconds\n", benchmark_time);
  }
  void benchmark_get_lat(void) {
    Kokkos::Timer timer;  // TODO: how come this can't go up top?
    benchmark_time = 0.0;
    if (me < nranks / 2) {
      Kokkos::parallel_for("KOKKOS::get_lat WARMUP",
                           policy_get_t(my_start_i, my_start_i + interval),
                           *this);
    }
    RemoteSpace_t().fence();
    Kokkos::fence();
    for (t = 1; t <= iterations; t++) {
      if (t % 100 == 0 && me == 0) printf("iteration %d\n", t);
      start_time = timer.seconds();
      if (me < nranks / 2) {
        Kokkos::parallel_for("KOKKOS::get_lat",
                             policy_get_t(my_start_i, my_start_i + interval),
                             *this);
      }
      RemoteSpace_t().fence();
      Kokkos::fence();
      end_time = timer.seconds();
      benchmark_time += (end_time - start_time);
    }
    if (me == 0) printf("total benchmark time: %lf seconds\n", benchmark_time);
  }
  void benchmark_put_mr(void) {
    Kokkos::Timer timer;  // TODO: how come this can't go up top?
    benchmark_time = 0.0;
    if (me < nranks / 2) {
      Kokkos::parallel_for("KOKKOS::put_mr WARMUP",
                           policy_put_t(my_start_i, my_start_i + interval),
                           *this);
    }
    RemoteSpace_t().fence();
    Kokkos::fence();
    for (t = 1; t <= iterations; t++) {
      if (t % 100 == 0 && me == 0) printf("iteration %d\n", t);
      start_time = timer.seconds();
      if (me < nranks / 2) {
        Kokkos::parallel_for("KOKKOS::put_mr",
                             policy_put_t(my_start_i, my_start_i + interval),
                             *this);
      }
      RemoteSpace_t().fence();
      Kokkos::fence();
      end_time = timer.seconds();
      benchmark_time += (end_time - start_time);
    }
    if (me == 0) printf("total benchmark time: %lf seconds\n", benchmark_time);
  }
  void benchmark_get_mr(void) {
    Kokkos::Timer timer;  // TODO: how come this can't go up top?
    benchmark_time = 0.0;
    if (me < nranks / 2) {
      Kokkos::parallel_for("KOKKOS::get_mr WARMUP",
                           policy_get_t(my_start_i, my_start_i + interval),
                           *this);
    }
    RemoteSpace_t().fence();
    Kokkos::fence();
    for (t = 1; t <= iterations; t++) {
      if (t % 100 == 0 && me == 0) printf("iteration %d\n", t);
      start_time = timer.seconds();
      if (me < nranks / 2) {
        Kokkos::parallel_for("KOKKOS::get_mr",
                             policy_get_t(my_start_i, my_start_i + interval),
                             *this);
      }
      RemoteSpace_t().fence();
      Kokkos::fence();
      end_time = timer.seconds();
      benchmark_time += (end_time - start_time);
    }
    if (me == 0) printf("total benchmark time: %lf seconds\n", benchmark_time);
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
    int mode  = 0;
    int len   = 1e4;
    int iters = 1e4;
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
        break;
      }
      if (strcmp(argv[i], "-m") == 0)
        mode = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-l") == 0)
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

    Bench_Manager<RemoteDoubleView_t> sys(len, iters);
    if (mode == 0) {
      sys.benchmark_put_lat();
    } else if (mode == 1) {
      sys.benchmark_get_lat();
    } else if (mode == 2) {
      sys.benchmark_put_mr();
    } else if (mode == 3) {
      sys.benchmark_get_mr();
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
