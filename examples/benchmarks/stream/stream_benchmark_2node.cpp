/* A micro benchmark ported mainly from Heat3D to test overhead of RMA */
/*
 * UNLIKE the nnodes version of the stream benchmark, this version *only*
 * has mpi rank 0 add mpi rank 1's data
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

struct CommInfo {
  MPI_Comm comm;

  int lneighbor, rneighbor;

  int me;      // this process
  int nranks;  // total processes

  CommInfo(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    lneighbor = (me == 0) ? nranks - 1 : me - 1;
    rneighbor = (me == nranks - 1) ? 0 : me + 1;

    printf("NumRanks: %i Me: %i", nranks, me);
    printf(" rneighbor rank: %i\n", rneighbor);
  }
};

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double *, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double *>;
using policy_t      = Kokkos::RangePolicy<int>;

template <typename ViewType_t>
struct Stream_Manager {
  int N;          /* size of vector */
  int iterations; /* number of iterations */
  int my_min_i;   /* my rank's minimum i */
  int my_max_i;   /* my rank's maximum i */
  int interval;
  int rstart;
  int rend;
  int rinterval;
  bool remote;

  MPI_Request mpi_request_recv;
  MPI_Request mpi_request_send;

  ViewType_t A;
  ViewType_t B; /* B, an initially empty view only used for mpi version */

  CommInfo comm;

  Stream_Manager(MPI_Comm comm_, ViewType_t &a, ViewType_t &b, int n, int i)
      : comm(comm_), A(a), B(b), N(n), iterations(i) {
    if (std::is_same<ViewType_t, PlainView_t>::value) {
      remote = false;
    }
    if (std::is_same<ViewType_t, RemoteView_t>::value) {
      remote = true;
    }

    if (remote) {
      interval = (N + comm.nranks - 1) / comm.nranks;
      my_min_i = interval * comm.me;
      my_max_i = (interval) * (comm.me + 1); /* interval non-inclusive */
      if (my_max_i > N) {
        my_max_i = N;
      }

      rstart = (interval)*comm.rneighbor;
      rend   = (interval) * (comm.rneighbor + 1);
      if (rend > N) {
        rend = N;
      }
      rinterval = rend - rstart;

      interval = my_max_i - my_min_i;
    } else {
      rinterval = interval = N;
      my_min_i = rstart = 0;

      if (N % interval != 0) {
        if (comm.me == comm.nranks - 1) {
          interval = N % interval;
        }
        if (comm.rneighbor == comm.nranks - 1) {
          rinterval = N % interval;
        }
      }

      my_max_i = interval;
      rend     = rinterval;
    }
    printf("my_min_i: %d my_max_i: %d interval: %d\n", my_min_i, my_max_i,
           interval);
  }

  struct remote_add {
    ViewType_t A;
    int my_min_i, rstart;

    remote_add(ViewType_t A_, int my_min_i_, int rstart_)
        : A(A_), my_min_i(my_min_i_), rstart(rstart_) {
      ;
    }
    KOKKOS_FUNCTION
    void operator()(int i) const { A(my_min_i + i) += A(rstart + i); }
  };

  double remote_benchmark(int minterval) {
    Kokkos::Timer timer;
    double time_begin, time_end;
    time_begin = timer.seconds();
    if (comm.me == 0) {
      Kokkos::parallel_for("remote_stream", policy_t({0}, {minterval}),
                           remote_add(A, my_min_i, rstart));
    }
    RemoteSpace_t().fence();
    time_end = timer.seconds();
    return time_end - time_begin;
  }

  struct mpi_add {
    ViewType_t A, B;

    mpi_add(ViewType_t A_, ViewType_t B_) : A(A_), B(B_) { ; }
    KOKKOS_FUNCTION
    void operator()(int i) const { A(i) += B(i); }
  };

  double mpi_benchmark(int minterval) {
    Kokkos::Timer timer;
    double time_begin, time_end;
    time_begin = timer.seconds();
    if (comm.me == 1) {
      MPI_Isend(A.data(), minterval, MPI_DOUBLE, 0, 1, comm.comm,
                &mpi_request_send);
      MPI_Waitall(1, &mpi_request_send, MPI_STATUSES_IGNORE);
    }
    if (comm.me == 0) {
      MPI_Irecv(B.data(), minterval, MPI_DOUBLE, 1, 1, comm.comm,
                &mpi_request_recv);
      MPI_Waitall(1, &mpi_request_recv, MPI_STATUSES_IGNORE);
    }

    if (comm.me == 0) {
      Kokkos::parallel_for("mpi_stream", policy_t({0}, {minterval}),
                           mpi_add(A, B));
    }
    RemoteSpace_t().fence();
    MPI_Barrier(MPI_COMM_WORLD);
    time_end = timer.seconds();
    return time_end - time_begin;
  }

  // run stream benchmark
  void benchmark() {
    Kokkos::Timer timer;
    double time_stream = 0;
    double old_time    = 0.0;
    int minterval;
    minterval = rinterval < interval ? rinterval : interval;
    /* warmup run */
    for (int t = 1; t <= 1; t++) {
      if (remote)
        remote_benchmark(minterval);
      else
        mpi_benchmark(minterval);
    }
    for (int t = 1; t <= iterations; t++) {
      if (remote)
        time_stream += remote_benchmark(minterval);
      else
        time_stream += mpi_benchmark(minterval);

      if ((t % 100 == 0 || t == iterations) && (comm.me == 0)) {
        double time = timer.seconds();
        printf("%d Time (%lf %lf)\n", t, time, time - old_time);
        printf("    stream: %lf\n", time_stream);
        old_time = time;
      }
    }
    if (comm.me == 0) {
      double elements_updated = 1.0 * iterations * minterval;
      double gups             = elements_updated * 1e-9 / time_stream;
      printf("GUPs: %lf bandwidth: %lf\n", gups, gups * 8);
    }
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

  int nranks;
  int myrank;
  MPI_Comm_size(mpi_comm, &nranks);
  MPI_Comm_rank(mpi_comm, &myrank);
  if (nranks < 2) {
    printf("required *exactly* 2 processes for this benchmark\n");
    return 0;
  }

  Kokkos::initialize(argc, argv);
  {
    /* use 'mode' variable to pack any of two benchmarks into one here */
    int mode = 0;
    int N;
    int iterations;
    iterations = 1e3;
    N          = 1e7;
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        printf("Options (default):\n");
        printf("  -i IARG: (%i) num iterations of streams\n", iterations);
        printf("  -N IARG: (%i) number of elements in the V vector\n", N);
        printf("  -m IARG: (%d) which mode to choose for views\n", mode);
        printf("modes:\n");
        printf("  0: Kokkos (Normal)  View\n");
        printf("  1: Kokkos Remote    View\n");
        return 0;
      }
      if (strcmp(argv[i], "-m") == 0) mode = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-mode") == 0) mode = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-iters") == 0) iterations = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-i") == 0) iterations = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-l") == 0) N = atoi(argv[i + 1]);
    }

    if (mode == 0) {
      using Type_t = PlainView_t;
      Type_t a;
      Type_t b;
      Type_t::HostMirror c;
      int view_size = (N + nranks - 1) / nranks;
      a             = Type_t("System::A", view_size);
      b             = Type_t("System::B", view_size);
      c             = Type_t::HostMirror("Host::init", view_size);
      Kokkos::deep_copy(c, myrank);
      Kokkos::deep_copy(a, c);
      Kokkos::deep_copy(b, c);
      Stream_Manager<Type_t> sys(mpi_comm, a, b, view_size, iterations);
      sys.benchmark();
    }
    if (mode == 1) {
      using Type_t = RemoteView_t;
      Type_t a;
      Type_t b;
      Type_t::HostMirror c;
      int host_view_size = (N + nranks - 1) / nranks;
      a                  = Type_t("System::A", N);
      b                  = Type_t("System::B", N);

      c = Type_t::HostMirror("Host::init", host_view_size);
      Kokkos::deep_copy(c, myrank);
      Kokkos::deep_copy(a, c);
      Kokkos::deep_copy(b, c);
      Stream_Manager<Type_t> sys(mpi_comm, a, b, N, iterations);
      sys.benchmark();
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
