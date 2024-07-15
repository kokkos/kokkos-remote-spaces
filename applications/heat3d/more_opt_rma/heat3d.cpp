//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

#include <comm.hpp>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using LocalView_t   = Kokkos::View<double***>;
using RemoteView_t  = Kokkos::View<double***, RemoteSpace_t>;
using HostView_t    = typename RemoteView_t::HostMirror;
using Exec_t        = Kokkos::DefaultExecutionSpace;

struct CommHelper {
  MPI_Comm comm;
  // location in the grid
  int x, y, z;
  // Num MPI ranks in each dimension
  int nx, ny, nz;

  // Neightbor Ranks
  int up, down, left, right, front, back;

  // this process
  int me;
  int nranks;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    nx    = nranks;
    ny    = 1;
    nz    = 1;
    x     = me % nx;
    y     = (me / nx) % ny;
    z     = (me / nx / ny);
    left  = x == 0 ? -1 : me - 1;
    right = x == nx - 1 ? -1 : me + 1;
    down  = y == 0 ? -1 : me - nx;
    up    = y == ny - 1 ? -1 : me + nx;
    front = z == 0 ? -1 : me - nx * ny;
    back  = z == nz - 1 ? -1 : me + nx * ny;
#ifdef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    printf("Me: %d MyNeighbors: %i %i %i %i %i %i\n", me, left, right, down, up,
           front, back);
#endif
  }
};

struct System {
  // Communicator
  CommHelper comm;

  // size of system
  int X, Y, Z;

  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;
  int my_lo_x, my_hi_x;

  // number of timesteps
  int N;

  // interval for print
  int I;

  // Temperature and delta Temperature
  RemoteView_t T;
  LocalView_t dT, T_l;
  HostView_t T_h;

  Exec_t E_left, E_right, E_up, E_down, E_front, E_back, E_bulk;

  // Initial Temmperature
  double T0;

  // timestep width
  double dt;

  // thermal transfer coefficient
  double q;

  // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double sigma;

  // incoming power
  double P;

  // init_system
  System(MPI_Comm comm_) : comm(comm_) {
    // populate with defaults, set the rest in setup_subdomain.
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi               = X;
    Y_hi               = Y;
    Z_hi               = Z;
    my_lo_x            = 0;
    my_hi_x            = 0;
    N                  = 10000;
#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    I = 10;
#else
    I = N - 1;
#endif
    T0    = 0.0;
    dt    = 0.1;
    q     = 1.0;
    sigma = 1.0;
    P     = 1.0;

    auto exec_inst =
        Kokkos::Experimental::partition_space(Exec_t(), 1, 1, 1, 1, 1, 1, 1);
    E_left  = exec_inst[left];
    E_right = exec_inst[right];
    E_up    = exec_inst[up];
    E_down  = exec_inst[down];
    E_front = exec_inst[front];
    E_back  = exec_inst[back];
    E_bulk  = exec_inst[bulk];
  }

  void setup_subdomain() {
    X_lo = Y_lo = Z_lo = 0;
    X_hi               = X;
    Y_hi               = Y;
    Z_hi               = Z;

    auto local_range = Kokkos::Experimental::get_local_range(X);
    my_lo_x          = local_range.first;
    my_hi_x          = local_range.second;

#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    printf("My Domain: %i (%i %i %i) (%i %i %i)\n", comm.me, my_lo_x, Y_lo,
           Z_lo, my_hi_x, Y_hi, Z_hi);
#endif
    T   = RemoteView_t("System::T", X, Y, Z);
    T_l = LocalView_t(T.data(), T.extent(0), Y, Z);
    T_h = HostView_t("Host::T", T.extent(0), Y, Z);
    dT  = LocalView_t("System::dT", T.extent(0), Y, Z);

    Kokkos::deep_copy(T_h, T0);
    Kokkos::deep_copy(T, T_h);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in the X direction\n", X);
    printf("  -Y IARG: (%i) num elements in the Y direction\n", Y);
    printf("  -Z IARG: (%i) num elements in the Z direction\n", Z);
    printf("  -N IARG: (%i) num timesteps\n", N);
    printf("  -I IARG: (%i) print interval\n", I);
    printf("  -T0 FARG: (%lf) initial temperature\n", T0);
    printf("  -dt FARG: (%lf) timestep size\n", dt);
    printf("  -q FARG: (%lf) thermal conductivity\n", q);
    printf("  -sigma FARG: (%lf) thermal radiation\n", sigma);
    printf("  -P FARG: (%lf) incoming power\n", P);
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
      if (strcmp(argv[i], "-I") == 0) I = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-T0") == 0) T0 = atof(argv[i + 1]);
      if (strcmp(argv[i], "-dt") == 0) dt = atof(argv[i + 1]);
      if (strcmp(argv[i], "-q") == 0) q = atof(argv[i + 1]);
      if (strcmp(argv[i], "-sigma") == 0) sigma = atof(argv[i + 1]);
      if (strcmp(argv[i], "-P") == 0) P = atof(argv[i + 1]);
    }
    setup_subdomain();
    return true;
  }

  // compute both inner and outer updates. This function is suitable for both.
  struct ComputeInnerDT {};
  template <int Surface>
  struct ComputeSurfaceDT {};

  enum { left, right, down, up, front, back, bulk };

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T_l(x, y, z);
    dT_xyz += q * (T_l(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T_l(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T_l(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T_l(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T_l(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T_l(x, y, z + 1) - T_xyz);
    dT(x, y, z) = dT_xyz;
  }

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i,
                                  int j) const {
    int x, y, z;
    if (Surface == left) {
      x = my_lo_x;
      y = i;
      z = j;
    }
    if (Surface == right) {
      x = my_hi_x - 1;
      y = i;
      z = j;
    }
    if (Surface == down) {
      x = i;
      y = 0;
      z = j;
    }
    if (Surface == up) {
      x = i;
      y = Y - 1;
      z = j;
    }
    if (Surface == front) {
      x = i;
      y = j;
      z = 0;
    }
    if (Surface == back) {
      x = i;
      y = j;
      z = Z - 1;
    }

    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);

    // Heat conduction on body (local accesses)
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < X - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < Y - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < Z - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction on dim0 boarder (remote accesses)
    if (x == 0 && my_lo_x != 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x == (my_hi_x - 1) && my_hi_x != X)
      dT_xyz += q * (T(x + 1, y, z) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // Thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (X - 1) && my_hi_x == X) ? 1 : 0) +
                       ((y == 0) ? 1 : 0) + (y == (Y - 1) ? 1 : 0) +
                       ((z == 0) ? 1 : 0) + (z == (Z - 1) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  struct updateT {
    RemoteView_t T;
    LocalView_t dT;
    double dt;
    updateT(RemoteView_t T_, LocalView_t dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_T() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    double my_T;
    Kokkos::parallel_reduce(
        "ComputeT",
        Kokkos::Experimental::require(
            policy_t({my_lo_x, 0, 0}, {my_hi_x, Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        updateT(T, dT, dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return sum_T;
  }

  void compute_inner_dT() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = T_l.extent(0);
    int myY = T_l.extent(1);
    int myZ = T_l.extent(2);
    Kokkos::parallel_for(
        "ComputeInnerDT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  };

  void compute_surface_dT() {
    using policy_left_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<left>, int>;
    using policy_right_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<right>, int>;
    using policy_down_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<down>, int>;
    using policy_up_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<up>, int>;
    using policy_front_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<front>, int>;
    using policy_back_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<back>, int>;

    int Y = T.extent(1);
    int Z = T.extent(2);

    Kokkos::parallel_for(
        "ComputeSurfaceDT_Left",
        Kokkos::Experimental::require(
            policy_left_t(E_left, {0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Right",
        Kokkos::Experimental::require(
            policy_right_t(E_right, {0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);

    Kokkos::parallel_for(
        "ComputeSurfaceDT_Down",
        Kokkos::Experimental::require(
            policy_down_t(E_down, {my_lo_x + 1, 0}, {my_hi_x - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Up",
        Kokkos::Experimental::require(
            policy_up_t(E_up, {my_lo_x + 1, 0}, {my_hi_x - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_front",
        Kokkos::Experimental::require(
            policy_front_t(E_front, {my_lo_x + 1, 1}, {my_hi_x - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_back",
        Kokkos::Experimental::require(
            policy_back_t(E_back, {my_lo_x + 1, 1}, {my_hi_x - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  }

  // run_time_loops
  void timestep() {
    Kokkos::Timer timer;
    double time_a, time_b, time_c, time_d;
    double time_inner, time_surface, time_update, time_all;
    time_inner = time_surface = time_update = time_all = 0.0;

    double time_z = timer.seconds();

    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0; /* stop heat in halfway through */
      time_a = timer.seconds();
      compute_inner_dT();
      Kokkos::fence();
      time_b = timer.seconds();
      compute_surface_dT();
      Kokkos::fence();
      RemoteSpace_t().fence();
      time_c       = timer.seconds();
      double T_ave = update_T();
      time_d       = timer.seconds();
      time_inner += time_b - time_a;
      time_surface += time_c - time_b;
      time_update += time_d - time_c;
      T_ave /= (X * Y * Z);
#ifdef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
      if ((t % I == 0 || t == N) && (comm.me == 0))
        printf("Timestep: %i/%i t_avg: %lf\n", t, N, T_ave);
#endif
    }

    time_all = timer.seconds() - time_z;

    if (comm.me == 0) {
      printf(
          "heat3D,KokkosRemoteSpaces_more_opt,%i,%lf,%lf,%lf,%i,%lf,%lf,%lf\n",
          comm.nranks, time_inner / N, time_surface / N, time_update / N, X,
          double(2 * T.span() * sizeof(double)) / 1024 / 1024, time_all / N,
          time_all);
    }
  }
};

int main(int argc, char* argv[]) {
  comm_init(argc, argv);
  Kokkos::initialize(argc, argv);
  {
    System sys(MPI_COMM_WORLD);
    if (sys.check_args(argc, argv)) sys.timestep();
  }
  Kokkos::finalize();
  comm_fini();
  return 0;
}
