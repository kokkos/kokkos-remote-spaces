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
using LocalView_t   = Kokkos::View<double****>;
using RemoteView_t =
    Kokkos::View<double****, Kokkos::PartitionedLayoutLeft, RemoteSpace_t>;
using HostView_t =
    Kokkos::View<double****, Kokkos::PartitionedLayoutLeft, Kokkos::HostSpace>;

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

    nx = std::pow(1.0 * nranks, 1.0 / 3.0);
    while (nranks % nx != 0) nx++;

    ny = std::sqrt(1.0 * (nranks / nx));
    while ((nranks / ny) % ny != 0) ny++;

    nz    = nranks / nx / ny;
    x     = me % nx;
    y     = (me / nx) % ny;
    z     = (me / nx / ny);
    left  = (x == 0) ? -1 : me - 1;
    right = (x == nx - 1) ? -1 : me + 1;
    down  = (y == 0) ? -1 : me - nx;
    up    = (y == ny - 1) ? -1 : me + nx;
    front = (z == 0) ? -1 : me - nx * ny;
    back  = (z == nz - 1) ? -1 : me + nx * ny;

#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    printf("NumRanks: %i Me: %i Grid: %i %i %i MyPos: %i %i %i\n", nranks, me,
           nx, ny, nz, x, y, z);
    printf("Me: %d MyNeighbors: %i %i %i %i %i %i\n", me, left, right, down, up,
           front, back);
#endif
  }
};

struct System {
  // Using theoretical physicists' way of describing the system,
  // i.e. we stick to everything in as few constants as possible
  // let i and i+1 two timesteps dt apart:
  // T(x,y,z)_(i+1) =  T(x,y,z)_(i) + dT(x,y,z)*dt
  // dT(x,y,z) = q * sum_{dxdydz}( T(x + dx, y + dy, z + dz) - T(x,y,z))
  // If it's the surface of the body, add
  // dT(x,y,z) += -sigma * T(x,y,z)^4
  // If it's the z == 0 surface, add incoming radiation energy
  // dT(x,y,0) += P

  // Communicator
  CommHelper comm;

  // size of system
  int X, Y, Z;

  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;
  int X_ra, Y_ra, Z_ra;

  // number of timesteps
  int N;

  // interval for print
  int I;

  // Temperature and delta Temperature
  RemoteView_t T;
  LocalView_t dT;
  HostView_t T_h;

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
    X_ra               = X;
    Y_ra               = Y;
    Z_ra               = Z;
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
  }

  void setup_subdomain() {
    int dX =
        (X + comm.nx - 1) / comm.nx; /* Divide the space up to each MPI rank */
    X_lo = dX * comm.x;
    X_hi = X_lo + dX;
    if (X_hi > X) X_hi = X;
    X_ra = X_hi - X_lo;

    int dY = (Y + comm.ny - 1) / comm.ny; /* ceil(Y/comm.ny) */
    Y_lo   = dY * comm.y;
    Y_hi   = Y_lo + dY;
    if (Y_hi > Y) Y_hi = Y;
    Y_ra = Y_hi - Y_lo;

    int dZ = (Z + comm.nz - 1) / comm.nz;
    Z_lo   = dZ * comm.z;
    Z_hi   = Z_lo + dZ;
    if (Z_hi > Z) Z_hi = Z;
    Z_ra = Z_hi - Z_lo;

#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    printf("My Domain: %i (%i %i %i) (%i %i %i)\n", comm.me, X_lo, Y_lo, Z_lo,
           X_hi, Y_hi, Z_hi);
#endif
    T_h = HostView_t("Host::T", 1, dX, dY, dZ);
    T   = RemoteView_t("System::T", comm.nranks, dX, dY, dZ);
    dT  = LocalView_t("System::dT", 1, T.extent(1), T.extent(2), T.extent(3));
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
    for (int i = 1; i < argc; i++) { /* no i=i+1? no else if? */
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

  // Compute inner update
  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(comm.me, x, y, z);
    dT_xyz += q * (T(comm.me, x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(comm.me, x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(comm.me, x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(comm.me, x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(comm.me, x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(comm.me, x, y, z + 1) - T_xyz);

    dT(comm.me, x, y, z) = dT_xyz;
  }
  void computer_inner_dT() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = X_ra;
    int myY = Y_ra;
    int myZ = Z_ra;
    Kokkos::parallel_for(
        "ComputerInnerDT",
        Kokkos::Experimental::require(
            policy_t({1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  }

  // Compute non-exposed surface
  // Later, our dispatch makes sure that we don't hit elements twice
  enum { left, right, down, up, front, back };

  template <int Surface>
  struct ComputeSurfaceDT {};

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i,
                                  int j) const {
    int NX  = T.extent(1);
    int NY  = T.extent(2);
    int NZ  = T.extent(3);
    int myX = X_ra;
    int myY = Y_ra;
    int myZ = Z_ra;
    int x, y, z;
    if (Surface == left) {
      x = 0;
      y = i;
      z = j;
    }
    if (Surface == right) {
      x = myX - 1;
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
      y = myY - 1;
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
      z = myZ - 1;
    }

    double dT_xyz = 0.0;
    double T_xyz  = T(comm.me, x, y, z);

    // Heat conduction from the inner body
    if (x > 0) dT_xyz += q * (T(comm.me, x - 1, y, z) - T_xyz);
    if (x < myX - 1) dT_xyz += q * (T(comm.me, x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(comm.me, x, y - 1, z) - T_xyz);
    if (y < myY - 1) dT_xyz += q * (T(comm.me, x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(comm.me, x, y, z - 1) - T_xyz);
    if (z < myZ - 1) dT_xyz += q * (T(comm.me, x, y, z + 1) - T_xyz);

    // Heat conduction with Neighbors
    if (x == 0 && X_lo != 0)
      dT_xyz += q * (T(comm.left, myX - 1, y, z) - T_xyz);
    if (x == (myX - 1) && X_hi != X)
      dT_xyz += q * (T(comm.right, 0, y, z) - T_xyz);
    if (y == 0 && Y_lo != 0)
      dT_xyz += q * (T(comm.down, x, myY - 1, z) - T_xyz);
    if (y == (myY - 1) && Y_hi != Y)
      dT_xyz += q * (T(comm.up, x, 0, z) - T_xyz);
    if (z == 0 && Z_lo != 0)
      dT_xyz += q * (T(comm.front, x, y, myZ - 1) - T_xyz);
    if (z == (myZ - 1) && Z_hi != Z)
      dT_xyz += q * (T(comm.back, x, y, 0) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (myX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) +
                       ((y == (myY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) +
                       ((z == (myZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(comm.me, x, y, z) = dT_xyz;
  }
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

    int X = X_ra;
    int Y = Y_ra;
    int Z = Z_ra;

    Kokkos::parallel_for(
        "ComputeSurfaceDT_Left",
        Kokkos::Experimental::require(
            policy_left_t({0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Right",
        Kokkos::Experimental::require(
            policy_right_t({0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Down",
        Kokkos::Experimental::require(
            policy_down_t({1, 0}, {X - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Up",
        Kokkos::Experimental::require(
            policy_up_t({1, 0}, {X - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Front",
        Kokkos::Experimental::require(
            policy_front_t({1, 1}, {X - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Back",
        Kokkos::Experimental::require(
            policy_back_t({1, 1}, {X - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  }

  // Some compilers have deduction issues if this were just a tagget operator
  // So it is instead a full Functor
  struct computeT {
    RemoteView_t T;
    LocalView_t dT;
    double dt;
    computeT(RemoteView_t T_, LocalView_t dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int me, int x, int y, int z, double& sum_T) const {
      sum_T += T(me, x, y, z);
      T(me, x, y, z) += dt * dT(me, x, y, z);
    }
  };

  double update_T() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<4>, Kokkos::IndexType<int>>;
    int X = X_ra;
    int Y = Y_ra;
    int Z = Z_ra;
    double my_T;
    Kokkos::parallel_reduce(
        "ComputeT",
        Kokkos::Experimental::require(
            policy_t({comm.me, 0, 0, 0}, {comm.me + 1, X, Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        computeT(T, dT, dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM,
                  comm.comm); /* also a barrier */
    return sum_T;
  }

  // run time loops
  void timestep() {
    Kokkos::Timer timer;
    double old_time = 0.0;
    double GUPs     = 0.0;
    double time_a, time_b, time_c, time_update, time_compute, time_all;
    time_all = time_update = time_compute = 0.0;
    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0; /* stop heat in halfway through */
      time_a = timer.seconds();
      computer_inner_dT();
      RemoteSpace_t().fence();
      compute_surface_dT();
      RemoteSpace_t().fence();
      time_b       = timer.seconds();
      double T_ave = update_T();
      time_c       = timer.seconds();
      time_compute += time_b - time_a;
      time_update += time_c - time_b;
      T_ave /= 1e-9 * (X * Y * Z);
      if ((t % I == 0 || t == N) && (comm.me == 0)) {
        double time = timer.seconds();
        time_all += time - old_time;
        GUPs += 1e-9 * (dT.size() / time_compute);
#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
        if ((t % I == 0 || t == N) && (comm.me == 0)) {
#else
        if ((t == N) && (comm.me == 0)) {
#endif
          printf(
              "heat3D,KokkosRemoteSpaces_partitioned,%i,%i,%lf,%lf,%lf,%lf,%lf,"
              "%lf,%lf,%i,%f\n",
              comm.nranks, t, T_ave, 0.0, time_compute, time_update,
              time - old_time, /* time last iter */
              time_all,        /* current runtime  */
              GUPs / t, X, 1e-6 * (dT.size() * sizeof(double)));
          old_time = time;
        }
      }
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
