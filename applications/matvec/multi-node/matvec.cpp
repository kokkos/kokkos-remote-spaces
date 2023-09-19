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
#include <cassert>

using ORDINAL_T       = int;
using CONST_ORDINAL_T = const ORDINAL_T;
using VALUE_T         = double;

#define DEFAULT_DIM_SIZE 4096
#define LEAGUE_SIZE 32
#define TEAM_SIZE 256
#define VEC_LEN 1

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteVector_t =
    Kokkos::View<VALUE_T **, Kokkos::PartitionedLayoutLeft, RemoteSpace_t>;
using VectorHost_r_t =
    Kokkos::View<VALUE_T **, Kokkos::PartitionedLayoutLeft, Kokkos::HostSpace>;

using VectorHost_t = Kokkos::View<VALUE_T *, Kokkos::HostSpace>;
using MatrixHost_t = Kokkos::View<VALUE_T **, Kokkos::HostSpace>;
using Vector_t     = Kokkos::View<VALUE_T *>;
using Matrix_t     = Kokkos::View<VALUE_T **>;

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

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  // Vars
  float time = 0;
  ORDINAL_T nx;
  ORDINAL_T nx_proc;

  int league_size = LEAGUE_SIZE;
  int team_size   = TEAM_SIZE;
  int vec_len     = VEC_LEN;

  nx = argc > 1 ? atoi(argv[1]) : DEFAULT_DIM_SIZE;

  Kokkos::initialize(argc, argv);
  using TeamPolicy  = Kokkos::TeamPolicy<>;
  TeamPolicy policy = TeamPolicy(league_size, team_size, vec_len);
  {
    nx_proc = (nx + numRanks - 1) / numRanks;
    MatrixHost_t A_h("A_h", nx_proc, nx);
    VectorHost_t b_h("b_h", nx_proc);
    VectorHost_r_t x_h("x_h", 1, nx_proc);
    RemoteVector_t x("x", numRanks, nx_proc);

    Kokkos::deep_copy(A_h, 2.0);
    Kokkos::deep_copy(b_h, 0.0);
    Kokkos::deep_copy(x_h, 1.0);

    using DeviceMemorySpace =
        typename Kokkos::DefaultExecutionSpace::memory_space;

    auto A = Kokkos::create_mirror_view_and_copy(DeviceMemorySpace{}, A_h);
    auto b = Kokkos::create_mirror_view_and_copy(DeviceMemorySpace{}, b_h);

    // Copy host device data into global vector
    Kokkos::deep_copy(x, x_h);

    Kokkos::Timer timer;

    Kokkos::parallel_for(
        "mv", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx_proc),
                               [&](CONST_ORDINAL_T row) {
                                 double b_row = 0.0;
                                 Kokkos::parallel_reduce(
                                     Kokkos::ThreadVectorRange(team, nx),
                                     [=](CONST_ORDINAL_T col, VALUE_T &sum) {
                                       int rank   = col / nx_proc;
                                       int offset = col % nx_proc;
                                       sum += A(row, col) * x(rank, offset);
                                     },
                                     b_row);
                                 b(row) = b_row;
                               });
        });

    Kokkos::fence();
    time = timer.seconds();

    // check local results
    Kokkos::deep_copy(b_h, b);
    for (ORDINAL_T i = 0; i < nx_proc; ++i) assert(b_h(i) == 2 * nx);
    if (myRank == 0) {
      printf("%.2f sec, %.2f MB/sec\n", time,
             ((nx * nx + 2 * nx) * sizeof(VALUE_T) >> 10) / time);
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
