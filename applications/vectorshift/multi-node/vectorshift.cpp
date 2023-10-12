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

#define T int
#define OFFSET 1
#define NUM_SHIFTS 16
#define SIZE 1024

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t =
    Kokkos::View<T **, Kokkos::PartitionedLayoutLeft, RemoteSpace_t>;
using HostView_t =
    Kokkos::View<T **, Kokkos::PartitionedLayoutLeft, Kokkos::HostSpace>;

#define swap(a, b, T) \
  T tmp = a;          \
  a     = b;          \
  b     = tmp;

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

  int myPE, numPEs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myPE);
  MPI_Comm_size(MPI_COMM_WORLD, &numPEs);

  int k = OFFSET;
  int n = SIZE;

  // Excercise: Compute process-local n
  int myN = n / numPEs;

  k = (k > myN) ? myN : k;

  Kokkos::initialize(argc, argv);
  {
    RemoteView_t a("A", numPEs, myN);
    RemoteView_t b("B", numPEs, myN);

    // Adding dimension to match remote memory view (1,DIM0,...)
    HostView_t a_h("A_h", 1, myN);

    // Initialize to Zero
    Kokkos::deep_copy(a_h, 0);

    // Initialize one element to non-zero
    a_h(0, 0) = 1;

    // Copy to Remote Memory Space
    Kokkos::deep_copy(a, a_h);

    for (int shift = 0; shift < NUM_SHIFTS; ++shift) {
      // Iteration space over global array
      Kokkos::parallel_for(
          "Shift", Kokkos::RangePolicy<>(myPE * myN, (myPE + 1) * myN),
          KOKKOS_LAMBDA(const int i) {
            int j = i + k;  // Shift

            // From global array index i, dermining PE and offset within PE
            // using two-dimensional indexing
            b((j / myN) % numPEs, j % myN) = (T)a(myPE, i);
          });

      RemoteSpace_t().fence();

      swap(a, b, RemoteView_t);
    }
    // Copy back to Host memory space
    Kokkos::deep_copy(a_h, a);

    // Correctness check on corresponding PE
    if (myPE == NUM_SHIFTS * OFFSET / myN) {
      assert(a_h(0, (NUM_SHIFTS * OFFSET % myN)) == 1);
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
