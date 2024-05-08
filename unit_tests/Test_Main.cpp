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

#include <cstdlib>
#include <gtest/gtest.h>

#include <Kokkos_RemoteSpaces.hpp>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

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

  MPI_Comm mpi_comm;

#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

#ifdef KRS_ENABLE_ROCSHMEMSPACE
  roc_shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  RemoteSpace_t::fence();
  int result = RUN_ALL_TESTS();

  Kokkos::finalize();

#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
#ifdef KRS_ENABLE_ROCSHMEMSPACE
  roc_shmem_finalize();
#endif
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#else
  MPI_Finalize();
#endif

  return result;
}
