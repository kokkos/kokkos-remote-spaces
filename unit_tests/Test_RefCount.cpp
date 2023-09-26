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

#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class DataType, class RemoteSpace>
void test_reference_counting() {
  int my_rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  Kokkos::View<DataType*, RemoteSpace> outer("outer", num_ranks,
                                             10 * sizeof(DataType));
  {
    Kokkos::View<DataType*, RemoteSpace> inner = outer;
    ASSERT_EQ(inner.use_count(), 2);
  }
  ASSERT_EQ(outer.use_count(), 1);
}

TEST(TEST_CATEGORY, test_reference_counting) {
  test_reference_counting<int, RemoteSpace_t>();
  test_reference_counting<double, RemoteSpace_t>();

  MPI_Barrier(MPI_COMM_WORLD);
}
