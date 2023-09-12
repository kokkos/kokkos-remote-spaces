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

using RemoteMemSpace = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class ViewType>
void check_extents(ViewType view, int r) {
  int rank = view.rank;
  ASSERT_EQ(r, rank);
}

template <class ViewType, class... Args>
void check_extents(ViewType view, int r, int N, Args... args) {
  if (r != 0) ASSERT_EQ(view.extent(r), N);
  check_extents(view, r + 1, args...);
}

template <class DataType, class RemoteSpace, class... Args>
void test_allocate_symmetric_remote_view_by_rank(Args... args) {
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  using RemoteView_t = Kokkos::View<DataType, RemoteSpace>;

  // Check explicit mem space allocation through API call
  RemoteView_t view("MyRemoteView", numRanks, args...);
  check_extents(view, 0, numRanks, args...);

  // Check implicit memort space allocaton
  view = RemoteView_t("MyRemoteView", numRanks, args...);
  check_extents(view, 0, numRanks, args...);
}

TEST(TEST_CATEGORY, test_allocate_symmetric_remote_view_by_rank) {
  test_allocate_symmetric_remote_view_by_rank<double *, RemoteMemSpace>();
  test_allocate_symmetric_remote_view_by_rank<double **, RemoteMemSpace>(113);
  test_allocate_symmetric_remote_view_by_rank<double ***, RemoteMemSpace>(7, 5);
  test_allocate_symmetric_remote_view_by_rank<double ****, RemoteMemSpace>(
      9, 10, 7);
  test_allocate_symmetric_remote_view_by_rank<double *****, RemoteMemSpace>(
      9, 10, 7, 2);
  test_allocate_symmetric_remote_view_by_rank<double ******, RemoteMemSpace>(
      9, 10, 7, 2, 1);
  test_allocate_symmetric_remote_view_by_rank<double *******, RemoteMemSpace>(
      9, 10, 7, 2, 1, 1);

  test_allocate_symmetric_remote_view_by_rank<int *, RemoteMemSpace>();
  test_allocate_symmetric_remote_view_by_rank<int **, RemoteMemSpace>(113);
  test_allocate_symmetric_remote_view_by_rank<int ***, RemoteMemSpace>(7, 5);
  test_allocate_symmetric_remote_view_by_rank<int ****, RemoteMemSpace>(9, 10,
                                                                        7);
  test_allocate_symmetric_remote_view_by_rank<int *****, RemoteMemSpace>(9, 10,
                                                                         7, 2);
  test_allocate_symmetric_remote_view_by_rank<int ******, RemoteMemSpace>(
      9, 10, 7, 2, 1);
  test_allocate_symmetric_remote_view_by_rank<int *******, RemoteMemSpace>(
      9, 10, 7, 2, 1, 1);
}
