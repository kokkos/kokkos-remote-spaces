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

#define is_View_PL(ViewType) \
  std::enable_if_t<Kokkos::Experimental::Is_Partitioned_Layout<ViewType>::value>
#define is_View_GL(ViewType) \
  std::enable_if_t<          \
      !Kokkos::Experimental::Is_Partitioned_Layout<ViewType>::value>

#define is_Layout_PL(Layout) \
  std::enable_if_t<Kokkos::Experimental::Is_Partitioned_Layout<Layout>::value>
#define is_Layout_GL(Layout) \
  std::enable_if_t<!Kokkos::Experimental::Is_Partitioned_Layout<Layout>::value>

template <class ViewType>
void check_extents(ViewType view, int r) {
  int rank = view.rank;
  ASSERT_EQ(r, rank);
}

template <class ViewType, class... Args>
is_View_PL(ViewType) check_extents(ViewType view, int r, int N, Args... args) {
  if (r == 0) ASSERT_EQ(view.extent(r), 1);
  if (r != 0) ASSERT_EQ(view.extent(r), N);
  check_extents(view, r + 1, args...);
}

template <class ViewType, class... Args>
is_View_GL(ViewType) check_extents(ViewType view, int r, int N, Args... args) {
  if (r == 0) {
    auto m     = Kokkos::Experimental::get_local_range(N).second;
    auto range = m - Kokkos::Experimental::get_local_range(N).first;
    ASSERT_GE(view.extent(r), range);
  }
  if (r != 0) ASSERT_EQ(view.extent(r), N);
  check_extents(view, r + 1, args...);
}

template <class DataType, class Layout, class RemoteSpace, class... Args>
is_Layout_PL(Layout) test_allocate_symmetric_remote_view_by_rank(Args... args) {
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  using RemoteView_t = Kokkos::View<DataType, Layout, RemoteSpace>;

  // Check explicit mem space allocation through API call
  RemoteView_t view("MyRemoteView", numRanks, args...);
  check_extents(view, 0, numRanks, args...);

  // Check implicit memory space allocaton
  view = RemoteView_t("MyRemoteView", numRanks, args...);
  check_extents(view, 0, numRanks, args...);
}

template <class DataType, class Layout, class RemoteSpace, class... Args>
is_Layout_GL(Layout) test_allocate_symmetric_remote_view_by_rank(Args... args) {
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  using RemoteView_t = Kokkos::View<DataType, Layout, RemoteSpace>;

  // Check explicit mem space allocation through API call
  RemoteView_t view("MyRemoteView", args...);
  check_extents(view, 0, args...);

  // Check implicit memory space allocaton
  view = RemoteView_t("MyRemoteView", args...);
  check_extents(view, 0, args...);
}

#define GEN_BLOCK_1(Type, Layout, Space)                                       \
  test_allocate_symmetric_remote_view_by_rank<Type *, Layout,                  \
                                              RemoteSpace_t>();                \
  test_allocate_symmetric_remote_view_by_rank<Type **, Layout, RemoteSpace_t>( \
      113);                                                                    \
  test_allocate_symmetric_remote_view_by_rank<Type ***, Layout,                \
                                              RemoteSpace_t>(7, 5);            \
  test_allocate_symmetric_remote_view_by_rank<Type ****, Layout,               \
                                              RemoteSpace_t>(9, 10, 7);        \
  test_allocate_symmetric_remote_view_by_rank<Type *****, Layout,              \
                                              RemoteSpace_t>(9, 10, 7, 2);     \
  test_allocate_symmetric_remote_view_by_rank<Type ******, Layout,             \
                                              RemoteSpace_t>(9, 10, 7, 2, 1);  \
  test_allocate_symmetric_remote_view_by_rank<Type *******, Layout,            \
                                              RemoteSpace_t>(9, 10, 7, 2, 1,   \
                                                             1);

#define GEN_BLOCK_2(Type, Layout, Space)                                       \
  test_allocate_symmetric_remote_view_by_rank<Type, Layout, RemoteSpace_t>();  \
  test_allocate_symmetric_remote_view_by_rank<Type *, Layout, RemoteSpace_t>(  \
      113);                                                                    \
  test_allocate_symmetric_remote_view_by_rank<Type **, Layout, RemoteSpace_t>( \
      7, 5);                                                                   \
  test_allocate_symmetric_remote_view_by_rank<Type ***, Layout,                \
                                              RemoteSpace_t>(9, 10, 7);        \
  test_allocate_symmetric_remote_view_by_rank<Type ****, Layout,               \
                                              RemoteSpace_t>(9, 10, 7, 2);     \
  test_allocate_symmetric_remote_view_by_rank<Type *****, Layout,              \
                                              RemoteSpace_t>(9, 10, 7, 2, 1);  \
  test_allocate_symmetric_remote_view_by_rank<Type ******, Layout,             \
                                              RemoteSpace_t>(9, 10, 7, 2, 1,   \
                                                             1);

TEST(TEST_CATEGORY, test_allocate_symmetric_remote_view_by_rank) {
  using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
  using PLL_t         = Kokkos::PartitionedLayoutLeft;
  using PLR_t         = Kokkos::PartitionedLayoutRight;
  using LL_t          = Kokkos::LayoutLeft;
  using LR_t          = Kokkos::LayoutRight;

  GEN_BLOCK_1(int, PLL_t, RemoteSpace_t)
  GEN_BLOCK_1(int, PLR_t, RemoteSpace_t)
  GEN_BLOCK_1(double, PLL_t, RemoteSpace_t)
  GEN_BLOCK_1(double, PLR_t, RemoteSpace_t)

  GEN_BLOCK_2(int, LL_t, RemoteSpace_t)
  GEN_BLOCK_2(int, LR_t, RemoteSpace_t)
  GEN_BLOCK_2(double, LL_t, RemoteSpace_t)
  GEN_BLOCK_2(double, LR_t, RemoteSpace_t)

  RemoteSpace_t::fence();
}
