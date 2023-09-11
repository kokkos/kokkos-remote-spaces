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

#ifndef TEST_PARTITIONED_SUBVIEW_HPP_
#define TEST_PARTITIONED_SUBVIEW_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#define VAL 123

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t, class Layout>
void test_partitioned_subview1D(int i1, int i2, int sub1, int sub2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_1D_t = Kokkos::View<Data_t *, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  deep_copy(v_h, VAL);

  auto v_sub = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);

  auto v_sub_1 = Kokkos::subview(v, Kokkos::ALL, sub1, sub2);
  auto v_sub_2 = ViewRemote_1D_t(v, Kokkos::ALL, sub1, sub2);

  Kokkos::deep_copy(v_sub, v_h);

  Kokkos::parallel_for(
      "Increment", 1, KOKKOS_LAMBDA(const int i) {
        v_sub_1(my_rank)++;
        v_sub_2(my_rank)++;
      });

  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      if (i == sub1 && j == sub2) {
        ASSERT_EQ(v_h(0, i, j), VAL + 2);
      } else {
        ASSERT_EQ(v_h(0, i, j), VAL);
      }
}

template <class Data_t, class Layout>
void test_partitioned_subview2D(int i1, int i2, int sub1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  deep_copy(v_h, VAL);

  auto v_sub = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);

  auto v_sub_1 = Kokkos::subview(v, Kokkos::ALL, sub1, Kokkos::ALL);
  auto v_sub_2 = ViewRemote_2D_t(v, Kokkos::ALL, sub1, Kokkos::ALL);

  Kokkos::deep_copy(v_sub, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(1), KOKKOS_LAMBDA(const int i) {
        v_sub_1(my_rank, i)++;
        v_sub_2(my_rank, i)++;
      });

  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      if (i == sub1)
        ASSERT_EQ(v_h(0, i, j), VAL + 2);
      else
        ASSERT_EQ(v_h(0, i, j), VAL);
}

template <class Data_t, class Layout>
void test_partitioned_subview3D(int i1, int i2, int sub1, int sub2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote3D_t  = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  deep_copy(v_h, VAL);

  auto v_sub = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_1 =
      Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, std::make_pair(sub1, sub2));

  Kokkos::deep_copy(v_sub, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(1), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(2); ++j) {
          v_sub_1(my_rank, i, j)++;
        }
      });

  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      if ((sub1 <= j) && (j < sub2))
        ASSERT_EQ(v_h(0, i, j), VAL + 1);
      else
        ASSERT_EQ(v_h(0, i, j), VAL);
}

template <class Data_t, class Layout>
void test_partitioned_subview2D_byRank_localRank(int i1, int i2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) v_h(0, i, j) = my_rank;

  auto v_sub = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);

  auto v_sub_1 = Kokkos::subview(v, my_rank, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(v_sub, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(1); ++j) v_sub_1(i, j)++;
      });

  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank + 1);
}

template <class Data_t, class Layout>
void test_partitioned_subview2D_byRank_nextRank(int i1, int i2) {
  int my_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  next_rank = (my_rank + 1) % num_ranks;

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) v_h(0, i, j) = my_rank;

  auto v_sub      = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_next = Kokkos::subview(v, next_rank, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(v_sub, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_next.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_next.extent(1); ++j) v_sub_next(i, j)++;
      });

  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank + 1);
}

TEST(TEST_CATEGORY, test_partitioned_subview) {
  // 1D subview
  test_partitioned_subview1D<int, Kokkos::PartitionedLayoutRight>(4, 4, 0, 0);
  test_partitioned_subview1D<int, Kokkos::PartitionedLayoutRight>(50, 20, 8,
                                                                  12);
  test_partitioned_subview1D<int, Kokkos::PartitionedLayoutRight>(255, 20, 49,
                                                                  19);

  // 2D subview
  test_partitioned_subview2D<int, Kokkos::PartitionedLayoutRight>(202, 20, 0);
  test_partitioned_subview2D<int, Kokkos::PartitionedLayoutRight>(50, 50, 4);
  test_partitioned_subview2D<int, Kokkos::PartitionedLayoutRight>(102, 20, 49);

  // 3D subview
  test_partitioned_subview3D<int, Kokkos::PartitionedLayoutRight>(50, 20, 0, 0);
  test_partitioned_subview3D<int, Kokkos::PartitionedLayoutRight>(30, 120, 3,
                                                                  10);
  test_partitioned_subview3D<int, Kokkos::PartitionedLayoutRight>(70, 20, 0,
                                                                  19);

  // 2D subview split by dim0
  test_partitioned_subview2D_byRank_localRank<int,
                                              Kokkos::PartitionedLayoutRight>(
      8, 1);
  test_partitioned_subview2D_byRank_localRank<int,
                                              Kokkos::PartitionedLayoutRight>(
      55, 20);
  test_partitioned_subview2D_byRank_localRank<int,
                                              Kokkos::PartitionedLayoutRight>(
      50, 77);

  // 2D subview split by dim0
  test_partitioned_subview2D_byRank_nextRank<int,
                                             Kokkos::PartitionedLayoutRight>(
      8, 10);
  test_partitioned_subview2D_byRank_nextRank<int,
                                             Kokkos::PartitionedLayoutLeft>(55,
                                                                            20);
  test_partitioned_subview2D_byRank_nextRank<int,
                                             Kokkos::PartitionedLayoutLeft>(50,
                                                                            77);

  // 1D subview
  test_partitioned_subview1D<int, Kokkos::PartitionedLayoutLeft>(4, 4, 0, 0);
  test_partitioned_subview1D<int, Kokkos::PartitionedLayoutLeft>(50, 20, 8, 12);
  test_partitioned_subview1D<int, Kokkos::PartitionedLayoutLeft>(255, 20, 49,
                                                                 19);

  // 2D subview
  test_partitioned_subview2D<int, Kokkos::PartitionedLayoutLeft>(202, 20, 0);
  test_partitioned_subview2D<int, Kokkos::PartitionedLayoutLeft>(50, 50, 4);
  test_partitioned_subview2D<int, Kokkos::PartitionedLayoutLeft>(102, 20, 49);

  // 3D subview
  test_partitioned_subview3D<int, Kokkos::PartitionedLayoutLeft>(50, 20, 0, 0);
  test_partitioned_subview3D<int, Kokkos::PartitionedLayoutLeft>(30, 120, 3,
                                                                 10);
  test_partitioned_subview3D<int, Kokkos::PartitionedLayoutLeft>(70, 20, 0, 19);

  // 2D subview split by dim0
  test_partitioned_subview2D_byRank_localRank<int,
                                              Kokkos::PartitionedLayoutLeft>(8,
                                                                             1);
  test_partitioned_subview2D_byRank_localRank<int,
                                              Kokkos::PartitionedLayoutLeft>(
      55, 20);
  test_partitioned_subview2D_byRank_localRank<int,
                                              Kokkos::PartitionedLayoutLeft>(
      50, 77);

  // 2D subview split by dim0
  test_partitioned_subview2D_byRank_nextRank<int,
                                             Kokkos::PartitionedLayoutLeft>(8,
                                                                            10);
  test_partitioned_subview2D_byRank_nextRank<int,
                                             Kokkos::PartitionedLayoutLeft>(55,
                                                                            20);
  test_partitioned_subview2D_byRank_nextRank<int,
                                             Kokkos::PartitionedLayoutLeft>(50,
                                                                            77);
}

#endif /* TEST_PARTITIONED_SUBVIEW_HPP_ */
