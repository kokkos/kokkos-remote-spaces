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

#ifndef TEST_DEEP_COPY_HPP_
#define TEST_DEEP_COPY_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

/*
  Deep_copy can move data residing on the local node
  We need to take the dim0 offset into account to support deep_copying
  between memory spaces.
*/

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t, class Space_A, class Space_B>
void test_deepcopy(
    typename std::enable_if<(std::is_same<Space_A, Kokkos::HostSpace>::value &&
                             std::is_same<Space_B, RemoteSpace_t>::value)>::type
        * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_t = Kokkos::View<Data_t **, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;

  ViewHost_t v_H("HostView", 1, 1);
  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, 1);

  RemoteSpace_t().fence();

  Kokkos::parallel_for(
      "Team", 1, KOKKOS_LAMBDA(const int i) { v_R(my_rank, 0) = 0x123; });

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);
  ASSERT_EQ(0x123, v_H(0, 0));
}

template <class Data_t, class Space_A, class Space_B>
void test_deepcopy(
    int i1,
    typename std::enable_if<(std::is_same<Space_A, Kokkos::HostSpace>::value &&
                             std::is_same<Space_B, RemoteSpace_t>::value)>::type
        * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_t = Kokkos::View<Data_t **, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;

  ViewHost_t v_H("HostView", 1, i1);
  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1);

  Kokkos::parallel_for(
      "Team", i1, KOKKOS_LAMBDA(const int i) { v_R(my_rank, i) = 0x123; });

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);
  for (int i = 0; i < i1; ++i) {
    ASSERT_EQ(0x123, v_H(0, i));
  }
}

template <class Data_t, class Space_A, class Space_B>
void test_deepcopy(
    int i1, int i2,
    typename std::enable_if<(std::is_same<Space_A, Kokkos::HostSpace>::value &&
                             std::is_same<Space_B, RemoteSpace_t>::value)>::type
        * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;

  ViewHost_t v_H("HostView", 1, i1, i2);
  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);

  Kokkos::parallel_for(
      "Team", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R(my_rank, i, j) = 0x123;
      });

  Kokkos::deep_copy(v_H, v_R);
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(0x123, v_H(0, i, j));
}

template <class Data_t, class Space_A, class Space_B>
void test_deepcopy(
    typename std::enable_if<
        (std::is_same<Space_A, RemoteSpace_t>::value &&
         std::is_same<Space_B, Kokkos::HostSpace>::value)>::type * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_t = Kokkos::View<Data_t **, Space_A>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;

  ViewHost_t v_H("HostView", 1, 1);
  v_H(0, 0)        = 0x123;
  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, 1);

  Kokkos::deep_copy(v_R, v_H);

  Kokkos::parallel_for(
      "Team", 1,
      KOKKOS_LAMBDA(const int i) { assert(v_R(my_rank, 0) == (Data_t)0x123); });

  Kokkos::fence();
}

template <class Data_t, class Space_A, class Space_B>
void test_deepcopy(
    int i1,
    typename std::enable_if<
        (std::is_same<Space_A, RemoteSpace_t>::value &&
         std::is_same<Space_B, Kokkos::HostSpace>::value)>::type * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_t = Kokkos::View<Data_t **, Space_A>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;

  ViewHost_t v_H("HostView", 1, i1);
  for (int i = 0; i < i1; ++i) v_H(0, i) = 0x123;

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1);
  Kokkos::deep_copy(v_R, v_H);

  Kokkos::parallel_for(
      "Team", i1,
      KOKKOS_LAMBDA(const int i) { assert(v_R(my_rank, i) == (Data_t)0x123); });

  Kokkos::fence();
}

template <class Data_t, class Space_A, class Space_B>
void test_deepcopy(
    int i1, int i2,
    typename std::enable_if<
        (std::is_same<Space_A, RemoteSpace_t>::value &&
         std::is_same<Space_B, Kokkos::HostSpace>::value)>::type * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_A>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;

  ViewHost_t v_H("HostView", 1, i1, i2);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) v_H(0, i, j) = 0x123;

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);

  Kokkos::deep_copy(v_R, v_H);

  Kokkos::parallel_for(
      "Team", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j)
          assert(v_R(my_rank, i, j) == (Data_t)0x123);
      });
  Kokkos::fence();
}

TEST(TEST_CATEGORY, test_deepcopy) {
  // scalar
  test_deepcopy<int, RemoteSpace_t, Kokkos::HostSpace>();
  test_deepcopy<int, Kokkos::HostSpace, RemoteSpace_t>();
  test_deepcopy<int64_t, RemoteSpace_t, Kokkos::HostSpace>();
  test_deepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>();
  test_deepcopy<double, RemoteSpace_t, Kokkos::HostSpace>();
  test_deepcopy<double, Kokkos::HostSpace, RemoteSpace_t>();

  // 1D
  test_deepcopy<int, Kokkos::HostSpace, RemoteSpace_t>(10);
  test_deepcopy<int, RemoteSpace_t, Kokkos::HostSpace>(100);
  test_deepcopy<int64_t, RemoteSpace_t, Kokkos::HostSpace>(200);
  test_deepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>(200);
  test_deepcopy<double, RemoteSpace_t, Kokkos::HostSpace>(300);
  test_deepcopy<double, Kokkos::HostSpace, RemoteSpace_t>(300);

  // 2D
  test_deepcopy<int, RemoteSpace_t, Kokkos::HostSpace>(100, 200);
  test_deepcopy<int, Kokkos::HostSpace, RemoteSpace_t>(100, 200);
  test_deepcopy<int64_t, RemoteSpace_t, Kokkos::HostSpace>(200, 100);
  test_deepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>(200, 100);
  test_deepcopy<double, RemoteSpace_t, Kokkos::HostSpace>(100, 300);
  test_deepcopy<double, Kokkos::HostSpace, RemoteSpace_t>(100, 300);
}

#endif /* TEST_DEEP_COPY_HPP_ */