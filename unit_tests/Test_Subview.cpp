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

#define LIVE(EXPR, ARGS, DYNRANK) EXPECT_NO_THROW(EXPR)
#define DIE(EXPR) ASSERT_DEATH(EXPR, "Deep_copy of remote view not allowed.")

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t>
void test_subview1D(int i1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_1D_t = Kokkos::View<Data_t *, RemoteSpace_t>;
  using ViewHost_1D_t   = typename ViewRemote_1D_t::HostMirror;

  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", i1);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);

  // Set to next rank
  auto v_sub_1 = Kokkos::subview(v, remote_range);
  auto v_sub_2 = ViewRemote_1D_t(v, remote_range);

  size_t iters = remote_range.second - remote_range.first;

  // Init
  for (int i = 0; i < v_h.extent(0); ++i) v_h(i) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", iters, KOKKOS_LAMBDA(const int i) {
        v_sub_1(i)++;
        v_sub_2(i)++;
      });

  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);

  for (int i = 0; i < local_range.second - local_range.first; ++i) {
    ASSERT_EQ(v_h(i), 2);
  }
}

template <class Data_t>
void test_subview2D(int i1, int i2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_2D_t = Kokkos::View<Data_t **, RemoteSpace_t>;
  using ViewHost_2D_t   = typename ViewRemote_2D_t::HostMirror;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", i1, i2);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);

  // Set to next rank
  auto v_sub_1 = Kokkos::subview(v, remote_range, Kokkos::ALL);
  auto v_sub_2 = ViewRemote_2D_t(v, remote_range, Kokkos::ALL);

  size_t iters = remote_range.second - remote_range.first;

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j) v_h(i, j) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", iters, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(1); ++j) {
          v_sub_1(i, j)++;
          v_sub_2(i, j)++;
        }
      });

  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j) ASSERT_EQ(v_h(i, j), 2);
}

template <class Data_t>
void test_subview3D(int i1, int i2, int i3) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", i1, i2, i3);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);

  // Set to next rank
  auto v_sub_1 = Kokkos::subview(v, remote_range, Kokkos::ALL, Kokkos::ALL);
  auto v_sub_2 = ViewRemote_3D_t(v, remote_range, Kokkos::ALL, Kokkos::ALL);

  size_t iters = remote_range.second - remote_range.first;

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", iters, KOKKOS_LAMBDA(const int j) {
        for (int k = 0; k < v_sub_1.extent(1); ++k)
          for (int l = 0; l < v_sub_1.extent(2); ++l) {
            v_sub_1(j, k, l)++;
            v_sub_2(j, k, l)++;
          }
      });

  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 2);
}

template <class Data_t>
void test_subview3D_DCCopiesSubviewAccess(int i1, int i2, int i3) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", i1, i2, i3);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);

  // Set to next rank
  auto v_sub_1 = Kokkos::subview(v, remote_range, Kokkos::ALL, Kokkos::ALL);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  // Offening deep_copy below
  Kokkos::deep_copy(v_h, v_sub_1);
}

TEST(TEST_CATEGORY, test_subview) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  // 1D subview - Subview with GlobalLayout
  test_subview1D<int>(20);
  test_subview1D<float>(555);
  test_subview1D<double>(123);

  // 2D subview - Subview with GlobalLayout
  test_subview2D<int>(20, 20);
  test_subview2D<float>(555, 11);
  test_subview2D<double>(123, 321);

  // 3D subview - Subview with GlobalLayout
  test_subview3D<int>(20, 20, 20);
  test_subview3D<float>(55, 11, 13);
  test_subview3D<double>(13, 31, 23);

  // 3D subview - Subview with GlobalLayout and
  // deep_copy accessing the subview directly/

  /* TODO: find out why death test hangs in MPI_Finalize*/

  // DIE(test_subview3D_DCCopiesSubviewAccess<int>(20, 20, 20));
  // DIE(test_subview3D_DCCopiesSubviewAccess<float>(55, 11, 13));
  // DIE(test_subview3D_DCCopiesSubviewAccess<double>(13, 31, 23));

  RemoteSpace_t::fence();
}
