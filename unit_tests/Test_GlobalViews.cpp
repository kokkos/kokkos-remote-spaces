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

template <class T>
struct Is_Partitioned_Layout {
  enum : bool {
    value = std::is_base_of<Kokkos::PartitionedLayout,
                            typename T::array_layout>::value
  };
};

template <class Data_t, class Layout_t>
void test_globalview1D(
    int dim0,
    typename std::enable_if_t<!Is_Partitioned_Layout<Layout_t>::value> * =
        nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_1D_t = Kokkos::View<Data_t *, Layout_t, RemoteSpace_t>;
  using ViewHost_1D_t   = typename ViewRemote_1D_t::HostMirror;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", dim0);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  auto remote_range =
      Kokkos::Experimental::get_range(dim0, (my_rank + 1) % num_ranks);

  // Initialize
  for (int i = 0; i < v_h.extent(0); ++i) v_h(i) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  auto policy = Kokkos::RangePolicy<>(remote_range.first, remote_range.second);

  Kokkos::parallel_for(
      "Increment", policy, KOKKOS_LAMBDA(const int i) { v(i)++; });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    ASSERT_EQ(v_h(i), 1);
}

template <class Data_t, class Layout_t>
void test_globalview1D(
    int dim0,
    typename std::enable_if_t<Is_Partitioned_Layout<Layout_t>::value> * =
        nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_1D_t = Kokkos::View<Data_t *, Layout_t, RemoteSpace_t>;
  using ViewHost_1D_t   = typename ViewRemote_1D_t::HostMirror;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", dim0);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  auto remote_range =
      Kokkos::Experimental::get_range(dim0, (my_rank + 1) % num_ranks);

  // Initialize
  for (int i = 0; i < v_h.extent(0); ++i) v_h(i) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  auto policy = Kokkos::RangePolicy<>(remote_range.first, remote_range.second);

  Kokkos::parallel_for(
      "Increment", policy, KOKKOS_LAMBDA(const int i) { v(i)++; });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    ASSERT_EQ(v_h(i), 1);
}

template <class Data_t, class Layout_t>
void test_globalview2D(
    int dim0, int dim1,
    typename std::enable_if_t<!Is_Partitioned_Layout<Layout_t>::value> * =
        nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_2D_t = Kokkos::View<Data_t **, Layout_t, RemoteSpace_t>;
  using ViewHost_2D_t   = typename ViewRemote_2D_t::HostMirror;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", dim0, dim1);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  auto remote_range =
      Kokkos::Experimental::get_range(dim0, (my_rank + 1) % num_ranks);

  // Initialize
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j) v_h(i, j) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  auto policy = Kokkos::RangePolicy<>(remote_range.first, remote_range.second);

  Kokkos::parallel_for(
      "Increment", policy, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j) v(i, j)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j) ASSERT_EQ(v_h(i, j), 1);
}

template <class Data_t, class Layout_t>
void test_globalview2D(
    int dim0, int dim1,
    typename std::enable_if_t<Is_Partitioned_Layout<Layout_t>::value> * =
        nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_2D_t = Kokkos::View<Data_t **, Layout_t, RemoteSpace_t>;
  using ViewHost_2D_t   = typename ViewRemote_2D_t::HostMirror;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", dim0, dim1);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  auto remote_range =
      Kokkos::Experimental::get_range(dim0, (my_rank + 1) % num_ranks);

  // Initialize
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j) v_h(i, j) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  auto policy = Kokkos::RangePolicy<>(remote_range.first, remote_range.second);

  Kokkos::parallel_for(
      "Increment", policy, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j) v(i, j)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j) ASSERT_EQ(v_h(i, j), 1);
}

template <class Data_t, class Layout_t>
void test_globalview3D(
    int dim0, int dim1, int dim2,
    typename std::enable_if_t<!Is_Partitioned_Layout<Layout_t>::value> * =
        nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout_t, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", dim0, dim1, dim2);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(dim0, (my_rank + 1) % num_ranks);

  // Initialize
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  auto policy = Kokkos::RangePolicy<>(remote_range.first, remote_range.second);

  Kokkos::parallel_for(
      "Increment", policy, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j)
          for (int k = 0; k < dim2; ++k) v(i, j, k)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 1);
}

template <class Data_t, class Layout_t>
void test_globalview3D(
    int dim0, int dim1, int dim2,
    typename std::enable_if_t<Is_Partitioned_Layout<Layout_t>::value> * =
        nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout_t, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", dim0, dim1, dim2);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(dim0, (my_rank + 1) % num_ranks);

  // Initialize
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  auto policy = Kokkos::RangePolicy<>(remote_range.first, remote_range.second);

  Kokkos::parallel_for(
      "Increment", policy, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j)
          for (int k = 0; k < dim2; ++k) v(i, j, k)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 1);
}

#define GENBLOCK_CORNERCASES(TYPE, LAYOUT)  \
  /*Corner cases*/                          \
  test_globalview1D<TYPE, LAYOUT>(0);       \
  test_globalview1D<TYPE, LAYOUT>(1);       \
  test_globalview2D<TYPE, LAYOUT>(0, 0);    \
  test_globalview2D<TYPE, LAYOUT>(1, 1);    \
  test_globalview3D<TYPE, LAYOUT>(0, 0, 0); \
  test_globalview3D<TYPE, LAYOUT>(1, 1, 1);

#define GENBLOCK_CORNERCASES(TYPE, LAYOUT)  \
  /*Corner cases*/                          \
  test_globalview1D<TYPE, LAYOUT>(0);       \
  test_globalview1D<TYPE, LAYOUT>(1);       \
  test_globalview2D<TYPE, LAYOUT>(0, 0);    \
  test_globalview2D<TYPE, LAYOUT>(1, 1);    \
  test_globalview3D<TYPE, LAYOUT>(0, 0, 0); \
  test_globalview3D<TYPE, LAYOUT>(1, 1, 1);

#define GENBLOCK_OTHERCASES(TYPE, LAYOUT)     \
  test_globalview1D<TYPE, LAYOUT>(1235);      \
  test_globalview2D<TYPE, LAYOUT>(1129, 313); \
  test_globalview3D<TYPE, LAYOUT>(3, 33, 1025);

TEST(TEST_CATEGORY, test_globalview) {
  /*Corner cases*/
  GENBLOCK_CORNERCASES(int, Kokkos::LayoutLeft)
  GENBLOCK_CORNERCASES(float, Kokkos::LayoutLeft)
  GENBLOCK_CORNERCASES(double, Kokkos::LayoutLeft)

  GENBLOCK_CORNERCASES(int, Kokkos::LayoutRight)
  GENBLOCK_CORNERCASES(float, Kokkos::LayoutRight)
  GENBLOCK_CORNERCASES(double, Kokkos::LayoutRight)

  GENBLOCK_CORNERCASES(int, Kokkos::PartitionedLayoutLeft)
  GENBLOCK_CORNERCASES(float, Kokkos::PartitionedLayoutLeft)
  GENBLOCK_CORNERCASES(double, Kokkos::PartitionedLayoutLeft)

  GENBLOCK_CORNERCASES(int, Kokkos::PartitionedLayoutRight)
  GENBLOCK_CORNERCASES(float, Kokkos::PartitionedLayoutRight)
  GENBLOCK_CORNERCASES(double, Kokkos::PartitionedLayoutRight)

  /*Other cases*/
  GENBLOCK_OTHERCASES(int, Kokkos::LayoutLeft);
  GENBLOCK_OTHERCASES(float, Kokkos::LayoutLeft);
  GENBLOCK_OTHERCASES(double, Kokkos::LayoutLeft);

  GENBLOCK_OTHERCASES(int, Kokkos::LayoutRight);
  GENBLOCK_OTHERCASES(float, Kokkos::LayoutRight);
  GENBLOCK_OTHERCASES(double, Kokkos::LayoutRight);

  GENBLOCK_OTHERCASES(int, Kokkos::PartitionedLayoutLeft);
  GENBLOCK_OTHERCASES(float, Kokkos::PartitionedLayoutLeft);
  GENBLOCK_OTHERCASES(double, Kokkos::PartitionedLayoutLeft);

  GENBLOCK_OTHERCASES(int, Kokkos::PartitionedLayoutRight);
  GENBLOCK_OTHERCASES(float, Kokkos::PartitionedLayoutRight);
  GENBLOCK_OTHERCASES(double, Kokkos::PartitionedLayoutRight);

  RemoteSpace_t::fence();
}
