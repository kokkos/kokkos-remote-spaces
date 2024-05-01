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

template <class Data_t>
void test_scalar_reduce_1D(int dim0) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_1D_t = Kokkos::View<Data_t *, RemoteSpace_t>;
  using ViewHost_1D_t   = typename ViewRemote_1D_t::HostMirror;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", dim0);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    v_h(i) = static_cast<Data_t>(local_range.first + i);

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Data_t gsum = 0;

  Kokkos::parallel_reduce(
      "Global reduce", dim0,
      KOKKOS_LAMBDA(const int i, Data_t &lsum) { lsum += v(i); }, gsum);

  ASSERT_EQ((dim0 - 1) * (dim0) / 2, gsum);
}

template <class Data_t>
void test_scalar_reduce_2D(int dim0, int dim1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_2D_t = Kokkos::View<Data_t **, RemoteSpace_t>;
  using ViewHost_2D_t   = typename ViewRemote_2D_t::HostMirror;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", dim0, dim1);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      v_h(i, j) =
          static_cast<Data_t>(local_range.first + i) * v_h.extent(1) + j;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Data_t gsum = 0;

  Kokkos::parallel_reduce(
      "Global reduce", dim0,
      KOKKOS_LAMBDA(const int i, Data_t &lsum) {
        for (int j = 0; j < dim1; ++j) lsum += v(i, j);
      },
      gsum);

  size_t total = dim0 * dim1;
  ASSERT_EQ((total - 1) * (total) / 2, gsum);
}

template <class Data_t>
void test_scalar_reduce_partitioned_1D(int dim1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;

  size_t dim1_block = dim1 / num_ranks;
  size_t block      = dim1_block;
  size_t start      = my_rank * block;

  ViewRemote_2D_t v =
      ViewRemote_2D_t("RemoteView", num_ranks /*dim0*/, dim1_block);

  // Init
  Kokkos::parallel_for(
      "Local init", block, KOKKOS_LAMBDA(const int i) {
        v(my_rank, i) = static_cast<Data_t>(start + i);
      });

  RemoteSpace_t::fence();

  Data_t gsum = 0;
  Kokkos::parallel_reduce(
      "Global reduce", dim1_block * num_ranks,
      KOKKOS_LAMBDA(const int i, Data_t &lsum) {
        size_t pe, index;
        pe    = i / dim1_block;
        index = i % dim1_block;
        lsum += v(pe, index);
      },
      gsum);

  size_t total = block * num_ranks;
  ASSERT_EQ((total - 1) * (total) / 2, gsum);
}

template <class Data_t>
void test_scalar_reduce_partitioned_2D(int dim1, int dim2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t =
      Kokkos::View<Data_t ***, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;
  using ViewHost_3D_t = typename ViewRemote_3D_t::HostMirror;

  size_t dim1_block = dim1 / num_ranks;
  size_t block      = dim1_block * dim2;
  size_t start      = my_rank * block;

  ViewRemote_3D_t v =
      ViewRemote_3D_t("RemoteView", num_ranks /*dim0*/, dim1_block, dim2);

  ViewHost_3D_t v_h = ViewHost_3D_t("HostView", 1 /*dim0*/, dim1_block, dim2);

  auto v_sub = Kokkos::subview(v, Kokkos::pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  for (int i = 0; i < dim1_block; ++i)
    for (int j = 0; j < v_h.extent(2); ++j)
      v_h(0, i, j) = (Data_t)start + i * dim2 + j;

  Kokkos::deep_copy(v_sub, v_h);
  RemoteSpace_t::fence();

  Data_t gsum = 0;
  Kokkos::parallel_reduce(
      "Global reduce", dim1_block * num_ranks,
      KOKKOS_LAMBDA(const int i, Data_t &lsum) {
        size_t pe, index;
        pe    = i / dim1_block;
        index = i % dim1_block;

        for (int j = 0; j < v.extent(2); ++j) {
          int tmp = v(pe, index, j);
          lsum += v(pe, index, j);
        }
      },
      gsum);

  size_t total = block * num_ranks;
  ASSERT_EQ((total - 1) * (total) / 2, gsum);
}

#define GENBLOCK_1(TYPE)               \
  test_scalar_reduce_1D<TYPE>(0);      \
  test_scalar_reduce_1D<TYPE>(1);      \
  test_scalar_reduce_1D<TYPE>(127);    \
  test_scalar_reduce_2D<TYPE>(0, 0);   \
  test_scalar_reduce_2D<TYPE>(1, 1);   \
  test_scalar_reduce_2D<TYPE>(111, 3); \
  test_scalar_reduce_2D<TYPE>(773, 3);

#define GENBLOCK_2(TYPE)                         \
  test_scalar_reduce_partitioned_1D<TYPE>(20);   \
  test_scalar_reduce_partitioned_1D<TYPE>(337);  \
  test_scalar_reduce_partitioned_2D<TYPE>(4, 2); \
  test_scalar_reduce_partitioned_2D<TYPE>(773, 3);

TEST(TEST_CATEGORY, test_reduce) {
  GENBLOCK_1(int)
  GENBLOCK_1(float)
  GENBLOCK_1(double)

  GENBLOCK_2(int)
  GENBLOCK_2(float)
  GENBLOCK_2(double)

  RemoteSpace_t::fence();
}
