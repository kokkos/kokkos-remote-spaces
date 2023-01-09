/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jan Ciesko (jciesko@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef TEST_SUBVIEW_HPP_
#define TEST_SUBVIEW_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t>
void test_scalar_reduce_1D(int dim0) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_1D_t =
      Kokkos::View<Data_t *, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using ViewRemote_1D_t = Kokkos::View<Data_t *, RemoteSpace_t>;
  using RangePolicy_t   = Kokkos::RangePolicy<>;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", dim0);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    v_h(i) = (Data_t)local_range.first + i;

  Kokkos::deep_copy(v, v_h);

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

  using ViewHost_2D_t =
      Kokkos::View<Data_t **, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, RemoteSpace_t>;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", dim0, dim1);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      v_h(i, j) = (Data_t)(local_range.first + i) * v_h.extent(1) + j;

  Kokkos::deep_copy(v, v_h);

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

  using ViewHost_3D_t = Kokkos::View<Data_t **, Kokkos::HostSpace>;
  using ViewRemote_3D_t =
      Kokkos::View<Data_t **, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t *, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;

  ViewRemote_3D_t v =
      ViewRemote_3D_t("RemoteView", num_ranks /*dim0*/, dim1 / num_ranks);
  ViewHost_3D_t v_h("HostView", 1 /*dim0*/, v.extent(1) /*dim1*/);

  auto v_sub =
      Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1), Kokkos::ALL);

  // Use a more sophisticated function to partition data if needed but may come
  // at the expense of operator cost. Here we rely on that KRS internally
  // allocates (dim1+num_ranks)/num_ranks symetrically.
  size_t dim1_block = dim1 / num_ranks;
  size_t block      = dim1_block;
  size_t start      = my_rank * block;

  // Init
  for (int i = 0; i < dim1_block; ++i) v_h(0, i) = (Data_t)start + i;

  Kokkos::deep_copy(v_sub, v_h);

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

  size_t total = dim1_block * num_ranks;
  ASSERT_EQ((total - 1) * (total) / 2, gsum);
}

template <class Data_t>
void test_scalar_reduce_partitioned_2D(int dim1, int dim2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t = Kokkos::View<Data_t ***, Kokkos::HostSpace>;
  using ViewRemote_3D_t =
      Kokkos::View<Data_t ***, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;

  ViewRemote_3D_t v =
      ViewRemote_3D_t("RemoteView", num_ranks /*dim0*/, dim1 / num_ranks, dim2);
  ViewHost_3D_t v_h("HostView", 1 /*dim0*/, v.extent(1) /*dim1*/,
                    v.extent(2) /*dim2*/);

  auto v_sub = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);

  size_t dim1_block = dim1 / num_ranks;
  size_t block      = dim1_block * dim2;
  size_t start      = my_rank * block;

  // Init
  for (int i = 0; i < dim1_block; ++i)
    for (int j = 0; j < v_h.extent(2); ++j)
      v_h(0, i, j) = (Data_t)start + i * dim2 + j;

  Kokkos::deep_copy(v_sub, v_h);

  Data_t gsum = 0;
  Kokkos::parallel_reduce(
      "Global reduce", dim1_block * num_ranks,
      KOKKOS_LAMBDA(const int i, Data_t &lsum) {
        size_t pe, index;
        pe    = i / dim1_block;
        index = i % dim1_block;
        for (int j = 0; j < dim2; ++j) lsum += v(pe, index, j);
      },
      gsum);

  size_t total = dim1_block * num_ranks * dim2;
  ASSERT_EQ((total - 1) * (total) / 2, gsum);
}

TEST(TEST_CATEGORY, test_reduce) {
  // Param 1: array size

  // Scalar reduce
  test_scalar_reduce_1D<int>(0);
  test_scalar_reduce_1D<int>(1);

  test_scalar_reduce_1D<float>(127);
  test_scalar_reduce_1D<double>(773);

  test_scalar_reduce_2D<int>(0, 0);
  test_scalar_reduce_2D<int>(1, 1);

  test_scalar_reduce_2D<float>(111, 3);
  test_scalar_reduce_2D<double>(773, 3);

  test_scalar_reduce_partitioned_1D<int>(20);
  test_scalar_reduce_partitioned_1D<double>(337);

  test_scalar_reduce_partitioned_2D<int>(4, 2);
  test_scalar_reduce_partitioned_2D<double>(773, 3);
}

#endif /* TEST_SUBVIEW_HPP_ */
