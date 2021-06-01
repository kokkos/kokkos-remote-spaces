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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
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
void test_subview1D(int i1, int i2, int sub1, int sub2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t = Kokkos::View<Data_t ***, Kokkos::HostSpace>;
  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewRemote_1D_t = Kokkos::View<Data_t *, RemoteSpace_t>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      v_h(0, i, j) = 0;

  auto v_sub_1 = Kokkos::subview(v, Kokkos::ALL, sub1, sub2);
  auto v_sub_2 = ViewRemote_1D_t(v, Kokkos::ALL, sub1, sub2);

  Kokkos::Experimental::deep_copy(v, v_h);
  Kokkos::parallel_for(
      "Increment", 1, KOKKOS_LAMBDA(const int i) {
        v_sub_1(my_rank)++;
        v_sub_2(my_rank)++;
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      if (i == sub1 && j == sub2)
        ASSERT_EQ(v_h(0, i, j), 2);
      else
        ASSERT_EQ(v_h(0, i, j), 0);
}

template <class Data_t> void test_subview2D(int i1, int i2, int sub1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t = Kokkos::View<Data_t ***, Kokkos::HostSpace>;
  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, RemoteSpace_t>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      v_h(0, i, j) = 0;

  auto v_sub_1 = Kokkos::subview(v, Kokkos::ALL, sub1, Kokkos::ALL);
  auto v_sub_2 = ViewRemote_2D_t(v, Kokkos::ALL, sub1, Kokkos::ALL);

  Kokkos::Experimental::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(1), KOKKOS_LAMBDA(const int i) {
        v_sub_1(my_rank, i)++;
        v_sub_2(my_rank, i)++;
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      if (i == sub1)
        ASSERT_EQ(v_h(0, i, j), 2);
      else
        ASSERT_EQ(v_h(0, i, j), 0);
}

template <class Data_t>
void test_subview3D(int i1, int i2, int sub1, int sub2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t = Kokkos::View<Data_t ***, Kokkos::HostSpace>;
  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewRemote3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      v_h(0, i, j) = 0;

  auto v_sub_1 =
      Kokkos::subview(v, Kokkos::ALL, Kokkos::ALL, std::make_pair(sub1, sub2));

  Kokkos::Experimental::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(1), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(2); ++j) {
          v_sub_1(my_rank, i, j)++;
        }
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      if ((sub1 <= j) && (j < sub2))
        ASSERT_EQ(v_h(0, i, j), 1);
      else
        ASSERT_EQ(v_h(0, i, j), 0);
}

template <class Data_t> void test_subview2D_byRank(int i1, int i2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t = Kokkos::View<Data_t ***, Kokkos::HostSpace>;
  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, RemoteSpace_t>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      v_h(0, i, j) = 0;

  auto v_sub_1 =
      Kokkos::subview(v, (my_rank + 1) % num_ranks, Kokkos::ALL, Kokkos::ALL);
  Kokkos::Experimental::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(1); ++j) {
          v_sub_1(i, j)++;
        }
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      ASSERT_EQ(v_h(0, i, j), 1);
}

template <class Data_t> void test_subview2D_byMulitpleRanks(int i1, int i2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t = Kokkos::View<Data_t ***, Kokkos::HostSpace>;
  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t>;
  using ViewRemote_2D_t = Kokkos::View<Data_t **, RemoteSpace_t>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  int next_rank = (my_rank + 1) % num_ranks;

  // Init
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      v_h(0, i, j) = 0;

  auto v_sub_1 = Kokkos::subview(v, std::make_pair(next_rank, next_rank),
                                 Kokkos::ALL, Kokkos::ALL);
  Kokkos::Experimental::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(1), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(2); ++j) {
          v_sub_1(0, i, j)++;
        }
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j)
      ASSERT_EQ(v_h(0, i, j), 1);
}

template <class Data_t> void test_subview2D_GlobalLayout(int i1, int i2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_2D_t =
      Kokkos::View<Data_t **, Kokkos::GlobalLayoutRight, Kokkos::HostSpace>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, Kokkos::GlobalLayoutRight, RemoteSpace_t>;

  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", i1, i2);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  int block = i1 / num_ranks;
  int next_rank = (my_rank + 1) % num_ranks;
  int start = next_rank * block;
  int end = (next_rank + 1) * block;
  end = (next_rank == num_ranks - 1) ? end + i1 % num_ranks : end;

  // Set to next rank
  auto v_sub_1 = Kokkos::subview(v, std::make_pair(start, end), Kokkos::ALL);
  auto v_sub_2 = ViewRemote_2D_t(v, std::make_pair(start, end), Kokkos::ALL);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      v_h(i, j) = 0;

  Kokkos::Experimental::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(1); ++j) {
          v_sub_1(i, j)++;
          v_sub_2(i, j)++;
        }
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  block = (my_rank == next_rank) ? block + (i1 % num_ranks) : block;

  for (int i = 0; i < block; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      ASSERT_EQ(v_h(i, j), 2);
}

template <class Data_t>
void test_subview3D_GlobalLayout(int i1, int i2, int i3) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_2D_t =
      Kokkos::View<Data_t ***, Kokkos::GlobalLayoutRight, Kokkos::HostSpace>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t ***, Kokkos::GlobalLayoutRight, RemoteSpace_t>;

  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", i1, i2, i3);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  int block = i1 / num_ranks;
  int next_rank = (my_rank + 1) % num_ranks;
  int start = next_rank * block;
  int end = (next_rank + 1) * block;
  end = (next_rank == num_ranks - 1) ? end + i1 % num_ranks : end;

  // Set to next rank
  auto v_sub_1 =
      Kokkos::subview(v, std::make_pair(start, end), Kokkos::ALL, Kokkos::ALL);
  auto v_sub_2 =
      ViewRemote_2D_t(v, std::make_pair(start, end), Kokkos::ALL, Kokkos::ALL);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k)
        v_h(i, j, k) = 0;

  Kokkos::Experimental::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(0), KOKKOS_LAMBDA(const int j) {
        for (int k = 0; k < v_sub_1.extent(1); ++k)
          for (int l = 0; l < v_sub_1.extent(2); ++l) {
            v_sub_1(j, k, l)++;
            v_sub_2(j, k, l)++;
          }
      });

  Kokkos::Experimental::deep_copy(v_h, v);

  block = (my_rank == next_rank) ? block + (i1 % num_ranks) : block;

  for (int i = 0; i < block; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k)
        ASSERT_EQ(v_h(i, j, k), 2);
}

TEST(TEST_CATEGORY, test_subview) {
  // 1D subview
  test_subview1D<int>(50, 20, 0, 0);
  test_subview1D<int>(50, 20, 8, 12);
  test_subview1D<int>(255, 20, 49, 19);

  // 2D subview
  test_subview2D<int>(202, 20, 0);
  test_subview2D<int>(50, 50, 4);
  test_subview2D<int>(1024, 20, 49);

  // 3D subview
  test_subview3D<int>(50, 20, 0, 0);
  test_subview3D<int>(30, 120, 3, 10);
  test_subview3D<int>(70, 20, 0, 19);

  // 1D subview split by dim0
  test_subview2D_byRank<int>(10, 10);
  test_subview2D_byRank<int>(55, 20);
  test_subview2D_byRank<int>(50, 77);

  test_subview2D_byMulitpleRanks<int>(10, 10);
  test_subview2D_byMulitpleRanks<int>(55, 221);
  test_subview2D_byMulitpleRanks<int>(108, 37);

  // 2D subview - Subview with GlobalLayout
  test_subview2D_GlobalLayout<int>(20, 20);
  test_subview2D_GlobalLayout<int>(555, 11);
  test_subview2D_GlobalLayout<int>(123, 321);

  // 3D subview - Subview with GlobalLayout
  test_subview3D_GlobalLayout<int>(20, 20, 20);
  test_subview3D_GlobalLayout<int>(55, 11, 13);
  test_subview3D_GlobalLayout<int>(13, 31, 23);
}

#endif /* TEST_SUBVIEW_HPP_ */