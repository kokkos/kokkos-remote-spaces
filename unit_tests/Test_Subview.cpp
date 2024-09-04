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

#define VAL 123

#define LIVE(EXPR, ARGS, DYNRANK) EXPECT_NO_THROW(EXPR)
#define DIE(EXPR) ASSERT_DEATH(EXPR, "Deep_copy of remote view not allowed.")

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t, class Layout>
void test_subview1D(int i1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_1D_t = Kokkos::View<Data_t *, Layout, RemoteSpace_t>;
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

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);

  for (int i = 0; i < local_range.second - local_range.first; ++i) {
    ASSERT_EQ(v_h(i), 2);
  }
}

template <class Data_t, class Layout>
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

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j) ASSERT_EQ(v_h(i, j), 2);
}

template <class Data_t, class Layout>
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

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 2);
}

template <class Data_t, class Layout>
void test_subview3D_byScalar(int i1, int i2, int i3) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, Kokkos::LayoutStride, RemoteSpace_t>;
  using ViewHost_3D_t = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", i1, i2, i3);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);
  // Set to next rank
  auto v_sub_1 =
      Kokkos::subview(v, remote_range.first, Kokkos::ALL, Kokkos::ALL);
  auto v_sub_2 =
      ViewRemote_2D_t(v, remote_range.first, Kokkos::ALL, Kokkos::ALL);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", v_sub_1.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1.extent(1); ++j) {
          v_sub_1(i, j)++;
        }
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  for (int i = 0; i < v_h.extent(0); ++i) {
    if (i == 0) {
      for (int j = 0; j < v_h.extent(1); ++j)
        for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 1);
    } else {
      for (int j = 0; j < v_h.extent(1); ++j)
        for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 0);
    }
  }

  Kokkos::parallel_for(
      "Increment", v_sub_2.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_2.extent(1); ++j) {
          v_sub_2(i, j)++;
        }
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  for (int i = 0; i < v_h.extent(0); ++i) {
    if (i == 0) {
      for (int j = 0; j < v_h.extent(1); ++j)
        for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 2);
    } else {
      for (int j = 0; j < v_h.extent(1); ++j)
        for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 0);
    }
  }
}

template <class Data_t, class Layout>
void test_subviewOfSubview_Scalar_3D(int i1, int i2, int i3) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, Kokkos::LayoutStride, RemoteSpace_t>;
  using ViewHost_3D_t = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", i1, i2, i3);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);

  // Set to next rank
  auto v_sub_1 =
      Kokkos::subview(v, remote_range.first, Kokkos::ALL, Kokkos::ALL);
  auto v_sub_2 =
      ViewRemote_2D_t(v, remote_range.first, Kokkos::ALL, Kokkos::ALL);

  int i2_half = static_cast<int>(i2 * 0.5);

  // Create subview on next rank using subview of a subview
  auto v_sub_1_half = Kokkos::subview(
      v_sub_1, Kokkos::pair<int, int>(i2_half, i2), Kokkos::ALL);
  auto v_sub_2_half = ViewRemote_2D_t(
      v_sub_2, Kokkos::pair<int, int>(i2_half, i2), Kokkos::ALL);

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", v_sub_1_half.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1_half.extent(1); ++j) {
          v_sub_1_half(i, j)++;
          v_sub_2_half(i, j)++;
        }
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  for (int j = 0; j < v_h.extent(1); ++j)
    if (j < i2_half)
      for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(0, j, k), 0);
    else
      for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(0, j, k), 2);
}

template <class Data_t, class Layout>
void test_subviewOfSubview_Range_3D(int i1, int i2, int i3) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int i2_half = static_cast<int>(i2 * 0.5);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewRemote_3D_LS_t =
      Kokkos::View<Data_t ***, Kokkos::LayoutStride, RemoteSpace_t>;
  using ViewHost_3D_t = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", i1, i2, i3);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  auto remote_range =
      Kokkos::Experimental::get_range(i1, (my_rank + 1) % num_ranks);

  // Set to next rank
  auto v_sub_1 = Kokkos::subview(v, remote_range, Kokkos::ALL, Kokkos::ALL);
  auto v_sub_2 = ViewRemote_3D_t(v, remote_range, Kokkos::ALL, Kokkos::ALL);

  // Create subview on next rank using subview of a subview
  auto v_sub_1_half = Kokkos::subview(
      v_sub_1, Kokkos::ALL, Kokkos::pair<int, int>(i2_half, i2), Kokkos::ALL);
  auto v_sub_2_half = ViewRemote_3D_LS_t(
      v_sub_2, Kokkos::ALL, Kokkos::pair<int, int>(i2_half, i2), Kokkos::ALL);

  size_t iters = remote_range.second - remote_range.first;

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int k = 0; k < v_h.extent(2); ++k) v_h(i, j, k) = 0;

  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", iters, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_1_half.extent(1); ++j)
          for (int k = 0; k < v_sub_1_half.extent(2); ++k) {
            v_sub_1_half(i, j, k)++;
            v_sub_2_half(i, j, k)++;
          }
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(i1);
  iters            = local_range.second - local_range.first;

  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      if (j < i2_half) {
        for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 0);
      } else {
        for (int k = 0; k < v_h.extent(2); ++k) ASSERT_EQ(v_h(i, j, k), 2);
      }
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
  // Offending deep_copy below
  Kokkos::deep_copy(v_h, v_sub_1);
}

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

  auto v_sub   = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_1 = Kokkos::subview(v, Kokkos::ALL, sub1, sub2);
  auto v_sub_2 = ViewRemote_1D_t(v, Kokkos::ALL, sub1, sub2);
  Kokkos::deep_copy(v_sub, v_h);
  RemoteSpace_t::fence();

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

  auto v_sub   = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_1 = Kokkos::subview(v, Kokkos::ALL, sub1, Kokkos::ALL);
  auto v_sub_2 = ViewRemote_2D_t(v, Kokkos::ALL, sub1, Kokkos::ALL);
  Kokkos::deep_copy(v_sub, v_h);
  RemoteSpace_t::fence();

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
  RemoteSpace_t::fence();

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
  deep_copy(v_h, my_rank);

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
  deep_copy(v_h, my_rank);

  auto v_sub      = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_next = Kokkos::subview(v, next_rank, Kokkos::ALL, Kokkos::ALL);
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_sub, v_h);

  Kokkos::parallel_for(
      "Increment", v_sub_next.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_next.extent(1); ++j) v_sub_next(i, j)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank + 1);
}

template <class Data_t, class Layout>
void test_partitioned_subviewOfSubviewRange_2D(int i1, int i2) {
  int my_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int i1_half = static_cast<int>(i1 * 0.5);

  next_rank = (my_rank + 1) % num_ranks;

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  deep_copy(v_h, my_rank);

  auto v_sub      = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_next = Kokkos::subview(v, next_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_sub_next_half = Kokkos::subview(
      v_sub_next, Kokkos::pair<int, int>(i1_half, i1), Kokkos::ALL);
  Kokkos::deep_copy(v_sub, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", v_sub_next_half.extent(0), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v_sub_next.extent(1); ++j) v_sub_next_half(i, j)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v_sub);

  for (int i = 0; i < i1_half; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank);

  for (int i = i1_half; i < i1; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank + 1);
}

template <class Data_t, class Layout>
void test_partitioned_subviewOfSubviewScalar_2D(int i1, int i2) {
  int my_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int i1_half = static_cast<int>(i1 * 0.5);

  next_rank = (my_rank + 1) % num_ranks;

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, Layout, RemoteSpace_t>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", num_ranks, i1, i2);
  ViewHost_3D_t v_h("HostView", 1, i1, i2);

  // Init
  deep_copy(v_h, my_rank);
  auto v_sub      = Kokkos::subview(v, std::make_pair(my_rank, my_rank + 1),
                               Kokkos::ALL, Kokkos::ALL);
  auto v_sub_next = Kokkos::subview(v, next_rank, Kokkos::ALL, Kokkos::ALL);

  // auto v_sub_next_1D = Kokkos::subview(v_sub_next, Kokkos::ALL, i1_half);
  auto v_sub_next_1D = Kokkos::subview(v_sub_next, i1_half, Kokkos::ALL);

  Kokkos::deep_copy(v_sub, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", v_sub_next_1D.extent(0),
      KOKKOS_LAMBDA(const int i) { v_sub_next_1D(i)++; });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v_sub);
  /*
      for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) printf("%i, %i, %i\n", i, j, v_h(0, i, j));

  */
  for (int i = 0; i < i1; ++i)
    if (i != i1_half)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank);
    else
      for (int j = 0; j < i2; ++j) ASSERT_EQ(v_h(0, i, j), my_rank + 1);
}

#define GENBLOCK1(TYPE, LAYOUT)           \
  test_subview1D<TYPE, LAYOUT>(555);      \
  test_subview2D<TYPE, LAYOUT>(123, 321); \
  test_subview3D<TYPE, LAYOUT>(13, 31, 23);

#define GENBLOCK2(TYPE, LAYOUT)                      \
  test_subview3D_byScalar<TYPE, LAYOUT>(5, 4, 4);    \
  test_subview3D_byScalar<TYPE, LAYOUT>(10, 11, 12); \
  test_subview3D_byScalar<TYPE, LAYOUT>(13, 31, 23);

#define GENBLOCK3(TYPE, LAYOUT) \
  DIE(test_subview3D_DCCopiesSubviewAccess<TYPE, LAYOUT>(13, 31, 23));

#define GENBLOCK4(TYPE, LAYOUT)                              \
  test_subviewOfSubview_Scalar_3D<TYPE, LAYOUT>(4, 4, 4);    \
  test_subviewOfSubview_Scalar_3D<TYPE, LAYOUT>(55, 11, 13); \
  test_subviewOfSubview_Scalar_3D<TYPE, LAYOUT>(13, 31, 23);

#define GENBLOCK5(TYPE, LAYOUT)                             \
  test_subviewOfSubview_Range_3D<TYPE, LAYOUT>(5, 4, 4);    \
  test_subviewOfSubview_Range_3D<TYPE, LAYOUT>(55, 11, 13); \
  test_subviewOfSubview_Range_3D<TYPE, LAYOUT>(13, 31, 23);

#define GENBLOCK6(TYPE, LAYOUT)                              \
  test_partitioned_subview1D<TYPE, LAYOUT>(4, 4, 0, 0);      \
  test_partitioned_subview1D<TYPE, LAYOUT>(50, 20, 8, 12);   \
  test_partitioned_subview1D<TYPE, LAYOUT>(255, 20, 49, 19); \
  test_partitioned_subview2D<TYPE, LAYOUT>(202, 20, 0);      \
  test_partitioned_subview2D<TYPE, LAYOUT>(50, 50, 4);       \
  test_partitioned_subview2D<TYPE, LAYOUT>(102, 20, 49);     \
  test_partitioned_subview3D<TYPE, LAYOUT>(50, 20, 0, 0);    \
  test_partitioned_subview3D<TYPE, LAYOUT>(30, 120, 3, 10);  \
  test_partitioned_subview3D<TYPE, LAYOUT>(70, 20, 0, 19);

#define GENBLOCK7(TYPE, LAYOUT)                                      \
  test_partitioned_subview2D_byRank_localRank<TYPE, LAYOUT>(8, 1);   \
  test_partitioned_subview2D_byRank_localRank<TYPE, LAYOUT>(55, 20); \
  test_partitioned_subview2D_byRank_localRank<TYPE, LAYOUT>(50, 77); \
  test_partitioned_subview2D_byRank_nextRank<TYPE, LAYOUT>(8, 10);   \
  test_partitioned_subview2D_byRank_nextRank<TYPE, LAYOUT>(55, 20);  \
  test_partitioned_subview2D_byRank_nextRank<TYPE, LAYOUT>(50, 77);

#define GENBLOCK8(TYPE, LAYOUT)                                    \
  test_partitioned_subviewOfSubviewRange_2D<TYPE, LAYOUT>(8, 1);   \
  test_partitioned_subviewOfSubviewRange_2D<TYPE, LAYOUT>(55, 20); \
  test_partitioned_subviewOfSubviewRange_2D<TYPE, LAYOUT>(50, 77);

#define GENBLOCK9(TYPE, LAYOUT)                                     \
  test_partitioned_subviewOfSubviewScalar_2D<TYPE, LAYOUT>(8, 5);   \
  test_partitioned_subviewOfSubviewScalar_2D<TYPE, LAYOUT>(55, 20); \
  test_partitioned_subviewOfSubviewScalar_2D<TYPE, LAYOUT>(50, 77);

TEST(TEST_CATEGORY, test_subview) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using LL_t  = Kokkos::LayoutLeft;
  using LR_t  = Kokkos::LayoutRight;
  using PLL_t = Kokkos::PartitionedLayoutLeft;
  using PLR_t = Kokkos::PartitionedLayoutRight;

  // Subview with GlobalLayout
  GENBLOCK1(int, LR_t);
  GENBLOCK1(float, LR_t);
  GENBLOCK1(double, LR_t);

  GENBLOCK1(int, LL_t);
  GENBLOCK1(float, LL_t);
  GENBLOCK1(double, LL_t);

  // Subview with GlobalLayout and split by dim0
  GENBLOCK2(int, LR_t);
  GENBLOCK2(float, LR_t);
  GENBLOCK2(double, LR_t);

  GENBLOCK2(int, LL_t);
  GENBLOCK2(float, LL_t);
  GENBLOCK2(double, LL_t);

  // 3D subview - Subview with GlobalLayout and
  // deep_copy accessing the subview directly
  /* TODO: find out why death test hangs in MPI_Finalize*/
  // GENBLOCK3(int);
  // GENBLOCK3(float);
  // GENBLOCK3(double);

  // 3D subview - Subview of subview with GlobalLayout
  // Unsupported use case
  GENBLOCK4(int, LR_t);
  GENBLOCK4(float, LR_t);
  GENBLOCK4(double, LR_t);

  GENBLOCK4(int, LL_t);
  GENBLOCK4(float, LL_t);
  GENBLOCK4(double, LL_t);

  // 3D subview - Subview of subview with GlobalLayout
  // Unsupported use case
  GENBLOCK5(int, LR_t);
  GENBLOCK5(float, LR_t);
  GENBLOCK5(double, LR_t);

  GENBLOCK5(int, LL_t);
  GENBLOCK5(float, LL_t);
  GENBLOCK5(double, LL_t);

  // Subiew with PartitionedLayout*
  GENBLOCK6(int, PLR_t);
  GENBLOCK6(float, PLR_t);
  GENBLOCK6(double, PLR_t);

  GENBLOCK6(int, PLL_t);
  GENBLOCK6(float, PLL_t);
  GENBLOCK6(double, PLL_t);

  // Subiew with PartitionedLayout* and split by rank
  GENBLOCK7(int, PLR_t);
  GENBLOCK7(float, PLR_t);
  GENBLOCK7(double, PLR_t);

  GENBLOCK7(int, PLL_t);
  GENBLOCK7(float, PLL_t);
  GENBLOCK7(double, PLL_t);

  // Subiew of subview with PartitionedLayout* and range
  GENBLOCK8(int, PLR_t);
  GENBLOCK8(float, PLR_t);
  GENBLOCK8(double, PLR_t);

  GENBLOCK8(int, PLL_t);
  GENBLOCK8(float, PLL_t);
  GENBLOCK8(double, PLL_t);

  // Subiew of subview with PartitionedLayout* and scalar
  GENBLOCK9(int, PLR_t);
  GENBLOCK9(float, PLR_t);
  GENBLOCK9(double, PLR_t);

  GENBLOCK9(int, PLL_t);
  GENBLOCK9(float, PLL_t);
  GENBLOCK9(double, PLL_t);

  RemoteSpace_t::fence();
}
