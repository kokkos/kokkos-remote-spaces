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
void test_atomic_globalview1D(int dim0) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_1D_t   = Kokkos::View<Data_t *, Kokkos::HostSpace>;
  using ViewRemote_1D_t = Kokkos::View<Data_t *, RemoteSpace_t,
                                       Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using TeamPolicy_t    = Kokkos::TeamPolicy<>;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", dim0);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  // Init
  Kokkos::deep_copy(v_h, 0);
  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", dim0, KOKKOS_LAMBDA(const int i) { v(i)++; });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);
  for (int i = 0; i < local_range.second - local_range.first; ++i) {
    ASSERT_EQ(v_h(i), num_ranks);
  }
}

template <class Data_t>
void test_atomic_globalview2D(int dim0, int dim1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, Kokkos::LayoutLeft, RemoteSpace_t,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using ViewHost_2D_t = typename ViewRemote_2D_t::HostMirror;
  ViewRemote_2D_t v   = ViewRemote_2D_t("RemoteView", dim0, dim1);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  // Init
  Kokkos::deep_copy(v_h, 0);
  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", dim0, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < v.extent(1); ++j) v(i, j)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);
  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j) {
      ASSERT_EQ(v_h(i, j), num_ranks);
    }
}

template <class Data_t>
void test_atomic_globalview3D(int dim0, int dim1, int dim2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewRemote_3D_t = Kokkos::View<Data_t ***, RemoteSpace_t,
                                       Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using ViewHost_3D_t   = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", dim0, dim1, dim2);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));

  // Init
  Kokkos::deep_copy(v_h, 0);
  Kokkos::deep_copy(v, v_h);
  RemoteSpace_t::fence();

  Kokkos::parallel_for(
      "Increment", dim0, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j)
          for (int k = 0; k < dim2; ++k) v(i, j, k)++;
      });

  Kokkos::fence();
  RemoteSpace_t::fence();
  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);
  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int l = 0; l < v_h.extent(2); ++l) {
        ASSERT_EQ(v_h(i, j, l), num_ranks);
      }
}

#define GENBLOCK1(TYPE)              \
  test_atomic_globalview1D<TYPE>(0); \
  test_atomic_globalview1D<TYPE>(1); \
  test_atomic_globalview1D<TYPE>(31);

#define GENBLOCK2(TYPE)                     \
  test_atomic_globalview2D<TYPE>(1, 1);     \
  test_atomic_globalview2D<TYPE>(128, 312); \
  test_atomic_globalview2D<TYPE>(256, 237);

#define GENBLOCK3(TYPE)                       \
  test_atomic_globalview3D<TYPE>(1, 1, 1);    \
  test_atomic_globalview3D<TYPE>(2, 17, 123); \
  test_atomic_globalview3D<TYPE>(3, 8, 123);

TEST(TEST_CATEGORY, test_atomic_globalview) {
  // 1D
  GENBLOCK1(int)
  GENBLOCK1(int64_t)
  // 2D
  GENBLOCK2(int)
  GENBLOCK2(int64_t)
  // 3D
  GENBLOCK3(int)
  GENBLOCK3(int64_t)

  RemoteSpace_t::fence();
}
