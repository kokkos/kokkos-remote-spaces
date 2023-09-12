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

template <class Data_t, class Space_t>
void test_remote_accesses(int size) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  TeamPolicy policy = TeamPolicy(1, Kokkos::AUTO);

  using RemoteView_t = Kokkos::View<Data_t **, Space_t>;
  using HostSpace_t  = typename RemoteView_t::HostMirror;
  RemoteView_t v_R   = RemoteView_t("RemoteView", num_ranks, size);
  HostSpace_t v_H("HostView", v_R.extent(0), size);

  int next_rank = (my_rank + 1) % num_ranks;
  int prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;

  Kokkos::parallel_for(
      "Update", size, KOKKOS_LAMBDA(const int i) {
        v_R(next_rank, i) = (Data_t)my_rank * size + i;
      });

  Kokkos::deep_copy(v_H, v_R);

  Data_t check(0), ref(0);
  for (int i = 0; i < size; i++) {
    check += v_H(0, i);
    ref += prev_rank * size + i;
  }
  ASSERT_EQ(check, ref);
}

TEST(TEST_CATEGORY, test_remote_accesses) {
  test_remote_accesses<int, RemoteSpace_t>(0);
  test_remote_accesses<int, RemoteSpace_t>(1);
  test_remote_accesses<float, RemoteSpace_t>(64);
  test_remote_accesses<int64_t, RemoteSpace_t>(4567);
  test_remote_accesses<double, RemoteSpace_t>(89);
}
