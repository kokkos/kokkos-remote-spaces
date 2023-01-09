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

#ifndef TEST_LOCAL_DEEP_COPY_HPP_
#define TEST_LOCAL_DEEP_COPY_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t, class Space_A, class Space_B>
void test_localdeepcopy(
    typename std::enable_if<(std::is_same<Space_A, Kokkos::HostSpace>::value &&
                             std::is_same<Space_B, RemoteSpace_t>::value)>::type
        * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_t   = Kokkos::View<Data_t **, Space_A>;
  using ViewRemote_t = Kokkos::View<Data_t **, Space_B>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, 1);

  ViewRemote_t v_R     = ViewRemote_t("RemoteView", num_ranks, 1);
  ViewRemote_t v_R_cpy = ViewRemote_t("RemoteView", num_ranks, 1);

  Kokkos::parallel_for(
      "Team", TeamPolicy_t(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1),
                             [&](const int i) { v_R(my_rank, 0) = 0x123; });

        team.team_barrier();
        Kokkos::single(Kokkos::PerThread(team), [&]() {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(team, v_R_cpy,
                                                              v_R);
        });
      });

  Kokkos::deep_copy(v_H, v_R_cpy);
  ASSERT_EQ(0x123, v_H(0, 0));
}

template <class Data_t, class Space_A, class Space_B>
void test_localdeepcopy(
    int i1,
    typename std::enable_if<(std::is_same<Space_A, Kokkos::HostSpace>::value &&
                             std::is_same<Space_B, RemoteSpace_t>::value)>::type
        * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_t   = Kokkos::View<Data_t **, Space_A>;
  using ViewRemote_t = Kokkos::View<Data_t **, Space_B>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1);

  ViewRemote_t v_R     = ViewRemote_t("RemoteView", num_ranks, i1);
  ViewRemote_t v_R_cpy = ViewRemote_t("RemoteView", num_ranks, i1);

  Kokkos::parallel_for(
      "Team", TeamPolicy_t(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1),
                             [&](const int i) {
                               for (int j = 0; j < i1; ++j)
                                 v_R(my_rank, j) = 0x123;
                             });

        team.team_barrier();
        Kokkos::single(Kokkos::PerThread(team), [&]() {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(v_R_cpy, v_R);
        });
      });

  Kokkos::deep_copy(v_H, v_R_cpy);
  for (int j = 0; j < i1; ++j) ASSERT_EQ(0x123, v_H(0, j));
}

template <class Data_t, class Space_A, class Space_B>
void test_localdeepcopy(
    int i1, int i2,
    typename std::enable_if<(std::is_same<Space_A, Kokkos::HostSpace>::value &&
                             std::is_same<Space_B, RemoteSpace_t>::value)>::type
        * = nullptr) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_t   = Kokkos::View<Data_t ***, Space_A>;
  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);

  ViewRemote_t v_R     = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  ViewRemote_t v_R_cpy = ViewRemote_t("RemoteView", num_ranks, i1, i2);

  Kokkos::parallel_for(
      "Team", TeamPolicy_t(1, Kokkos::AUTO),
      KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1),
                             [&](const int i) {
                               for (int j = 0; j < i1; ++j)
                                 for (int k = 0; k < i2; ++k)
                                   v_R(my_rank, j, k) = 0x123;
                             });

        team.team_barrier();
        Kokkos::single(Kokkos::PerThread(team), [&]() {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(v_R_cpy, v_R);
        });
      });

  Kokkos::deep_copy(v_H, v_R_cpy);
  for (int i = 0; i < i1; ++i)
    for (int j = 0; j < i2; ++j) ASSERT_EQ(0x123, v_H(0, i, j));
}

TEST(TEST_CATEGORY, test_localdeepcopy) {
  // Scalar
  test_localdeepcopy<int, Kokkos::HostSpace, RemoteSpace_t>();
  test_localdeepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>();
  test_localdeepcopy<double, Kokkos::HostSpace, RemoteSpace_t>();

  // 1D
  test_localdeepcopy<int, Kokkos::HostSpace, RemoteSpace_t>(50);
  test_localdeepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>(150);
  test_localdeepcopy<double, Kokkos::HostSpace, RemoteSpace_t>(1500);

  // 2D
  test_localdeepcopy<int, Kokkos::HostSpace, RemoteSpace_t>(50, 20);
  test_localdeepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>(150, 99);
  test_localdeepcopy<double, Kokkos::HostSpace, RemoteSpace_t>(1500, 2199);
}

#endif /* TEST_LOCAL_DEEP_COPY_HPP_ */
