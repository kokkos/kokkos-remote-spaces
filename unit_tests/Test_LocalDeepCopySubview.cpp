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

enum flavor : int { with_team, without_team };
enum subview_gen : int { with_ranges, with_scalar };
enum block_ops : int { get_op, put_op };
enum team_sizes : int { big = 32, small = 12, very_small = 3 };

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == without_team && block_op_type == get_op &&
         subview_gen == with_ranges)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto next_range  = Kokkos::Experimental::get_range(num_ranks, next_rank);
  auto prev_range  = Kokkos::Experimental::get_range(num_ranks, prev_rank);
  auto local_range = Kokkos::Experimental::get_local_range(num_ranks);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, local_range, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R(my_rank, i, j) = my_rank;
      });

  RemoteSpace_t().fence();

  // Copy from next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_local, v_R_subview_next);
          });
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H(0, i, j));
  }
  RemoteSpace_t().fence();
  // Copy from previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_local, v_R_subview_prev);
          });
        });
  }

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(prev_rank, v_H(0, i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == with_team && block_op_type == get_op &&
         subview_gen == with_ranges)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto next_range  = Kokkos::Experimental::get_range(num_ranks, next_rank);
  auto prev_range  = Kokkos::Experimental::get_range(num_ranks, prev_rank);
  auto local_range = Kokkos::Experimental::get_local_range(num_ranks);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, local_range, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R(my_rank, i, j) = my_rank;
      });

  RemoteSpace_t().fence();
  // Copy from next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::big, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_local, v_R_subview_next);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H(0, i, j));
  }
  RemoteSpace_t().fence();
  // Copy from previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::small, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_local, v_R_subview_prev);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(prev_rank, v_H(0, i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == without_team && block_op_type == put_op &&
         subview_gen == with_ranges)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto next_range  = Kokkos::Experimental::get_range(num_ranks, next_rank);
  auto prev_range  = Kokkos::Experimental::get_range(num_ranks, prev_rank);
  auto local_range = Kokkos::Experimental::get_local_range(num_ranks);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, local_range, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R(my_rank, i, j) = my_rank;
      });

  RemoteSpace_t().fence();

  // Put to next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_next, v_R_subview_local);
          });
        });
  }

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) {
        ASSERT_EQ(prev_rank, v_H(0, i, j));
      }
  }
  RemoteSpace_t().fence();

  // Put to previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_prev, v_R_subview_local);
          });
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H(0, i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == with_team && block_op_type == put_op &&
         subview_gen == with_ranges)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto next_range  = Kokkos::Experimental::get_range(num_ranks, next_rank);
  auto prev_range  = Kokkos::Experimental::get_range(num_ranks, prev_rank);
  auto local_range = Kokkos::Experimental::get_local_range(num_ranks);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_range, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, local_range, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R(my_rank, i, j) = my_rank;
      });
  RemoteSpace_t().fence();
  // Put to next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::small, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_next, v_R_subview_local);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);
  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(prev_rank, v_H(0, i, j));
  }
  RemoteSpace_t().fence();
  // Put to previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::very_small, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_prev, v_R_subview_local);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);
  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H(0, i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == without_team && block_op_type == get_op &&
         subview_gen == with_scalar)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);
  auto v_H_sub = Kokkos::subview(v_H, 0, Kokkos::ALL, Kokkos::ALL);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, my_rank, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R_subview_local(i, j) = my_rank;
      });

  RemoteSpace_t().fence();

  // Copy from next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_local, v_R_subview_next);
          });
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H_sub(i, j));
  }
  RemoteSpace_t().fence();
  // Copy from previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_local, v_R_subview_prev);
          });
        });
  }

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(prev_rank, v_H_sub(i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == with_team && block_op_type == get_op &&
         subview_gen == with_scalar)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);
  auto v_H_sub = Kokkos::subview(v_H, 1, Kokkos::ALL, Kokkos::ALL);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, my_rank, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R_subview_local(i, j) = my_rank;
      });

  RemoteSpace_t().fence();
  // Copy from next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::big, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_local, v_R_subview_next);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H_sub(i, j));
  }
  RemoteSpace_t().fence();
  // Copy from previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::small, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_local, v_R_subview_prev);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);

  if (my_rank % 2 == 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(prev_rank, v_H_sub(i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == without_team && block_op_type == put_op &&
         subview_gen == with_scalar)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);
  auto v_H_sub = Kokkos::subview(v_H, 1, Kokkos::ALL, Kokkos::ALL);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, my_rank, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R_subview_local(i, j) = my_rank;
      });

  RemoteSpace_t().fence();

  // Put to next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_next, v_R_subview_local);
          });
        });
  }

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H, v_R);

  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) {
        ASSERT_EQ(prev_rank, v_H(0, i, j));
      }
  }
  RemoteSpace_t().fence();

  // Put to previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(1, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            Kokkos::Experimental::RemoteSpaces::local_deep_copy(
                v_R_subview_prev, v_R_subview_local);
          });
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);

  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H_sub(i, j));
  }
}

template <class Data_t, class Space_A, class Space_B, int is_enabled_team,
          int block_op_type, int subview_gen>
void test_localdeepcopy_withSubview(
    int i1, int i2,
    typename std::enable_if_t<
        (std::is_same<Space_A, Kokkos::HostSpace>::value &&
         std::is_same<Space_B, RemoteSpace_t>::value &&
         is_enabled_team == with_team && block_op_type == put_op &&
         subview_gen == with_scalar)> * = nullptr) {
  int my_rank;
  int prev_rank, next_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  prev_rank = (my_rank - 1) < 0 ? num_ranks - 1 : my_rank - 1;
  next_rank = (my_rank + 1) % num_ranks;

  if (num_ranks % 2 && num_ranks > 1) return;  // skip

  using ViewRemote_t = Kokkos::View<Data_t ***, Space_B>;
  using ViewHost_t   = typename ViewRemote_t::HostMirror;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewHost_t v_H("HostView", 1, i1, i2);
  auto v_H_sub = Kokkos::subview(v_H, 1, Kokkos::ALL, Kokkos::ALL);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1, i2);
  auto v_R_subview_next =
      Kokkos::subview(v_R, next_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_prev =
      Kokkos::subview(v_R, prev_rank, Kokkos::ALL, Kokkos::ALL);
  auto v_R_subview_local =
      Kokkos::subview(v_R, my_rank, Kokkos::ALL, Kokkos::ALL);

  Kokkos::parallel_for(
      "Init", i1, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < i2; ++j) v_R_subview_local(i, j) = my_rank;
      });
  RemoteSpace_t().fence();
  // Put to next
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::small, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_next, v_R_subview_local);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);
  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(prev_rank, v_H_sub(i, j));
  }
  RemoteSpace_t().fence();
  // Put to previous
  if (my_rank % 2 == 0) {
    Kokkos::parallel_for(
        "Team", TeamPolicy_t(team_sizes::very_small, 1),
        KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
          Kokkos::Experimental::RemoteSpaces::local_deep_copy(
              team, v_R_subview_prev, v_R_subview_local);
        });
  }
  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_H_sub, v_R_subview_local);
  if (my_rank % 2 != 0) {
    for (int i = 0; i < i1; ++i)
      for (int j = 0; j < i2; ++j) ASSERT_EQ(next_rank, v_H_sub(i, j));
  }
}

TEST(TEST_CATEGORY, test_localdeepcopywithsubview) {
  // With subviews using ranges
  // 2D with Subviews (get block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 without_team, get_op, with_ranges>(12, 15);
  // 2D with Teams and Subviews (get block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 with_team, get_op, with_ranges>(14, 19);
  // 2D with Subviews (put block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 without_team, put_op, with_ranges>(1, 16);
  // 2D with Teams and Subviews (put block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 with_team, put_op, with_ranges>(33, 2);

  // With subviews using scalar
  // 2D with Subviews (get block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 without_team, get_op, with_scalar>(12, 15);
  // 2D with Teams and Subviews (get block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 with_team, get_op, with_scalar>(14, 19);
  // 2D with Subviews (put block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 without_team, put_op, with_scalar>(1, 16);
  // 2D with Teams and Subviews (put block transfer)
  test_localdeepcopy_withSubview<int, Kokkos::HostSpace, RemoteSpace_t,
                                 with_team, put_op, with_scalar>(33, 2);

  MPI_Barrier(MPI_COMM_WORLD);
}
