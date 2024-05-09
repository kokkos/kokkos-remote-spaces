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

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>
#include <typeinfo>
#include <type_traits>
#include <string>

#define NUM_TEAMS 1 /* Recursive subview support needed. TBD */
#define ACCESS_LDC_USE_MULTI_LDC
//#define ACCESS_LDC_USE_MULTI_LDC_BUILTIN

//#define CHECK_FOR_CORRECTNESS

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double *, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double *, Kokkos::LayoutLeft>;
using UnmanagedView_t =
    Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView_t = typename RemoteView_t::HostMirror;

// Tags
struct InitTag {};
struct UpdateTag {};
struct UpdateTag_put {};
struct UpdateTag_get {};
struct CheckTag {};

// Exec policies
using policy_init_t            = Kokkos::RangePolicy<InitTag, size_t>;
using policy_update_t          = Kokkos::RangePolicy<UpdateTag, size_t>;
using policy_update_put_t      = Kokkos::RangePolicy<UpdateTag_put, size_t>;
using policy_update_get_t      = Kokkos::RangePolicy<UpdateTag_get, size_t>;
using team_policy_get_update_t = Kokkos::TeamPolicy<UpdateTag_get, size_t>;
using team_policy_put_update_t = Kokkos::TeamPolicy<UpdateTag_put, size_t>;
using policy_check_t           = Kokkos::RangePolicy<CheckTag, size_t>;

// Default values
#define default_Mode 0
#define default_N 134217728
#define default_Iters 3
#define default_RmaOp 0  // get
#define TAG 0

std::string modes[4] = {"Kokkos::View", "Kokkos::View-MPIIsCudaAware",
                        "Kokkos::RemoteView",
                        "Kokkos::RemoteViewBlockTransfer"};

enum { RMA_GET, RMA_PUT };

struct Args_t {
  int mode   = default_Mode;
  int N      = default_N;
  int iters  = default_Iters;
  int rma_op = default_RmaOp;
};

void print_help() {
  printf("Options (default):\n");
  printf("  -N IARG: (%i) num elements in the vector\n", default_N);
  printf("  -I IARG: (%i) num repititions\n", default_Iters);
  printf("  -M IARG: (%i) mode (view type)\n", default_Mode);
  printf("  -O IARG: (%i) rma operation (0...get, 1...put)\n", default_RmaOp);
  printf("     modes:\n");
  printf("       0: Kokkos (Normal)  View\n");
  printf("       1: Kokkos Remote    View\n");
}

// read command line args
bool read_args(int argc, char *argv[], Args_t &args) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      print_help();
      return false;
    }
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-N") == 0) args.N = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-I") == 0) args.iters = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-M") == 0) args.mode = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-O") == 0) args.rma_op = atoi(argv[i + 1]);
  }
  return true;
}

template <typename ViewType_t, typename Enable = void>
struct Access;

template <typename ViewType_t>
struct Access<ViewType_t, typename std::enable_if_t<
                              !std::is_same<ViewType_t, RemoteView_t>::value>> {
  size_t N;  /* size of vector */
  int iters; /* number of iterations */
  int mode;  /* View type */

  int my_rank, other_rank, num_ranks = 0;

  ViewType_t v;
  ViewType_t v_tmp;

  Access(Args_t args)
      : N(args.N),
        iters(args.iters),
        v(std::string(typeid(v).name()), args.N),
        v_tmp(std::string(typeid(v).name()) + "_tmp", args.N),
        mode(args.mode) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    other_rank = my_rank ^ 1;
    assert(num_ranks == 2);
  };

  KOKKOS_FUNCTION
  void operator()(const InitTag &, const size_t i) const { v(i) = my_rank; }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag &, const size_t i) const { v(i) += v_tmp(i); }

  KOKKOS_FUNCTION
  void operator()(const CheckTag &, const size_t i) const {
    assert(v(i) == typename ViewType_t::traits::value_type(iters * other_rank +
                                                           my_rank));
  }

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b = 0;
    double time     = 0;

    Kokkos::parallel_for("access_overhead-init", policy_init_t({0}, {N}),
                         *this);
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
      time_a = timer.seconds();

      if (my_rank == 1) {
        auto v_tmp_host = Kokkos::create_mirror_view(v_tmp);
        Kokkos::deep_copy(v_tmp_host, v);
        MPI_Send(v_tmp_host.data(), N, MPI_DOUBLE, other_rank, TAG,
                 MPI_COMM_WORLD);

      } else {
        auto v_tmp_host = Kokkos::create_mirror_view(v_tmp);
        MPI_Recv(v_tmp_host.data(), N, MPI_DOUBLE, other_rank, TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        Kokkos::deep_copy(v_tmp, v_tmp_host);
        Kokkos::parallel_for("access_overhead", policy_update_t({0}, {N}),
                             *this);

        Kokkos::fence();

        time_b = timer.seconds();
        time += time_b - time_a;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
#ifdef CHECK_FOR_CORRECTNESS
      Kokkos::parallel_for("access_overhead-check", policy_check_t({0}, {N}),
                           *this);
      Kokkos::fence();
#endif

      double gups = 1e-9 * ((N * iters) / time);
      double size = N * sizeof(double) / 1024.0 / 1024.0;
      double bw   = gups * sizeof(double);
      printf("access_overhead_p2p,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
             modes[mode].c_str(), N, size, iters, time, gups, bw);
    }
  }
};

template <typename ViewType_t, typename Enable = void>
struct Access_CudaAware;

template <typename ViewType_t>
struct Access_CudaAware<
    ViewType_t,
    typename std::enable_if_t<!std::is_same<ViewType_t, RemoteView_t>::value>> {
  size_t N;  /* size of vector */
  int iters; /* number of iterations */
  int mode;  /* View type */

  int my_rank, other_rank, num_ranks = 0;

  ViewType_t v;
  ViewType_t v_tmp;

  Access_CudaAware(Args_t args)
      : N(args.N),
        iters(args.iters),
        v(std::string(typeid(v).name()), args.N),
        v_tmp(std::string(typeid(v).name()) + "_tmp", args.N),
        mode(args.mode) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    other_rank = my_rank ^ 1;
    assert(num_ranks == 2);
  };

  KOKKOS_FUNCTION
  void operator()(const InitTag &, const size_t i) const { v(i) = my_rank; }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag &, const size_t i) const { v(i) += v_tmp(i); }

  KOKKOS_FUNCTION
  void operator()(const CheckTag &, const size_t i) const {
    assert(v(i) == typename ViewType_t::traits::value_type(iters * other_rank +
                                                           my_rank));
  }

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b = 0;
    double time     = 0;

    Kokkos::parallel_for("access_overhead-init", policy_init_t({0}, {N}),
                         *this);
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
      time_a = timer.seconds();

      if (my_rank == 1) {
        MPI_Send(v.data(), N, MPI_DOUBLE, other_rank, TAG, MPI_COMM_WORLD);

      } else {
        MPI_Recv(v_tmp.data(), N, MPI_DOUBLE, other_rank, TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        Kokkos::parallel_for("access_overhead", policy_update_t({0}, {N}),
                             *this);

        Kokkos::fence();

        time_b = timer.seconds();
        time += time_b - time_a;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
#ifdef CHECK_FOR_CORRECTNESS
      Kokkos::parallel_for("access_overhead-check", policy_check_t({0}, {N}),
                           *this);
      Kokkos::fence();
#endif

      double gups = 1e-9 * ((N * iters) / time);
      double size = N * sizeof(double) / 1024.0 / 1024.0;
      double bw   = gups * sizeof(double);
      printf("access_overhead_p2p,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
             (modes[mode]).c_str(), N, size, iters, time, gups, bw);
    }
  }
};

template <typename ViewType_t>
struct Access<ViewType_t, typename std::enable_if_t<
                              std::is_same<ViewType_t, RemoteView_t>::value>> {
  size_t N;  /* size of vector */
  int iters; /* number of iterations */
  int mode;  /* View type */
  int rma_op;

  int my_rank, other_rank, num_ranks;

  ViewType_t v;

  Access(Args_t args)
      : N(args.N), iters(args.iters), mode(args.mode), rma_op(args.rma_op) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    assert(num_ranks == 2);
    other_rank = my_rank ^ 1;
    v          = ViewType_t(std::string(typeid(v).name()), num_ranks * args.N);
  };

  KOKKOS_FUNCTION
  void operator()(const InitTag &, const size_t i) const { v(i) = my_rank; }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag_get &, const size_t i) const {
    v(i) += v(other_rank * N + i);
  }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag_put &, const size_t i) const {
    v(other_rank * N + i) = v(i);
  }

  KOKKOS_FUNCTION
  void operator()(const CheckTag &, const size_t i) const {
    assert(v(i) == typename ViewType_t::traits::value_type(iters * other_rank +
                                                           my_rank));
  }

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b = 0;
    double time     = 0;

    auto local_range =
        Kokkos::Experimental::get_local_range(num_ranks * v.size());
    Kokkos::parallel_for("access_overhead-init",
                         policy_init_t(local_range.first, local_range.second),
                         *this);
    RemoteSpace_t().fence();

    if (rma_op == RMA_GET) {
      for (int i = 0; i < iters; i++) {
        if (my_rank == 0) {
          time_a = timer.seconds();
          Kokkos::parallel_for(
              "access_overhead",
              policy_update_get_t(local_range.first, local_range.second),
              *this);
          RemoteSpace_t().fence();
          time_b = timer.seconds();
          time += time_b - time_a;
        } else {
          RemoteSpace_t().fence();
        }
      }
    } else if (rma_op == RMA_PUT) {
      for (int i = 0; i < iters; i++) {
        if (my_rank == 0) {
          time_a = timer.seconds();
          Kokkos::parallel_for(
              "access_overhead",
              policy_update_put_t(local_range.first, local_range.second),
              *this);
          RemoteSpace_t().fence();
          time_b = timer.seconds();
          time += time_b - time_a;
        } else {
          RemoteSpace_t().fence();
        }
      }
    } else {
      printf("What rma_op is this? Exiting.\n");
      exit(1);
    }

    if (rma_op == RMA_GET) {
      // check on rank 0
      if (my_rank == 0) {
#ifdef CHECK_FOR_CORRECTNESS
        Kokkos::parallel_for(
            "access_overhead-check",
            policy_check_t(local_range.first, local_range.second), *this);
        Kokkos::fence();
#endif
      }
    } else {
      // check on rank 1
      if (my_rank == 1) {
#ifdef CHECK_FOR_CORRECTNESS
        Kokkos::parallel_for(
            "access_overhead-check",
            policy_check_t(local_range.first, local_range.second), *this);
        Kokkos::fence();
#endif
      }
    }

    if (my_rank == 0) {
      double gups = 1e-9 * ((N * iters) / time);
      double size = N * sizeof(double) / 1024.0 / 1024.0;
      double bw   = gups * sizeof(double);
      if (rma_op == RMA_GET) {
        printf("access_overhead_p2p,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
               modes[mode].c_str(), N, size, iters, time, gups, bw);
      } else {
        printf("access_overhead_p2p_put,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
               modes[mode].c_str(), N, size, iters, time, gups, bw);
      }
    }
  }
};

template <typename ViewType_t, typename Enable = void>
struct Access_LDC;

template <typename ViewType_t>
struct Access_LDC<
    ViewType_t,
    typename std::enable_if_t<std::is_same<ViewType_t, RemoteView_t>::value>> {
  size_t N;  /* size of vector */
  int iters; /* number of iterations */
  int mode;  /* View type */
  int rma_op;

  int my_rank, other_rank, num_ranks;

  ViewType_t v, v_tmp;
  ViewType_t v_subview_remote;

  Access_LDC(Args_t args)
      : N(args.N), iters(args.iters), mode(args.mode), rma_op(args.rma_op) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    assert(num_ranks == 2);
    other_rank = my_rank ^ 1;
    v          = ViewType_t(std::string(typeid(v).name()), num_ranks * args.N);
    v_tmp      = ViewType_t(std::string(typeid(v).name()), num_ranks * args.N);
  };

  KOKKOS_FUNCTION
  void operator()(const size_t i) const {
    double val1 = v_tmp(i);
    double val2 = v(i);
    printf("debug: %li, %.2f, %.2f\n", i, val1, val2);
  }

  KOKKOS_FUNCTION
  void operator()(const InitTag &, const size_t i) const { v(i) = my_rank; }

  KOKKOS_FUNCTION
  void operator()(const CheckTag &, const size_t i) const {
    assert(v(i) == typename ViewType_t::traits::value_type(iters * other_rank +
                                                           my_rank));
  }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag &, const size_t i) const { v(i) += v_tmp(i); }

#ifdef ACCESS_LDC_USE_MULTI_LDC
  KOKKOS_FUNCTION
  void operator()(const UpdateTag_get &,
                  typename team_policy_get_update_t::member_type team) const {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      auto league_size = team.league_size();
      auto team_ID     = team.league_rank();

      auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
      auto remote_range =
          Kokkos::Experimental::get_range(num_ranks * N, other_rank);

      // Local team range
      auto team_block = (local_range.second - local_range.first) / league_size;
      auto team_block_mod =
          (local_range.second - local_range.first) % league_size;
      auto start_offset = team_ID * team_block;
      team_block =
          team_ID == league_size - 1 ? team_block + team_block_mod : team_block;
      auto team_local_range =
          Kokkos::pair(local_range.first + start_offset,
                       local_range.first + start_offset + team_block);

      // Remote team range
      team_block     = (remote_range.second - remote_range.first) / league_size;
      team_block_mod = (remote_range.second - remote_range.first) % league_size;
      start_offset   = team_ID * team_block;
      team_block =
          team_ID == league_size - 1 ? team_block + team_block_mod : team_block;
      auto team_remote_range =
          Kokkos::pair(remote_range.first + start_offset,
                       remote_range.first + start_offset + team_block);

      // printf("[%lu, %lu], [%lu, %lu], [%lu, %lu], [%lu, %lu]\n",
      //        team_remote_range.first, team_remote_range.second,
      //        team_local_range.first, team_local_range.second,
      //        remote_range.first, remote_range.second, local_range.first,
      //        local_range.second);

      // Construct team subviews
      auto v_subview_remote    = Kokkos::subview(v, team_remote_range);
      auto v_tmp_subview_local = Kokkos::subview(v_tmp, team_local_range);

      // LDC
      Kokkos::Experimental::RemoteSpaces::local_deep_copy(v_tmp_subview_local,
                                                          v_subview_remote);
    });
  }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag_put &,
                  typename team_policy_put_update_t::member_type team) const {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      auto league_size = team.league_size();
      auto team_ID     = team.league_rank();
      auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
      auto remote_range =
          Kokkos::Experimental::get_range(num_ranks * N, other_rank);

      // Local team range
      auto team_block = (local_range.second - local_range.first) / league_size;
      auto team_block_mod =
          (local_range.second - local_range.first) % league_size;
      auto start_offset = team_ID * team_block;
      team_block =
          team_ID == league_size - 1 ? team_block + team_block_mod : team_block;
      auto team_local_range =
          Kokkos::pair(local_range.first + start_offset,
                       local_range.first + start_offset + team_block);

      // Remote team range
      team_block     = (remote_range.second - remote_range.first) / league_size;
      team_block_mod = (remote_range.second - remote_range.first) % league_size;
      start_offset   = team_ID * team_block;
      team_block =
          team_ID == league_size - 1 ? team_block + team_block_mod : team_block;
      auto team_remote_range =
          Kokkos::pair(remote_range.first + start_offset,
                       remote_range.first + start_offset + team_block);

      // Construct team subviews
      auto v_subview_remote    = Kokkos::subview(v_tmp, team_remote_range);
      auto v_tmp_subview_local = Kokkos::subview(v, team_local_range);

      // LDC
      Kokkos::Experimental::RemoteSpaces::local_deep_copy(v_subview_remote,
                                                          v_tmp_subview_local);
    });
  }
#elif defined(ACCESS_LDC_USE_MULTI_LDC_BUILTIN)
  KOKKOS_FUNCTION
  void operator()(const UpdateTag_get &,
                  typename team_policy_get_update_t::member_type team) const {
    auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
    auto remote_range =
        Kokkos::Experimental::get_range(num_ranks * N, other_rank);
    auto v_subview_remote    = Kokkos::subview(v, remote_range);
    auto v_tmp_subview_local = Kokkos::subview(v_tmp, local_range);
    Kokkos::Experimental::RemoteSpaces::local_deep_copy(
        team, v_tmp_subview_local, v_subview_remote);
  }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag_put &,
                  typename team_policy_put_update_t::member_type team) const {
    auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
    auto remote_range =
        Kokkos::Experimental::get_range(num_ranks * N, other_rank);
    auto v_subview_remote    = Kokkos::subview(v_tmp, remote_range);
    auto v_tmp_subview_local = Kokkos::subview(v, local_range);
    Kokkos::Experimental::RemoteSpaces::local_deep_copy(team, v_subview_remote,
                                                        v_tmp_subview_local);
  }
#else  // Single LDC
  KOKKOS_FUNCTION
  void operator()(const UpdateTag_get &,
                  typename team_policy_get_update_t::member_type team) const {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
      auto remote_range =
          Kokkos::Experimental::get_range(num_ranks * N, other_rank);
      auto v_subview_remote    = Kokkos::subview(v, remote_range);
      auto v_tmp_subview_local = Kokkos::subview(v_tmp, local_range);
      Kokkos::Experimental::RemoteSpaces::local_deep_copy(v_tmp_subview_local,
                                                          v_subview_remote);
    });
  }
  KOKKOS_FUNCTION
  void operator()(const UpdateTag_put &,
                  typename team_policy_put_update_t::member_type team) const {
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
      auto remote_range =
          Kokkos::Experimental::get_range(num_ranks * N, other_rank);
      auto v_subview_remote    = Kokkos::subview(v_tmp, remote_range);
      auto v_tmp_subview_local = Kokkos::subview(v, local_range);
      Kokkos::Experimental::RemoteSpaces::local_deep_copy(v_subview_remote,
                                                          v_tmp_subview_local);
    });
  }
#endif

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b  = 0;
    double time      = 0;
    auto local_range = Kokkos::Experimental::get_local_range(num_ranks * N);
    auto remote_range =
        Kokkos::Experimental::get_range(num_ranks * N, other_rank);

    Kokkos::parallel_for("access_overhead-init",
                         policy_init_t(local_range.first, local_range.second),
                         *this);
    Kokkos::fence();
    RemoteSpace_t().fence();

    if (rma_op == RMA_GET) {
      for (int i = 0; i < iters; i++) {
        if (my_rank == 0) {
          time_a = timer.seconds();
#if defined(ACCESS_LDC_USE_MULTI_LDC) || \
    defined(ACCESS_LDC_USE_MULTI_LDC_BUILTIN)
          Kokkos::parallel_for("block_transfer",
                               team_policy_get_update_t(NUM_TEAMS, 1), *this);
#else
          Kokkos::parallel_for("block_transfer", team_policy_get_update_t(1, 1),
                               *this);
#endif
          Kokkos::fence();
#if defined(KOKKOS_REMOTE_SPACES_ENABLE_DEBUG) && (1)
          Kokkos::parallel_for(
              "printf values for debugging",
              Kokkos::RangePolicy(local_range.first, local_range.second),
              *this);
          Kokkos::fence();
#endif
          Kokkos::parallel_for(
              "update", policy_update_t(local_range.first, local_range.second),
              *this);
          RemoteSpace_t().fence();
          time_b = timer.seconds();
          time += time_b - time_a;
        } else {
          RemoteSpace_t().fence();
        }
      }

    } else if (rma_op == RMA_PUT) {
      for (int i = 0; i < iters; i++) {
        if (my_rank == 0) {
          time_a = timer.seconds();
#if defined(ACCESS_LDC_USE_MULTI_LDC) || \
    defined(ACCESS_LDC_USE_MULTI_LDC_BUILTIN)
          Kokkos::parallel_for("block_transfer",
                               team_policy_put_update_t(NUM_TEAMS, 1), *this);
#else
          Kokkos::parallel_for("block_transfer",
                               team_policy_put_update_t(NUM_TEAMS, 1), *this);
#endif
          Kokkos::fence();
#if defined(KOKKOS_REMOTE_SPACES_ENABLE_DEBUG) && (0)
          Kokkos::parallel_for(
              "printf values for debugging",
              Kokkos::RangePolicy(local_range.first, local_range.second),
              *this);
#endif
          Kokkos::parallel_for(
              "update", policy_update_t(local_range.first, local_range.second),
              *this);

          RemoteSpace_t().fence();
          time_b = timer.seconds();
          time += time_b - time_a;

        } else {
          RemoteSpace_t().fence();
        }
      }
    } else {
      printf("What rma_op is this? Exiting.\n");
      exit(1);
    }

#ifdef CHECK_FOR_CORRECTNESS
    if (rma_op == RMA_GET) {
      // check on rank 0
      if (my_rank == 0) {
        Kokkos::parallel_for(
            "access_overhead-check",
            policy_check_t(local_range.first, local_range.second), *this);
        Kokkos::fence();
      }
    } else {
      // check on rank 1
      if (my_rank == 1) {
        Kokkos::parallel_for(
            "access_overhead-check",
            policy_check_t(local_range.first, local_range.second), *this);
        Kokkos::fence();
      }
    }
#endif

    if (my_rank == 0) {
      double gups = 1e-9 * ((N * iters) / time);
      double size = N * sizeof(double) / 1024.0 / 1024.0;
      double bw   = gups * sizeof(double);
      if (rma_op == RMA_GET) {
        printf("access_overhead_p2p,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
               modes[mode].c_str(), N, size, iters, time, gups, bw);
      } else {
        printf("access_overhead_p2p_put,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
               modes[mode].c_str(), N, size, iters, time, gups, bw);
      }
    }
  }
};

int main(int argc, char *argv[]) {
  int mpi_thread_level_available;
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;

#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
  mpi_thread_level_required = MPI_THREAD_SINGLE;
#endif

  MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);

#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

#ifdef KRS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  Kokkos::initialize(argc, argv);

  do {
    Args_t args;
    if (!read_args(argc, argv, args)) {
      break;
    };
    if (args.mode == 0) {
      Access<PlainView_t> s(args);
      s.run();
    } else if (args.mode == 1) {
      Access_CudaAware<PlainView_t> s(args);
      s.run();
    } else if (args.mode == 2) {
      Access<RemoteView_t> s(args);
      s.run();
    } else if (args.mode == 3) {
      Access_LDC<RemoteView_t> s(args);
      s.run();
    } else {
      printf("invalid mode selected (%d)\n", args.mode);
    }
  } while (false);

  Kokkos::fence();

  Kokkos::finalize();
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#endif
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
  MPI_Finalize();
  return 0;
}

#undef CHECK_FOR_CORRECTNESS
