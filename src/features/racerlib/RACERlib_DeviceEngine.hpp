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

#ifndef RACERLIB_DEVICE_ENGINE
#define RACERLIB_DEVICE_ENGINE

#include <RACERlib_DeviceWorker.hpp>

#define RAW_CUDA

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

#ifdef RAW_CUDA
template <typename T, class Team>
__device__ void pack_response_kernel(T *local_values, DeviceWorker<T> *worker,
                                     unsigned *completion_flag, Team &&team,
                                     bool final);

template <typename T, class Team>
__device__ void aggregate_requests_kernel(DeviceWorker<T> *worker, Team &&team,
                                          unsigned num_worker_teams);

#else

template <typename T, class Team>
KOKKOS_FUNCTION void pack_response_kernel(T *local_values,
                                          DeviceWorker<T> *worker,
                                          unsigned *completion_flag,
                                          Team &&team, bool final);

template <typename T, class Team>
KOKKOS_INLINE_FUNCTION void aggregate_requests_kernel(
    DeviceWorker<T> *worker, Team &&team, unsigned num_worker_teams);

#endif  // RAW_CUDA

template <class Policy, class Lambda, class RemoteView>
struct Worker {
  KOKKOS_FUNCTION void operator()(
      const typename Policy::member_type &team) const {
    DeviceWorker<double> *worker = m_view(0).worker;
    if (team.league_rank() == 0) {
      debug_2("Starting kernel 0 (aggregate_requests_kernel)\n");
      aggregate_requests_kernel(worker, team, team.league_size() - 2);
    } else if (team.league_rank() == 1) {
      debug_2("Starting kernel 1 (pack_response_kernel)\n");
      pack_response_kernel(m_view(0).ptr, worker, worker->response_done_flag,
                           team, false);
    } else {
      debug_2("Starting kernel 2 (user)\n");
      auto new_team = team.shrink_league(2);
      m_lambda(new_team);
      team.team_barrier();
      Kokkos::single(
          Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
            debug_2("User kernel 2 done\n");
            atomic_fetch_add(worker->request_done_flag, 1);
          });
    }
  }

  template <class L, class R>
  Worker(L &&lambda, R &&view)
      : m_lambda(std::forward<L>(lambda)), m_view(std::forward<R>(view)) {}

 private:
  Lambda m_lambda;
  RemoteView m_view;
};

template <class Policy, class RemoteView>
struct Respond_worker {
  Respond_worker(const RemoteView &view) : m_view(view) {}

  KOKKOS_FUNCTION void operator()(
      const typename Policy::member_type &team) const {
    DeviceWorker<double> *worker = m_view(0).worker;
    debug_2("Starting FINAL kernel (pack_response_kernel)\n");
    pack_response_kernel(m_view(0).ptr, worker, worker->fence_done_flag, team,
                         true);
  }

 private:
  RemoteView m_view;
};

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos

#endif  // RACERLIB_DEVICE_ENGINE