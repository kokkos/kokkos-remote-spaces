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

#ifndef RACERLIB_RDMA_INTERFACE
#define RACERLIB_RDMA_INTERFACE

#include <RACERlib_DeviceOps.hpp>

#define RAW_CUDA

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

#ifdef RAW_CUDA
template <typename T, class Team>
__device__ void pack_response_kernel(T *local_values,
                                     RdmaScatterGatherWorker<T> *sgw,
                                     unsigned *completion_flag, Team &&team,
                                     bool final);

template <typename T, class Team>
__device__ void aggregate_requests_kernel(RdmaScatterGatherWorker<T> *sgw,
                                          Team &&team,
                                          unsigned num_worker_teams);

#else

template <typename T, class Team>
KOKKOS_FUNCTION void pack_response_kernel(T *local_values,
                                          RdmaScatterGatherWorker<T> *sgw,
                                          unsigned *completion_flag,
                                          Team &&team, bool final);

template <typename T, class Team>
KOKKOS_INLINE_FUNCTION void aggregate_requests_kernel(
    RdmaScatterGatherWorker<T> *sgw, Team &&team, unsigned num_worker_teams);

#endif  // RAW_CUDA

template <class Policy, class Lambda, class RemoteView>
struct Worker {
  KOKKOS_FUNCTION void operator()(
      const typename Policy::member_type &team) const {
    RdmaScatterGatherWorker<double> *sgw = m_view(0).sgw;
    if (team.league_rank() == 0) {
      debug_2("Starting kernel 0 (aggregate_requests_kernel)\n");
      aggregate_requests_kernel(sgw, team, team.league_size() - 2);
    } else if (team.league_rank() == 1) {
      debug_2("Starting kernel 1 (pack_response_kernel)\n");
      pack_response_kernel(m_view(0).ptr, sgw, sgw->response_done_flag, team,
                           false);
    } else {
      debug_2("Starting kernel 2 (user)\n");
      auto new_team = team.shrink_league(2);
      m_lambda(new_team);
      team.team_barrier();
      Kokkos::single(
          Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
            debug_2("User kernel 2 done\n");
            atomic_fetch_add(sgw->request_done_flag, 1);
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
    RdmaScatterGatherWorker<double> *sgw = m_view(0).sgw;
    debug_2("Starting FINAL kernel (pack_response_kernel)\n");
    pack_response_kernel(m_view(0).ptr, sgw, sgw->fence_done_flag, team, true);
  }

 private:
  RemoteView m_view;
};

}  // namespace RACERlib

using namespace RACERlib;

template <class Policy, class Lambda, class RemoteView>
void remote_parallel_for(const std::string &name, Policy &&policy,
                         Lambda &&lambda, const RemoteView &view) {
  if (policy.league_size() == 0) {
    return;
  }

#ifdef KOKKOS_ENABLE_CUDA
  int vector_length = 1;  // policy.vector_length();
#else
  int vector_length = 1;
#endif

  using PolicyType   = typename std::remove_reference<Policy>::type;
  using LambdaType   = typename std::remove_reference<Lambda>::type;
  using remote_space = typename RemoteView::memory_space;
  using exec_space   = typename RemoteView::execution_space;

  PolicyType worker_policy(policy.league_size(), policy.team_size(),
                           vector_length);

  Worker<PolicyType, LambdaType, RemoteView> worker(
      std::forward<Lambda>(lambda), view);

  // *** Launch kernel triplet ***
  debug_2("Launch workers:%i\n", policy.league_size());
  Kokkos::parallel_for(name, worker_policy, worker);

  Kokkos::fence();
  debug_2("Workers finished\n");

  auto respond_policy =
      Kokkos::TeamPolicy<>(1, policy.team_size() * vector_length);

  Respond_worker<PolicyType, RemoteView> respond_worker(view);

  // *** Launch final respond_worker ***
  Kokkos::parallel_for("respond", respond_policy, respond_worker);

  // Fence the Engine (cache invalidate, MPI barrier, epoch++)
  view.impl_map().fence(exec_space{});

  remote_space().fence();  // MPI barier

  // Notify final kernel to finish response packing as we guarantee that no
  // remote kernels will be requesting local data
  // This only works if request messages and MPI barrier maintain ordering
  view.impl_map().clear_fence(exec_space{});

  // Wait for packing kernel to finish
  Kokkos::fence();
  debug_2("Respond worker finished\n");
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // RACERLIB_RDMA_INTERFACE