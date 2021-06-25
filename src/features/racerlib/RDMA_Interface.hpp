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

#include <RDMA_Worker.hpp>

namespace Kokkos {
namespace Experimental {

using namespace RACERlib;



#ifdef RAW_CUDA


template <typename T, class Team>
__device__ void pack_response(T *local_values, RdmaScatterGatherWorker<T> *sgw,
                              unsigned *completion_flag, Team &&team);

template <typename T, class Team>
__device__ void aggregate_requests(RdmaScatterGatherWorker<T> *sgw, Team &&team,
                                   unsigned num_worker_teams);

#else

template <typename T, class Team>
KOKKOS_FUNCTION void pack_response(T *local_values,
                                   RdmaScatterGatherWorker<T> *sgw,
                                   unsigned *completion_flag, Team &&team);

template <typename T, class Team>
KOKKOS_INLINE_FUNCTION void aggregate_requests(RdmaScatterGatherWorker<T> *sgw,
                                               Team &&team,
                                               unsigned num_worker_teams);

#endif // RAW_CUDA

template <class Policy, class Lambda, class RemoteView>
struct RemoteParallelFor {
  KOKKOS_FUNCTION void
  operator()(const typename Policy::member_type &team) const {
    RdmaScatterGatherWorker<int> *sgw = m_view(0).sgw;
    if (team.league_rank() == 0) {
      aggregate_requests(sgw, team, team.league_size() - 2);
    } else if (team.league_rank() == 1) {
      pack_response(m_view(0).ptr, sgw, sgw->response_done_flag, team);
    } else {
      auto new_team = team;//team.shrink_league(2);
      m_lambda(new_team);
      team.team_barrier();
      Kokkos::single(
          Kokkos::PerTeam(team),
          KOKKOS_LAMBDA() { atomic_fetch_add(sgw->request_done_flag, 1); });
    }
  }

  template <class L, class R>
  RemoteParallelFor(L &&lambda, R &&view)
      : m_lambda(std::forward<L>(lambda)), m_view(std::forward<R>(view)) {}

private:
  Lambda m_lambda;
  RemoteView m_view;
};

template <class Policy, class RemoteView> struct RespondParallelFor {
  RespondParallelFor(const RemoteView &view) : m_view(view) {}

  KOKKOS_FUNCTION void
  operator()(const typename Policy::member_type &team) const {
    RdmaScatterGatherWorker<int> *sgw = m_view(0).sgw;
    pack_response(m_view(0).ptr, sgw, sgw->fence_done_flag, team);
  }

private:
  RemoteView m_view;
};

template <class Policy, class Lambda, class RemoteView>
void remote_parallel_for(const std::string &name, Policy &&policy,
                         Lambda &&lambda, const RemoteView &view) {

  if (policy.league_size() == 0) {
    return;
  }
  using PolicyType = typename std::remove_reference<Policy>::type;
  using LambdaType = typename std::remove_reference<Lambda>::type;


  RemoteParallelFor<PolicyType, LambdaType, RemoteView> rpf(
      std::forward<Lambda>(lambda), view);

  int num_teams = policy.league_size();

#ifdef KOKKOS_ENABLE_CUDA
  int vector_length = policy.vector_length();
#else
  int vector_length = 1;
#endif
  PolicyType new_policy(num_teams + 2, policy.team_size(), vector_length);
  using remote_space = typename RemoteView::memory_space;
  using exec_space = typename RemoteView::execution_space;

  Kokkos::parallel_for(name, new_policy, rpf);
  exec_space().fence();

  RespondParallelFor<PolicyType, RemoteView> txpf(view);

  auto respond_policy =
      Kokkos::TeamPolicy<>(1, policy.team_size() * vector_length);
  Kokkos::parallel_for("respond", respond_policy, txpf);

  remote_space().fence();
  
  view.impl_map().clear_fence(exec_space{});
}

} // namespace Experimental
} // namespace Kokkos

#endif // RACERLIB_RDMA_INTERFACE