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
struct Worker {
  KOKKOS_FUNCTION void
  operator()(const typename Policy::member_type &team) const {
    RdmaScatterGatherWorker<int> *sgw = m_view(0).sgw;
    if (team.league_rank() == 0) {
      debug_2("Launching: aggregate_requests kernel:%i\n",0);
      aggregate_requests(sgw, team, team.league_size() - 2);
    } else if (team.league_rank() == 1) {
      debug_2("Launching: pack_response kernel:%i\n",0);
      pack_response(m_view(0).ptr, sgw, sgw->response_done_flag, team);
    } else {
      debug_2("Launching: user kernel:%i\n",0);
      auto new_team = team.shrink_league(2);
      m_lambda(new_team);
      team.team_barrier();
      Kokkos::single(
          Kokkos::PerTeam(team),
          KOKKOS_LAMBDA() {            
            //Terminate
            atomic_fetch_add(sgw->request_done_flag, 1); 
            debug_2("User-kernel done:%i\n",0);
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

template <class Policy, class RemoteView> struct Respond_worker {
  Respond_worker(const RemoteView &view) : m_view(view) {}

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

  #ifdef KOKKOS_ENABLE_CUDA
  int vector_length = policy.vector_length();
  #else
  int vector_length = 1;
  #endif

  using PolicyType = typename std::remove_reference<Policy>::type;
  using LambdaType = typename std::remove_reference<Lambda>::type;
  using remote_space = typename RemoteView::memory_space;
  using exec_space = typename RemoteView::execution_space;

  PolicyType worker_policy(policy.league_size(), policy.team_size(), vector_length);
  auto respond_policy = Kokkos::TeamPolicy<>(1, policy.team_size() * vector_length);

  Worker<PolicyType, LambdaType, RemoteView> worker(
      std::forward<Lambda>(lambda), view);

  Respond_worker<PolicyType, RemoteView> respond_worker(view);
  
  Kokkos::parallel_for(name, worker_policy, worker);
  remote_space().fence();
  exec_space().fence();
  Kokkos::parallel_for("respond", respond_policy, respond_worker);
  remote_space().fence();
    exec_space().fence();
  view.impl_map().clear_fence(exec_space{});
}

} // namespace Experimental
} // namespace Kokkos

#endif // RACERLIB_RDMA_INTERFACE