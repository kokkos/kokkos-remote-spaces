
#ifndef RACERLIB_RDMA_INTERFACE
#define RACERLIB_RDMA_INTERFACE

#include <RDMA_Worker.hpp>

namespace RACERlib {

#ifdef RAW_CUDA

template <typename T, class Team>
__device__ void pack_response(T *local_values, RdmaScatterGatherWorker *sgw,
                              unsigned *completion_flag, Team &&team);

template <class Team>
__device__ void aggregate_requests(RdmaScatterGatherWorker *sgw, Team &&team,
                                   unsigned num_worker_teams);

#else

template <typename T, class Team>
KOKKOS_FUNCTION void pack_response(T *local_values,
                                   RdmaScatterGatherWorker *sgw,
                                   unsigned *completion_flag, Team &&team);

template <class Team>
KOKKOS_INLINE_FUNCTION void aggregate_requests(RdmaScatterGatherWorker *sgw,
                                               Team &&team,
                                               unsigned num_worker_teams);

template <class Policy, class Lambda, class RemoteView>
struct RemoteParallelFor {
  KOKKOS_FUNCTION void
  operator()(const typename Policy::member_type &team) const {
    RdmaScatterGatherWorker *sgw = m_view(0, 0).sg;
    if (team.league_rank() == 0) {
      aggregate_requests(sgw, team, team.league_size() - 2);
    } else if (team.league_rank() == 1) {
      pack_response(m_view(0, 0).ptr, sgw, sgw->response_done_flag, team);
    } else {
      auto new_team = team.shrink_league(2);
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
    RdmaScatterGatherWorker *sgw = m_view(0, 0).sg;
    pack_response(m_view(0, 0).ptr, sgw, sgw->fence_done_flag, team);
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
#endif // RAW_CUDA

} // namespace RACERlib

#endif // RACERLIB_RDMA_INTERFACE