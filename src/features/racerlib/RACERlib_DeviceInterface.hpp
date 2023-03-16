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

#ifndef RACERLIB_DEVICE_INTERFACE
#define RACERLIB_DEVICE_INTERFACE

#include <RACERlib_DeviceWorker.hpp>
#include <RACERlib_DeviceEngine.hpp>

#define RAW_CUDA

namespace Kokkos {
namespace Experimental {

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

  // Fence the HostEngine (cache invalidate, MPI barrier, epoch++)
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

#endif  // RACERLIB_DEVICE_INTERFACE