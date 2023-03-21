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

#ifndef RACERLIB_HOSTENGINE
#define RACERLIB_HOSTENGINE

#include <Kokkos_Core.hpp>
#include <RDMAEngine.hpp>
#include <Helpers.hpp>
#include <RACERlib_DeviceInterface.hpp>
#include <RDMATransport.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

#define RACERLIB_SUCCESS 1

template <typename T>
struct HostEngine;

// Todo: template this on Feature for generic HostEngine feature support
template <typename T>
struct HostEngine {
  void put(void *target, T &value, int PE, int offset, MPI_Comm comm_id);
  T get(void *target, int PE, int offset, MPI_Comm comm_id);

  // Call this at View memory allocation (allocation record)
  int start(void *target, MPI_Comm comm_id);
  // Call this at View memory deallocation (~allocation record);
  int stop(void *target, MPI_Comm comm_id);
  // Call this on fence. We need to make sure that at sychronization points,
  // caches are empty
  int flush(void *allocation, MPI_Comm comm_id);
  // Call this on Kokkos initialize.
  int init(
      void *target,
      MPI_Comm comm_id);  // set communicator reference, return RACERLIB_STATUS
  // Call this on kokkos finalize
  int finalize();  // finalize communicator instance, return RECERLIB_STATUS

  RDMAEngine *rdma_engine;
  DeviceWorker<T> *worker;

  std::set<RDMAEngine *> rdma_engines;

  HostEngine();
  HostEngine(void *target, MPI_Comm comm_id);
  void allocate_host_device_component(void *p, MPI_Comm comm);
  void allocate_host_host_component();
  // Dealloc all for now.
  void deallocate_device_component();
  void deallocate_host_component();
  DeviceWorker<T> *get_worker() const;
  RDMAEngine *get_rdma_engine() const;
  ~HostEngine();
  void fence();
};

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos

#endif  // RACERLIB_HOSTENGINE