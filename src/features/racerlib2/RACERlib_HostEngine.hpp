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

#ifndef RACERLIB2_INTERFACE
#define RACERLIB2_INTERFACE

#include <Kokkos_Core.hpp>
#include <Helpers.hpp>
#include <RACERlib_DeviceInterface.hpp>
#include <mpi.h>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

#define RACERLIB_SUCCESS 1

template <typename T>
struct HostEngine;

// Todo: template this on Feature for generic feature support
template <typename T>
struct HostEngine {
  DeviceWorker<T> *worker;
  MPI_Comm comm;
  int num_ranks;
  int my_rank;
  // Call this on Kokkos initialize.
  int init(
      void *target,
      MPI_Comm comm);  // set communicator reference, return RACERLIB_STATUS
  // Call this on kokkos finalize
  int finalize();  // finalize communicator instance, return RECERLIB_STATUS
  HostEngine();
  HostEngine(void *target);
  void allocate_device_component(void *data, MPI_Comm comm);
  void deallocate_device_component(void *data);
  DeviceWorker<T> *get_worker() const;
  ~HostEngine();
  void fence();
};

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos

#endif  // RACERLIB2_INTERFACE