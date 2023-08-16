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

#ifndef KOKKOS_NVSHMEMSPACE_HPP
#define KOKKOS_NVSHMEMSPACE_HPP

#include <cstring>
#include <iosfwd>
#include <string>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

class RemoteSpaceSpecializeTag {};

class NVSHMEMSpace {
 public:
#if defined(KOKKOS_ENABLE_CUDA)
  using execution_space = Kokkos::Cuda;
#else
#error \
    "At least the following device execution space must be defined: Kokkos::Cuda."
#endif
  using memory_space = NVSHMEMSpace;
  using device_type  = Kokkos::Device<execution_space, memory_space>;
  using size_type    = size_t;

  NVSHMEMSpace();
  NVSHMEMSpace(NVSHMEMSpace &&rhs)      = default;
  NVSHMEMSpace(const NVSHMEMSpace &rhs) = default;
  NVSHMEMSpace &operator=(NVSHMEMSpace &&) = default;
  NVSHMEMSpace &operator=(const NVSHMEMSpace &) = default;
  ~NVSHMEMSpace()                               = default;

  explicit NVSHMEMSpace(const MPI_Comm &);

  void *allocate(const size_t arg_alloc_size) const;

  void deallocate(void *const arg_alloc_ptr, const size_t arg_alloc_size) const;

  void *allocate(const int *gids, const int &arg_local_alloc_size) const;

  void deallocate(const int *gids, void *const arg_alloc_ptr,
                  const size_t arg_alloc_size) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return m_name; }

  void fence();

  int allocation_mode;
  int64_t extent;

  void impl_set_allocation_mode(const int);
  void impl_set_extent(int64_t N);

 private:
  static constexpr const char *m_name = "NVSHMEM";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::Experimental::NVSHMEMSpace, void>;
};

KOKKOS_FUNCTION
size_t get_num_pes();
KOKKOS_FUNCTION
size_t get_my_pe();
KOKKOS_FUNCTION
size_t get_indexing_block_size(size_t size);

std::pair<size_t, size_t> getRange(size_t size, size_t pe);
KOKKOS_FUNCTION
size_t getRangeOnDevice(size_t size, size_t pe);

}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct DeepCopy<HostSpace, Kokkos::Experimental::NVSHMEMSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <>
struct DeepCopy<Kokkos::Experimental::NVSHMEMSpace, HostSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NVSHMEMSpace,
                Kokkos::Experimental::NVSHMEMSpace, ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n);
  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n);
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::NVSHMEMSpace,
                         Kokkos::Experimental::NVSHMEMSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::Experimental::NVSHMEMSpace> {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace,
                         Kokkos::Experimental::NVSHMEMSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

}  // namespace Impl
}  // namespace Kokkos

#include <Kokkos_RemoteSpaces_ViewLayout.hpp>
#include <Kokkos_RemoteSpaces_DeepCopy.hpp>
#include <Kokkos_RemoteSpaces_Options.hpp>
#include <Kokkos_RemoteSpaces_ViewOffset.hpp>
#include <Kokkos_RemoteSpaces_ViewMapping.hpp>
#include <Kokkos_NVSHMEMSpace_Ops.hpp>
#include <Kokkos_NVSHMEMSpace_BlockOps.hpp>
#include <Kokkos_NVSHMEMSpace_AllocationRecord.hpp>
#include <Kokkos_NVSHMEMSpace_DataHandle.hpp>
#include <Kokkos_RemoteSpaces_LocalDeepCopy.hpp>
#include <Kokkos_NVSHMEMSpace_ViewTraits.hpp>

#endif  // #define KOKKOS_NVSHMEMSPACE_HPP
