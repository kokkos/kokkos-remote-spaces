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

#ifndef KOKKOS_ROCSHMEMSPACE_HPP
#define KOKKOS_ROCSHMEMSPACE_HPP

#include <cstring>
#include <iosfwd>
#include <string>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>

namespace Kokkos {
namespace Experimental {

class ROCSHMEMSpace {
 public:
#if defined(KOKKOS_ENABLE_HIP)
  using execution_space = Kokkos::HIP;
#else
#error \
    "At least the following device execution space must be defined: Kokkos::HIP."
#endif
  using memory_space = ROCSHMEMSpace;
  using device_type  = Kokkos::Device<execution_space, memory_space>;
  using size_type    = size_t;

  ROCSHMEMSpace();
  ROCSHMEMSpace(ROCSHMEMSpace &&rhs)              = default;
  ROCSHMEMSpace(const ROCSHMEMSpace &rhs)         = default;
  ROCSHMEMSpace &operator=(ROCSHMEMSpace &&)      = default;
  ROCSHMEMSpace &operator=(const ROCSHMEMSpace &) = default;
  ~ROCSHMEMSpace()                                = default;

  explicit ROCSHMEMSpace(const MPI_Comm &);

  /**\brief  Allocate untracked memory in the space */
  void *allocate(const size_t arg_alloc_size) const;
  void *allocate(const char *arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void *const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char *arg_label, void *const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  void *impl_allocate(const char *arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char *arg_label, void *const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return m_name; }

  static void fence();

  int allocation_mode;
  int64_t extent;

  void impl_set_allocation_mode(const int);
  void impl_set_extent(int64_t N);

 private:
  static constexpr const char *m_name = "ROCSHMEM";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::Experimental::ROCSHMEMSpace, void>;
};

KOKKOS_FUNCTION
size_t get_num_pes();
KOKKOS_FUNCTION
size_t get_my_pe();

}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct DeepCopy<HostSpace, Kokkos::Experimental::ROCSHMEMSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <>
struct DeepCopy<Kokkos::Experimental::ROCSHMEMSpace, HostSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::ROCSHMEMSpace,
                Kokkos::Experimental::ROCSHMEMSpace, ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n);
  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n);
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::ROCSHMEMSpace,
                         Kokkos::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::Experimental::ROCSHMEMSpace> {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HIPSpace,
                         Kokkos::Experimental::ROCSHMEMSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::ROCSHMEMSpace,
                         Kokkos::HIPSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

}  // namespace Impl
}  // namespace Kokkos

#include <Kokkos_RemoteSpaces_Error.hpp>
#include <Kokkos_RemoteSpaces_Options.hpp>
#include <Kokkos_ROCSHMEM_ViewTraits.hpp>
#include <Kokkos_RemoteSpaces_ViewLayout.hpp>
#include <Kokkos_RemoteSpaces_Helpers.hpp>
#include <Kokkos_RemoteSpaces_DeepCopy.hpp>
#include <Kokkos_RemoteSpaces_ViewOffset.hpp>
#include <Kokkos_ROCSHMEM_Ops.hpp>
#include <Kokkos_ROCSHMEM_BlockOps.hpp>
#include <Kokkos_RemoteSpaces_ViewMapping.hpp>
#include <Kokkos_ROCSHMEM_AllocationRecord.hpp>
#include <Kokkos_ROCSHMEM_DataHandle.hpp>
#include <Kokkos_ROCSHMEM_LocalDeepCopy.hpp>

#endif  // #define KOKKOS_ROCSHMEMSPACE_HPP
