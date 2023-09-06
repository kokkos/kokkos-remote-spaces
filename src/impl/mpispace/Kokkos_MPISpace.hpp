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

#ifndef KOKKOS_MPISPACE_HPP
#define KOKKOS_MPISPACE_HPP

#include <cstring>
#include <iosfwd>
#include <string>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <vector>

namespace Kokkos {
namespace Experimental {

struct RemoteSpaceSpecializeTag {};

class MPISpace {
 public:
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
  using execution_space = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
  using execution_space = Kokkos::Threads;
#elif defined(KOKKOS_ENABLE_OPENMP)
  using execution_space = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_THREADS)
  using execution_space = Kokkos::Threads;
#elif defined(KOKKOS_ENABLE_SERIAL)
  using execution_space = Kokkos::Serial;
#else
#error \
    "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

  using memory_space = MPISpace;
  using device_type  = Kokkos::Device<execution_space, memory_space>;
  using size_type    = size_t;

  MPISpace();
  MPISpace(MPISpace &&rhs)      = default;
  MPISpace(const MPISpace &rhs) = default;
  MPISpace &operator=(MPISpace &&) = default;
  MPISpace &operator=(const MPISpace &) = default;
  ~MPISpace()                           = default;

  explicit MPISpace(const MPI_Comm &);

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
  template <class, class, class, class>
  friend class Kokkos::Experimental::LogicalMemorySpace;

  void *impl_allocate(const char *arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char *arg_label, void *const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return m_name; }

  void fence() const;

  int *rank_list;
  int allocation_mode;
  int64_t extent;

  static std::vector<MPI_Win> mpi_windows;
  static MPI_Win current_win;

  void impl_set_allocation_mode(const int);
  void impl_set_extent(int64_t N);

 private:
  static constexpr const char *m_name = "MPI";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::Experimental::MPISpace, void>;
};

size_t get_num_pes();
size_t get_my_pe();

}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct DeepCopy<HostSpace, Kokkos::Experimental::MPISpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <>
struct DeepCopy<Kokkos::Experimental::MPISpace, HostSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <>
struct DeepCopy<Kokkos::Experimental::MPISpace,
                Kokkos::Experimental::MPISpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::MPISpace, Kokkos::Experimental::MPISpace,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n);
  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n);
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::MPISpace,
                         Kokkos::Experimental::MPISpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::Experimental::MPISpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

// MPI locality based on an MPI window and offset
typedef struct MPIAccessLocation {
  mutable MPI_Win win;
  size_t offset;
  KOKKOS_INLINE_FUNCTION
  MPIAccessLocation() {
    win    = MPI_WIN_NULL;
    offset = 0;
  }

  KOKKOS_INLINE_FUNCTION
  MPIAccessLocation(MPI_Win win_, size_t offset_) {
    win    = win_;
    offset = offset_;
  }

  KOKKOS_INLINE_FUNCTION
  void operator=(const MPIAccessLocation &val) {
    win    = val.win;
    offset = val.offset;
  }
} MPIAccessLocation;

}  // namespace Impl
}  // namespace Kokkos

#include <Kokkos_RemoteSpaces_ViewLayout.hpp>
#include <Kokkos_RemoteSpaces_DeepCopy.hpp>
#include <Kokkos_RemoteSpaces_Options.hpp>
#include <Kokkos_RemoteSpaces_ViewOffset.hpp>
#include <Kokkos_RemoteSpaces_ViewMapping.hpp>
#include <Kokkos_MPISpace_Ops.hpp>
#include <Kokkos_MPISpace_BlockOps.hpp>
#include <Kokkos_MPISpace_AllocationRecord.hpp>
#include <Kokkos_MPISpace_DataHandle.hpp>
#include <Kokkos_RemoteSpaces_LocalDeepCopy.hpp>
#include <Kokkos_MPISpace_ViewTraits.hpp>

#endif  // #define KOKKOS_MPISPACE_HPP
