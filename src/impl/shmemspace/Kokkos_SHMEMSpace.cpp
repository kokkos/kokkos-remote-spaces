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

#include <Kokkos_SHMEMSpace.hpp>
#include <shmem.h>

namespace Kokkos {
namespace Experimental {

/* Default allocation mechanism */
SHMEMSpace::SHMEMSpace() : allocation_mode(Kokkos::Experimental::Symmetric) {}

void SHMEMSpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void SHMEMSpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *SHMEMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *SHMEMSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                           const size_t

                               arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *SHMEMSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  const size_t reported_size =
      (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment      = Kokkos::Impl::MEMORY_ALIGNMENT;
  constexpr uintptr_t alignment_mask = alignment - 1;

  void *ptr = nullptr;

  if (arg_alloc_size) {
    // Over-allocate to and round up to guarantee proper alignment.
    size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      int num_pes = shmem_n_pes();
      int my_id   = shmem_my_pe();
      ptr         = shmem_malloc(size_padded);
    } else {
      Kokkos::abort("SHMEMSpace only supports symmetric allocation policy.");
    }

    if (ptr) {
      auto address = reinterpret_cast<uintptr_t>(ptr);

      // offset enough to record the alloc_ptr
      address += sizeof(void *);
      uintptr_t rem    = address % alignment;
      uintptr_t offset = rem ? (alignment - rem) : 0u;
      address += offset;
      ptr = reinterpret_cast<void *>(address);
      // record the alloc'd pointer
      address -= sizeof(void *);
      *reinterpret_cast<void **>(address) = ptr;
    }
  }

  using MemAllocFailure =
      Kokkos::Impl::Experimental::RemoteSpacesMemoryAllocationFailure;
  using MemAllocFailureMode = Kokkos::Impl::Experimental::
      RemoteSpacesMemoryAllocationFailure::FailureMode;

  if ((ptr == nullptr) || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0)) ||
      (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)) {
    MemAllocFailureMode failure_mode =
        MemAllocFailureMode::AllocationNotAligned;
    if (ptr == nullptr) {
      failure_mode = MemAllocFailureMode::OutOfMemoryError;
    }

    MemAllocFailure::AllocationMechanism alloc_mec =
        MemAllocFailure::AllocationMechanism::SHMEMMALLOC;
    throw MemAllocFailure(arg_alloc_size, alignment, failure_mode, alloc_mec);
  }

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

void SHMEMSpace::deallocate(void *const arg_alloc_ptr,
                            const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void SHMEMSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                            const size_t arg_alloc_size,
                            const size_t

                                arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void SHMEMSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (arg_alloc_ptr) {
    Kokkos::fence("HostSpace::impl_deallocate before free");
    fence();
    size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                        reported_size);
    }
    shmem_free(arg_alloc_ptr);
  }
}

void SHMEMSpace::fence() const {
  Kokkos::fence();
  shmem_barrier_all();
}

size_t get_num_pes() { return shmem_n_pes(); }
size_t get_my_pe() { return shmem_my_pe(); }

}  // namespace Experimental

namespace Impl {

Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::SHMEMSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::SHMEMSpace().fence();
  memcpy(dst, src, n);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::SHMEMSpace, HostSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::SHMEMSpace().fence();
  memcpy(dst, src, n);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::SHMEMSpace,
                       Kokkos::Experimental::SHMEMSpace>::DeepCopy(void *dst,
                                                                   const void
                                                                       *src,
                                                                   size_t n) {
  Kokkos::Experimental::SHMEMSpace().fence();
  memcpy(dst, src, n);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::SHMEMSpace,
                       Kokkos::Experimental::SHMEMSpace,
                       ExecutionSpace>::DeepCopy(void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::SHMEMSpace().fence();
  memcpy(dst, src, n);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::SHMEMSpace,
                       Kokkos::Experimental::SHMEMSpace,
                       ExecutionSpace>::DeepCopy(const ExecutionSpace &exec,
                                                 void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::SHMEMSpace().fence();
  memcpy(dst, src, n);
}

}  // namespace Impl
}  // namespace Kokkos
