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

#include <Kokkos_NVSHMEMSpace.hpp>
#include <nvshmem.h>

namespace Kokkos {
namespace Experimental {

/* Default allocation mechanism */
NVSHMEMSpace::NVSHMEMSpace()
    : allocation_mode(Kokkos::Experimental::Symmetric) {}

void NVSHMEMSpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void NVSHMEMSpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *NVSHMEMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *NVSHMEMSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                             const size_t

                                 arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *NVSHMEMSpace::impl_allocate(
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
    //  size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      int num_pes = nvshmem_n_pes();
      int my_id   = nvshmem_my_pe();
      ptr         = /*(Kokkos::Impl::MEMORY_ALIGNMENT,
                          arg_alloc_size); */
          nvshmem_malloc(arg_alloc_size);
    } else {
      Kokkos::abort("SHMEMSpace only supports symmetric allocation policy.");
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
        MemAllocFailure::AllocationMechanism::NVSHMEMMALLOC;
    throw MemAllocFailure(arg_alloc_size, alignment, failure_mode, alloc_mec);
  }

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

void NVSHMEMSpace::deallocate(void *const arg_alloc_ptr,
                              const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void NVSHMEMSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                              const size_t arg_alloc_size,
                              const size_t

                                  arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void NVSHMEMSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (arg_alloc_ptr) {
    Kokkos::fence("HostSpace::impl_deallocate before free");
    Kokkos::Experimental::NVSHMEMSpace().fence();
    size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                        reported_size);
    }
    nvshmem_free(arg_alloc_ptr);
  }
}

void NVSHMEMSpace::fence() {
  Kokkos::fence();
  nvshmem_barrier_all();
}

KOKKOS_FUNCTION
int get_num_pes() { return nvshmem_n_pes(); }

KOKKOS_FUNCTION
int get_my_pe() { return nvshmem_my_pe(); }

}  // namespace Experimental

namespace Impl {

Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::NVSHMEMSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::NVSHMEMSpace, HostSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::NVSHMEMSpace,
                       Kokkos::Experimental::NVSHMEMSpace,
                       ExecutionSpace>::DeepCopy(void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::NVSHMEMSpace,
                       Kokkos::Experimental::NVSHMEMSpace,
                       ExecutionSpace>::DeepCopy(const ExecutionSpace &exec,
                                                 void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

}  // namespace Impl
}  // namespace Kokkos
