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

#include <Kokkos_ROCSHMEMSpace.hpp>
#include <roc_shmem.hpp>

namespace Kokkos {
namespace Experimental {

/* Default allocation mechanism */
ROCSHMEMSpace::ROCSHMEMSpace()
    : allocation_mode(Kokkos::Experimental::Symmetric) {}

void ROCSHMEMSpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void ROCSHMEMSpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *ROCSHMEMSpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  void *ptr = 0;
  if (arg_alloc_size) {
    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      int num_pes = roc_shmem_n_pes();
      int my_id   = roc_shmem_my_pe();
      ptr         = roc_shmem_malloc(arg_alloc_size);
    } else {
      Kokkos::abort("ROCSHMEMSpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}

void ROCSHMEMSpace::deallocate(void *const arg_alloc_ptr, const size_t) const {
  roc_shmem_free(arg_alloc_ptr);
}

void ROCSHMEMSpace::fence() {
  Kokkos::fence();
  roc_shmem_barrier_all();
}

KOKKOS_FUNCTION
size_t get_num_pes() { return roc_shmem_n_pes(); }

KOKKOS_FUNCTION
size_t get_my_pe() { return roc_shmem_my_pe(); }

}  // namespace Experimental

namespace Impl {

Kokkos::Impl::DeepCopy<
    HostSpace, Kokkos::Experimental::ROCSHMEMSpace>::DeepCopy(void *dst,
                                                              const void *src,
                                                              size_t n) {
  Kokkos::Experimental::ROCSHMEMSpace().fence();
  hipMemcpy(dst, src, n, hipMemcpyDefault);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::ROCSHMEMSpace,
                       HostSpace>::DeepCopy(void *dst, const void *src,
                                            size_t n) {
  Kokkos::Experimental::ROCSHMEMSpace().fence();
  hipMemcpy(dst, src, n, hipMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::ROCSHMEMSpace,
                       Kokkos::Experimental::ROCSHMEMSpace,
                       ExecutionSpace>::DeepCopy(void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::ROCSHMEMSpace().fence();
  hipMemcpy(dst, src, n, hipMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::ROCSHMEMSpace,
                       Kokkos::Experimental::ROCSHMEMSpace,
                       ExecutionSpace>::DeepCopy(const ExecutionSpace &exec,
                                                 void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::ROCSHMEMSpace().fence();
  hipMemcpy(dst, src, n, hipMemcpyDefault);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_get(void *dst, const void *src, size_t pe, size_t n) {
  roc_shmem_getmem(dst, src, pe, n);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_put(void *dst, const void *src, size_t pe, size_t n) {
  roc_shmem_putmem(dst, src, pe, n);
}

}  // namespace Impl
}  // namespace Kokkos
