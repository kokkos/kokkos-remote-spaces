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
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  void *ptr = 0;
  if (arg_alloc_size) {
    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      int num_pes = nvshmem_n_pes();
      int my_id   = nvshmem_my_pe();
      ptr         = nvshmem_malloc(arg_alloc_size);
    } else {
      Kokkos::abort("NVSHMEMSpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}

void NVSHMEMSpace::deallocate(void *const arg_alloc_ptr, const size_t) const {
  nvshmem_free(arg_alloc_ptr);
}

void NVSHMEMSpace::fence() {
  Kokkos::fence();
  nvshmem_barrier_all();
}

KOKKOS_FUNCTION
size_t get_num_pes() { return nvshmem_n_pes(); }

KOKKOS_FUNCTION
size_t get_my_pe() { return nvshmem_my_pe(); }

KOKKOS_FUNCTION
size_t get_indexing_block_size(size_t size) {
  size_t num_pes, block;
  num_pes = get_num_pes();
  block   = (size + num_pes - 1) / num_pes;
  return block;
}

std::pair<size_t, size_t> getRange(size_t size, size_t pe) {
  size_t start, end;
  size_t block = get_indexing_block_size(size);
  start        = pe * block;
  end          = (pe + 1) * block;

  size_t num_pes = get_num_pes();
  if (size < num_pes) {
    size_t diff = (num_pes * block) - size;
    if (pe > num_pes - 1 - diff) end--;
  } else {
    if (pe == num_pes - 1) {
      size_t diff = size - (num_pes - 1) * block;
      end         = start + diff;
    }
  }
  return std::make_pair(start, end);
}

KOKKOS_FUNCTION
size_t getRangeOnDevice(size_t size, size_t pe) {
  size_t start, end;
  size_t block = get_indexing_block_size(size);
  start        = pe * block;
  end          = (pe + 1) * block;

  size_t num_pes = get_num_pes();

  if (size < num_pes) {
    size_t diff = (num_pes * block) - size;
    if (pe > num_pes - 1 - diff) end--;
  } else {
    if (pe == num_pes - 1) {
      size_t diff = size - (num_pes - 1) * block;
      end         = start + diff;
    }
  }
  return end - start;
}

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

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_get(void *dst, const void *src, size_t pe, size_t n) {
  nvshmem_getmem(dst, src, pe, n);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_put(void *dst, const void *src, size_t pe, size_t n) {
  nvshmem_putmem(dst, src, pe, n);
}

}  // namespace Impl
}  // namespace Kokkos
