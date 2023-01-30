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

#ifndef KOKKOS_NVSHMEM_ALLOCREC_HPP
#define KOKKOS_NVSHMEM_ALLOCREC_HPP

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::Experimental::NVSHMEMSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  friend Kokkos::Experimental::NVSHMEMSpace;

  typedef SharedAllocationRecord<void, void> RecordBase;

  SharedAllocationRecord(const SharedAllocationRecord &) = delete;
  SharedAllocationRecord &operator=(const SharedAllocationRecord &) = delete;

  static void deallocate(RecordBase *);

  /**\brief  Root record for tracked allocations from this NVSHMEMSpace instance
   */
  static RecordBase s_root_record;

  const Kokkos::Experimental::NVSHMEMSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace &execution_space,
      const Kokkos::Experimental::NVSHMEMSpace &arg_space,
      const std::string &arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

  SharedAllocationRecord(
      const Kokkos::Experimental::NVSHMEMSpace &arg_space,
      const std::string &arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  inline std::string get_label() const {
    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<Kokkos::HostSpace, Kokkos::CudaSpace>(
        &header, RecordBase::head(), sizeof(SharedAllocationHeader));
    return std::string(header.m_label);
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord *allocate(
      const Kokkos::Experimental::NVSHMEMSpace &arg_space,
      const std::string &arg_label, const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord *)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  static void *allocate_tracked(
      const Kokkos::Experimental::NVSHMEMSpace &arg_space,
      const std::string &arg_label, const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void *reallocate_tracked(void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void *const arg_alloc_ptr);

  static SharedAllocationRecord *get_record(void *arg_alloc_ptr);

  static void print_records(std::ostream &,
                            const Kokkos::Experimental::NVSHMEMSpace &,
                            bool detail = false);
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_NVSHMEM_ALLOCREC_HPP
