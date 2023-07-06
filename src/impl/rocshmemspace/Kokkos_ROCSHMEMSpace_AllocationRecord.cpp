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
#include <Kokkos_ROCSHMEMSpace_AllocationRecord.hpp>

namespace Kokkos {
namespace Impl {

template <typename ExecutionSpace>
SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>::
    SharedAllocationRecord(
        const ExecutionSpace &execution_space,
        const Kokkos::Experimental::ROCSHMEMSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
          execution_space,
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace,
                                  void>::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader *>(arg_space.allocate(
              sizeof(SharedAllocationHeader) + arg_alloc_size)),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif
  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void> *>(this);

  strncpy(header.m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<HIPSpace, HostSpace>(RecordBase::m_alloc_ptr, &header,
                                              sizeof(SharedAllocationHeader));
}

SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::ROCSHMEMSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace,
                                  void>::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader *>(arg_space.allocate(
              sizeof(SharedAllocationHeader) + arg_alloc_size)),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif
  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void> *>(this);

  strncpy(header.m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<HIPSpace, HostSpace>(RecordBase::m_alloc_ptr, &header,
                                              sizeof(SharedAllocationHeader));
}

SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace,
                       void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<HIPSpace, HostSpace>(
        &header, RecordBase::m_alloc_ptr, sizeof(SharedAllocationHeader));

    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(
            Kokkos::Experimental::ROCSHMEMSpace::name()),
        header.m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::Experimental::ROCSHMEMSpace, void>::s_root_record;

void SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace,
                            void>::deallocate(SharedAllocationRecord<void, void>
                                                  *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

void *SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>::
    allocate_tracked(const Kokkos::Experimental::ROCSHMEMSpace &arg_space,
                     const std::string &arg_alloc_label,
                     const size_t arg_alloc_size) {
  if (!arg_alloc_size) return (void *)0;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);
  RecordBase::increment(r);
  return r->data();
}

void SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace,
                            void>::deallocate_tracked(void *const
                                                          arg_alloc_ptr) {
  if (arg_alloc_ptr != 0) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);
    RecordBase::decrement(r);
  }
}

void *SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>::
    reallocate_tracked(void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<HIPSpace, HIPSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);
  return r_new->data();
}

SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void> *
SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>::get_record(
    void *alloc_ptr) {
  using RecordROCSHMEM =
      SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>;

  using Header = SharedAllocationHeader;

  // Copy the header from the allocation
  Header head;
  Header const *const head_hip =
      alloc_ptr ? Header::get_header(alloc_ptr) : (Header *)0;

  if (alloc_ptr) {
    Kokkos::Impl::DeepCopy<HostSpace, HIPSpace>(&head, head_hip,
                                                sizeof(SharedAllocationHeader));
  }

  RecordROCSHMEM *const record =
      alloc_ptr ? static_cast<RecordROCSHMEM *>(head.m_record)
                : (RecordROCSHMEM *)0;

  if (!alloc_ptr || record->m_alloc_ptr != head_hip) {
    Kokkos::Impl::throw_runtime_exception(std::string(
        "Kokkos::Impl::SharedAllocationRecord< "
        "Kokkos::Experimental::ROCSHMEMSpace , void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<Kokkos::Experimental::ROCSHMEMSpace, void>::
    print_records(std::ostream &s, const Kokkos::Experimental::ROCSHMEMSpace &,
                  bool detail) {
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "ROCSHMEMSpace", &s_root_record, detail);
}

}  // namespace Impl
}  // namespace Kokkos
