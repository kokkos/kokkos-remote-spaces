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
#include <Kokkos_SHMEMSpace_AllocationRecord.hpp>

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_DEBUG
SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::Experimental::SHMEMSpace, void>::s_root_record;
#endif

SharedAllocationRecord<Kokkos::Experimental::SHMEMSpace,
                       void>::~SharedAllocationRecord() {
  // Let SharedAllocationRecordCommon do the deallocation
}

SharedAllocationHeader *_do_allocation(
    Kokkos::Experimental::SHMEMSpace const &space, std::string const &label,
    size_t alloc_size) {
  using MemAllocFailure =
      Kokkos::Impl::Experimental::RemoteSpacesMemoryAllocationFailure;
  try {
    return reinterpret_cast<SharedAllocationHeader *>(
        space.allocate(alloc_size));
  } catch (MemAllocFailure const &failure) {
    if (failure.failure_mode() ==
        MemAllocFailure::FailureMode::AllocationNotAligned) {
      // TODO: delete the misaligned memory
    }

    std::cerr << "Kokkos failed to allocate memory for label \"" << label
              << "\".  Allocation using MemorySpace named \"" << space.name()
              << " failed with the following error:  ";
    failure.print_error_message(std::cerr);
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception("Memory allocation failure");
  }
  return nullptr;  // unreachable
}

SharedAllocationRecord<Kokkos::Experimental::SHMEMSpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::SHMEMSpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::SHMEMSpace,
                                  void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
#if (KOKKOS_VERSION >= 40300)
  fill_host_accessible_header_info(this, *RecordBase::m_alloc_ptr, arg_label);
#else
  this->base_t::_fill_host_accessible_header_info(*RecordBase::m_alloc_ptr,
                                                  arg_label);
#endif
}

}  // namespace Impl
}  // namespace Kokkos

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

namespace Kokkos {
namespace Impl {

template class SharedAllocationRecordCommon<Kokkos::Experimental::SHMEMSpace>;

#undef KOKKOS_IMPL_PUBLIC_INCLUDE

}  // namespace Impl
}  // namespace Kokkos
