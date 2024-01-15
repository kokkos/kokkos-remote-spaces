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

#include <Kokkos_MPISpace.hpp>
#include <Kokkos_MPISpace_AllocationRecord.hpp>

#include <iostream>

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_DEBUG
SharedAllocationRecord<void, void>
    SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::s_root_record;
#endif

SharedAllocationRecord<Kokkos::Experimental::MPISpace,
                       void>::~SharedAllocationRecord() {
  m_space.deallocate(m_label.c_str(),
                     SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size,
                     (SharedAllocationRecord<void, void>::m_alloc_size -
                      sizeof(SharedAllocationHeader)));
}

SharedAllocationHeader *_do_allocation(
    Kokkos::Experimental::MPISpace const &space, std::string const &label,
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

SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::MPISpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : base_t(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::MPISpace,
                                  void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
          arg_label),
      m_space(arg_space) {
  this->base_t::_fill_host_accessible_header_info(*RecordBase::m_alloc_ptr,
                                                  arg_label);
  win = m_space.current_win;
}

}  // namespace Impl
}  // namespace Kokkos

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

namespace Kokkos {
namespace Impl {

template class SharedAllocationRecordCommon<Kokkos::Experimental::MPISpace>;

#undef KOKKOS_IMPL_PUBLIC_INCLUDE

}  // namespace Impl
}  // namespace Kokkos
