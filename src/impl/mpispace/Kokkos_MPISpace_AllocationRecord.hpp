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

#ifndef KOKKOS_REMOTESPACES_MPI_ALLOCREC_HPP
#define KOKKOS_REMOTESPACES_MPI_ALLOCREC_HPP

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>
    : public SharedAllocationRecordCommon<Kokkos::Experimental::MPISpace> {
 private:
  friend Kokkos::Experimental::MPISpace;
  friend class SharedAllocationRecordCommon<Kokkos::Experimental::MPISpace>;

  using base_t = SharedAllocationRecordCommon<Kokkos::Experimental::MPISpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

#ifdef KOKKOS_ENABLE_DEBUG
  /**\brief  Root record for tracked allocations from this HostSpace instance */
  static RecordBase s_root_record;
#endif

  const Kokkos::Experimental::MPISpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  // This constructor does not forward to the one without exec_space arg
  // in order to work around https://github.com/kokkos/kokkos/issues/5258
  // This constructor is templated so I can't just put it into the cpp file
  // like the other constructor.
  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /* exec_space*/,
      const Kokkos::Experimental::MPISpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
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
#if (KOKKOS_VERSION >= 40300)
    fill_host_accessible_header_info(this, *RecordBase::m_alloc_ptr, arg_label);
#else
    this->base_t::_fill_host_accessible_header_info(*RecordBase::m_alloc_ptr,
                                                    arg_label);
#endif
  }

  SharedAllocationRecord(
      const Kokkos::Experimental::MPISpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  MPI_Win win;

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const Kokkos::Experimental::MPISpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size) {
    KOKKOS_IF_ON_HOST((return new SharedAllocationRecord(arg_space, arg_label,
                                                         arg_alloc_size);))
    KOKKOS_IF_ON_DEVICE(((void)arg_space; (void)arg_label; (void)arg_alloc_size;
                         return nullptr;))
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_MPI_ALLOCREC_HPP
