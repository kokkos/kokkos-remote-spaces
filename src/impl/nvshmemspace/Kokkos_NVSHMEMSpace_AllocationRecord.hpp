/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jan Ciesko (jciesko@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

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


  #if defined(KOKKOS_ENABLE_RACERLIB)
  Kokkos::Experimental::RACERlib::Engine<int> e;
  #endif

  typedef SharedAllocationRecord<void, void> RecordBase;

  SharedAllocationRecord(const SharedAllocationRecord &) = delete;
  SharedAllocationRecord &operator=(const SharedAllocationRecord &) = delete;

  static void deallocate(RecordBase *);

  /**\brief  Root record for tracked allocations from this NVSHMEMSpace instance
   */
  static RecordBase s_root_record;

  Kokkos::Experimental::NVSHMEMSpace m_space;

protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

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

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord *
  allocate(const Kokkos::Experimental::NVSHMEMSpace &arg_space,
           const std::string &arg_label, const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord *)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  static void *
  allocate_tracked(const Kokkos::Experimental::NVSHMEMSpace &arg_space,
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


  #if defined(KOKKOS_ENABLE_RACERLIB)
  Kokkos::Experimental::RACERlib::Engine<int> *  RACERlib_get_engine();
  #endif
};



} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_NVSHMEM_ALLOCREC_HPP
