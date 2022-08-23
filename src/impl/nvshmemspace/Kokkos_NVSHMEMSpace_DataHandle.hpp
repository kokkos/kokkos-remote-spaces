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

#ifndef KOKKOS_REMOTESPACES_NVSHMEM_DATAHANDLE_HPP
#define KOKKOS_REMOTESPACES_NVSHMEM_DATAHANDLE_HPP

#if defined(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)
#include <RDMA_Worker.hpp>
#endif  // KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION

namespace Kokkos {
namespace Impl {

template <class T, class Traits, class IsCached = void>
struct NVSHMEMDataHandle;

template <class T, class Traits>
struct NVSHMEMDataHandle<
    T, Traits,
    typename std::enable_if<(
        !RemoteSpaces_MemoryTraits<
            typename Traits::memory_traits>::is_cached)>::type> {
  T *ptr;
  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataHandle() : ptr(NULL) {}
  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataHandle(T *ptr_) : ptr(ptr_) {}
  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataHandle(NVSHMEMDataHandle<T, Traits> const &arg) : ptr(arg.ptr) {}

  template <typename iType>
  KOKKOS_INLINE_FUNCTION NVSHMEMDataElement<T, Traits> operator()(
      const int &pe, const iType &i) const {
    NVSHMEMDataElement<T, Traits> element(ptr, pe, i);
    return element;
  }

  KOKKOS_INLINE_FUNCTION
  T *operator+(size_t &offset) const { return ptr + offset; }
};

template <class T, class Traits>
struct NVSHMEMDataHandle<
    T, Traits,
    typename std::enable_if<(
        RemoteSpaces_MemoryTraits<typename Traits::memory_traits>::is_cached)>::
        type> {
  using Worker = Kokkos::Experimental::RACERlib::RdmaScatterGatherWorker<T>;
  using Engine = Kokkos::Experimental::RACERlib::Engine<T>;

  T *ptr;
  Engine *e;
  Worker *sgw;
  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataHandle() : ptr(NULL), e(NULL), sgw(NULL) {}

  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataHandle(T *ptr_, Engine *e_, Worker *sgw_)
      : ptr(ptr_), e(e_), sgw(sgw_) {}

  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataHandle(NVSHMEMDataHandle<T, Traits> const &arg)
      : ptr(arg.ptr), e(arg.e), sgw(arg.sgw) {}

  template <typename iType>
  KOKKOS_INLINE_FUNCTION NVSHMEMDataElement<T, Traits> operator()(
      const int &pe, const iType &i) const {
    NVSHMEMDataElement<T, Traits> element(ptr, sgw, pe, i);
    return element;
  }

  KOKKOS_INLINE_FUNCTION
  T *operator+(size_t &offset) const { return ptr + offset; }
};

template <class Traits>
struct ViewDataHandle<
    Traits,
    typename std::enable_if<(
        std::is_same<typename Traits::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
        !RemoteSpaces_MemoryTraits<
            typename Traits::memory_traits>::is_cached)>::type> {
  using value_type  = typename Traits::value_type;
  using handle_type = NVSHMEMDataHandle<value_type, Traits>;
  using return_type = NVSHMEMDataElement<value_type, Traits>;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(
      value_type *arg_data_ptr, track_type const & /*arg_tracker*/,
      typename std::enable_if_t<!RemoteSpaces_MemoryTraits<
          typename Traits::memory_traits>::is_cached> * = 0) {
    return handle_type(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(
      value_type *arg_data_ptr, track_type const &arg_tracker,
      typename std::enable_if_t<RemoteSpaces_MemoryTraits<
          typename Traits::memory_traits>::is_cached> * = 0) {
    auto *record =
        arg_tracker.template get_record<Kokkos::Experimental::NVSHMEMSpace>();
    return handle_type(arg_data_ptr,
                       record->get_caching_and_aggregation_engine()->sgw);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr + offset);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const arg_data_ptr) {
    return handle_type(arg_data_ptr);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION handle_type operator=(SrcHandleType const &rhs) {
    return handle_type(rhs);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_NVSHMEM_DATAHANDLE_HPP
