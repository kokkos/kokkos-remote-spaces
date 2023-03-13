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

#if defined(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)

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

#endif  // KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION

template <class Traits>
struct ViewDataHandle<
    Traits, typename std::enable_if<std::is_same<
                typename Traits::specialize,
                Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type> {
  using value_type  = typename Traits::value_type;
  using handle_type = NVSHMEMDataHandle<value_type, Traits>;
  using return_type = NVSHMEMDataElement<value_type, Traits>;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  template <class T = Traits>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      value_type *arg_data_ptr, track_type const & /*arg_tracker*/,
      typename std::enable_if_t<
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::is_cached> * =
          0) {
    return handle_type(arg_data_ptr);
  }

#if defined(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)

  template <class T = Traits>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      value_type *arg_data_ptr, track_type const &arg_tracker,
      typename std::enable_if<
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::is_cached> * =
          0) {
    auto *record =
        arg_tracker.template get_record<Kokkos::Experimental::NVSHMEMSpace>();
    return handle_type(
        arg_data_ptr,
        record->get_caching_and_aggregation_engine()->get_worker());
  }

#endif  // KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION

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
