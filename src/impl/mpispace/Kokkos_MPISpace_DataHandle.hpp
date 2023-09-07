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

#ifndef KOKKOS_REMOTESPACES_MPI_DATAHANDLE_HPP
#define KOKKOS_REMOTESPACES_MPI_DATAHANDLE_HPP

namespace Kokkos {
namespace Impl {

template <class T, class Traits>
struct MPIDataHandle {
  T *ptr;
  MPIAccessLocation loc;

  KOKKOS_INLINE_FUNCTION
  MPIDataHandle() : ptr(NULL), loc(MPI_WIN_NULL, 0) {}

  KOKKOS_INLINE_FUNCTION
  MPIDataHandle(T *ptr_, MPI_Win win_ = MPI_WIN_NULL, size_t offset_ = 0)
      : ptr(ptr_ + offset_), loc(win_, offset_) {}

  KOKKOS_INLINE_FUNCTION
  MPIDataHandle(MPIDataHandle<T, Traits> const &arg)
      : ptr(arg.ptr), loc(arg.loc) {}

  template <typename iType>
  KOKKOS_INLINE_FUNCTION MPIDataElement<T, Traits> operator()(
      const int &pe, const iType &i) const {
    assert(loc.win != MPI_WIN_NULL);
    MPIDataElement<T, Traits> element(&loc.win, pe, i + loc.offset);
    return element;
  }

  KOKKOS_INLINE_FUNCTION
  MPIDataHandle operator+(size_t &offset) {
    return MPIDataHandle(ptr += offset, loc.offset += offset);
  }
};

template <class T, class Traits>
struct BlockDataHandle {
  T *ptr;
  MPIAccessLocation loc;
  size_t pe;
  size_t elems;

  KOKKOS_INLINE_FUNCTION
  BlockDataHandle(T *ptr_, MPI_Win win_, size_t offset_, size_t elems_,
                  size_t pe_)
      : ptr(ptr_), loc(win_, offset_), elems(elems_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  BlockDataHandle(BlockDataHandle<T, Traits> const &arg)
      : ptr(arg.ptr), loc(arg.loc), elems(arg.elems), pe(arg.pe) {}

  KOKKOS_INLINE_FUNCTION
  void get() {
    MPIBlockDataElement<T, Traits> element(ptr, loc.win, pe, loc.offset, elems);
    element.get();
  }

  KOKKOS_INLINE_FUNCTION
  void put() {
    MPIBlockDataElement<T, Traits> element(ptr, loc.win, pe, loc.offset, elems);
    element.put();
  }
};

template <class Traits>
struct ViewDataHandle<
    Traits, typename std::enable_if<std::is_same<
                typename Traits::specialize,
                Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type> {
  using value_type  = typename Traits::value_type;
  using handle_type = MPIDataHandle<value_type, Traits>;
  using return_type = MPIDataElement<value_type, Traits>;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const &arg_data_ptr, track_type const & /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type *arg_data_ptr,
                            track_type const & /*arg_tracker*/) {
    return handle_type(arg_data_ptr);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const arg_data_ptr, size_t offset) {
    return handle_type(arg_data_ptr + offset);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type assign(
      SrcHandleType const arg_data_ptr, MPI_Win win, size_t offset) {
    return handle_type(arg_data_ptr.ptr, win, offset);
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

#endif  // KOKKOS_REMOTESPACES_MPI_DATAHANDLE_HPP