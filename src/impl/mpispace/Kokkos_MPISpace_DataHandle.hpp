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

#ifndef KOKKOS_REMOTESPACES_MPI_DATAHANDLE_HPP
#define KOKKOS_REMOTESPACES_MPI_DATAHANDLE_HPP

namespace Kokkos {
namespace Impl {

template <class T, class Traits> struct MPIDataHandle {
  T *ptr;
  mutable MPI_Win win;
  KOKKOS_INLINE_FUNCTION
  MPIDataHandle() : ptr(NULL), win(MPI_WIN_NULL) {}
  KOKKOS_INLINE_FUNCTION
  MPIDataHandle(T *ptr_, MPI_Win &win_) : ptr(ptr_), win(win_) {}
  KOKKOS_INLINE_FUNCTION
  MPIDataHandle(MPIDataHandle<T, Traits> const &arg)
      : ptr(arg.ptr), win(arg.win) {}
  KOKKOS_INLINE_FUNCTION
  MPIDataHandle(T *ptr_) : ptr(ptr_), win(MPI_WIN_NULL) {}

  template <typename iType>
  KOKKOS_INLINE_FUNCTION MPIDataElement<T, Traits>
  operator()(const int &pe, const iType &i) const {
    assert(win != MPI_WIN_NULL);
    MPIDataElement<T, Traits> element(&win, pe, i);
    return element;
  }

  KOKKOS_INLINE_FUNCTION
  T *operator+(size_t &offset) const { return ptr + offset; }
};

template <class Traits>
struct ViewDataHandle<
    Traits, typename std::enable_if<std::is_same<
                typename Traits::specialize,
                Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type> {

  using value_type = typename Traits::value_type;
  using handle_type = MPIDataHandle<value_type, Traits>;
  using return_type = MPIDataElement<value_type, Traits>;
  using track_type = Kokkos::Impl::SharedAllocationTracker;

  // Fixme: Currently unused
  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type *arg_data_ptr,
                            track_type const &arg_tracker) {
    return handle_type(
        arg_data_ptr,
        arg_tracker.template get_record<Kokkos::Experimental::MPISpace>()->win);
  }

  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION static handle_type
  assign(SrcHandleType const arg_data_ptr, size_t offset) {
    // FIXME: Invocation of handle_type constructor sets win to MPI_WIN_NULL
    // This is invoked by subview ViewMapping so subviews will likely fail
    return handle_type(arg_data_ptr + offset);
  }
};

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_REMOTESPACES_MPI_DATAHANDLE_HPP