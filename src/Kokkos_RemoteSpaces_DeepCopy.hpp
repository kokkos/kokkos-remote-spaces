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

#ifndef KOKKOS_REMOTESPACES_DEEPCOPY_HPP
#define KOKKOS_REMOTESPACES_DEEPCOPY_HPP

#include <Kokkos_RemoteSpaces.hpp>

namespace Kokkos {
namespace Experimental {

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
typedef NVSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_SHMEMSPACE
typedef SHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_MPISPACE
typedef MPISpace DefaultRemoteMemorySpace;
#endif
#endif
#endif

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible
 * type, same non-zero rank, same contiguous layout.
 */

template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<(
        std::is_same<typename ViewTraits<DT, DP...>::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
        std::is_same<typename ViewTraits<ST, SP...>::specialize, void>::value &&
        (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
         unsigned(ViewTraits<ST, SP...>::rank) != 0))>::type * = nullptr) {
  typedef View<DT, DP...> dst_type;
  typedef View<ST, SP...> src_type;
  typedef typename dst_type::execution_space dst_execution_space;
  typedef typename src_type::execution_space src_execution_space;
  typedef typename dst_type::memory_space dst_memory_space;
  typedef typename src_type::memory_space src_memory_space;
  typedef typename dst_type::value_type dst_value_type;
  typedef typename src_type::value_type src_value_type;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::SpaceHandle(dst_memory_space::name()), dst.label(),
        dst.data(), Kokkos::Profiling::SpaceHandle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }
#endif

  if (dst.data() == nullptr || src.data() == nullptr) {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    // do nothing
#else
    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      for (int r = 0; r < dst_type::Rank - 1; r++) {
        message += std::to_string(dst.extent(r));
        message += ",";
      }
      message += std::to_string(dst.extent(dst_type::Rank - 1));
      message += ") ";
      message += src.label();
      message += "(";
      for (int r = 0; r < src_type::Rank - 1; r++) {
        message += std::to_string(src.extent(r));
        message += ",";
      }
      message += std::to_string(src.extent(src_type::Rank - 1));
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }
#endif
    DefaultRemoteMemorySpace().fence();
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
#endif
    return;
  }

  enum {
    DstExecCanAccessSrc =
        Kokkos::Impl::SpaceAccessibility<dst_execution_space,
                                         src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::Impl::SpaceAccessibility<src_execution_space,
                                         dst_memory_space>::accessible
  };

  // Checking for Overlapping Views.
  dst_value_type *dst_start = dst.data();
  dst_value_type *dst_end = dst.data() + dst.span();
  src_value_type *src_start = src.data();
  src_value_type *src_end = src.data() + src.span();
  if (((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end == (std::ptrdiff_t)src_end) &&
      (dst.span_is_contiguous() && src.span_is_contiguous())) {
    DefaultRemoteMemorySpace().fence();
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
#endif
    return;
  }

  if ((((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
       ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) &&
      ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end);
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)src_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)src_end);
    message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    DefaultRemoteMemorySpace().fence();
    if (DstExecCanAccessSrc) {
      // Copying data between views in accessible memory spaces and either
      // non-contiguous or incompatible shape.
      Kokkos::Impl::ViewRemap<dst_type, src_type>(dst, src);
    } else if (SrcExecCanAccessDst) {
      // Copying data between views in accessible memory spaces and either
      // non-contiguous or incompatible shape.
      Kokkos::Impl::ViewRemap<dst_type, src_type, src_execution_space>(dst,
                                                                       src);
    } else {
      Kokkos::Impl::throw_runtime_exception(
          "deep_copy given views that would require a temporary allocation");
    }
    DefaultRemoteMemorySpace().fence();
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
#endif
    return;
#else
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    for (int r = 0; r < dst_type::Rank - 1; r++) {
      message += std::to_string(dst.extent(r));
      message += ",";
    }
    message += std::to_string(dst.extent(dst_type::Rank - 1));
    message += ") ";
    message += src.label();
    message += "(";
    for (int r = 0; r < src_type::Rank - 1; r++) {
      message += std::to_string(src.extent(r));
      message += ",";
    }
    message += std::to_string(src.extent(src_type::Rank - 1));
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
#endif
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
      (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
       (dst_type::rank == 1 && src_type::rank == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
    DefaultRemoteMemorySpace().fence();
    if ((void *)dst.data() != (void *)src.data()) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
          dst.data(), src.data(), nbytes);
    }
    DefaultRemoteMemorySpace().fence();
  } else {
    DefaultRemoteMemorySpace().fence();
    // Kokkos::Impl::view_copy(dst, src); //not implemented
    Kokkos::abort("Error: Not implemented.");
    DefaultRemoteMemorySpace().fence();
  }
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endDeepCopy();
  }
#endif
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible
 * type, same non-zero rank, same contiguous layout.
 */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<(
        std::is_same<typename ViewTraits<DT, DP...>::specialize, void>::value &&
        std::is_same<typename ViewTraits<ST, SP...>::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
        (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
         unsigned(ViewTraits<ST, SP...>::rank) != 0))>::type * = nullptr) {
  typedef View<DT, DP...> dst_type;
  typedef View<ST, SP...> src_type;
  typedef typename dst_type::execution_space dst_execution_space;
  typedef typename src_type::execution_space src_execution_space;
  typedef typename dst_type::memory_space dst_memory_space;
  typedef typename src_type::memory_space src_memory_space;
  typedef typename dst_type::value_type dst_value_type;
  typedef typename src_type::value_type src_value_type;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::SpaceHandle(dst_memory_space::name()), dst.label(),
        dst.data(), Kokkos::Profiling::SpaceHandle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }
#endif

  if (dst.data() == nullptr || src.data() == nullptr) {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    // do nothing
#else
    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      for (int r = 0; r < dst_type::Rank - 1; r++) {
        message += std::to_string(dst.extent(r));
        message += ",";
      }
      message += std::to_string(dst.extent(dst_type::Rank - 1));
      message += ") ";
      message += src.label();
      message += "(";
      for (int r = 0; r < src_type::Rank - 1; r++) {
        message += std::to_string(src.extent(r));
        message += ",";
      }
      message += std::to_string(src.extent(src_type::Rank - 1));
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }
#endif
    DefaultRemoteMemorySpace().fence();
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
#endif
    return;
  }

  enum {
    DstExecCanAccessSrc =
        Kokkos::Impl::SpaceAccessibility<dst_execution_space,
                                         src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::Impl::SpaceAccessibility<src_execution_space,
                                         dst_memory_space>::accessible
  };

  // Checking for Overlapping Views.
  dst_value_type *dst_start = dst.data();
  dst_value_type *dst_end = dst.data() + dst.span();
  src_value_type *src_start = src.data();
  src_value_type *src_end = src.data() + src.span();
  if (((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end == (std::ptrdiff_t)src_end) &&
      (dst.span_is_contiguous() && src.span_is_contiguous())) {
    DefaultRemoteMemorySpace().fence();
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
#endif
    return;
  }

  if ((((std::ptrdiff_t)dst_start < (std::ptrdiff_t)src_end) &&
       ((std::ptrdiff_t)dst_end > (std::ptrdiff_t)src_start)) &&
      ((dst.span_is_contiguous() && src.span_is_contiguous()))) {
    std::string message("Error: Kokkos::deep_copy of overlapping views: ");
    message += dst.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)dst_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)dst_end);
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string((std::ptrdiff_t)src_start);
    message += ",";
    message += std::to_string((std::ptrdiff_t)src_end);
    message += ") ";
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    DefaultRemoteMemorySpace().fence();
    if (DstExecCanAccessSrc) {
      // Copying data between views in accessible memory spaces and either
      // non-contiguous or incompatible shape.
      Kokkos::Impl::ViewRemap<dst_type, src_type>(dst, src);
    } else if (SrcExecCanAccessDst) {
      // Copying data between views in accessible memory spaces and either
      // non-contiguous or incompatible shape.
      Kokkos::Impl::ViewRemap<dst_type, src_type, src_execution_space>(dst,
                                                                       src);
    } else {
      Kokkos::Impl::throw_runtime_exception(
          "deep_copy given views that would require a temporary allocation");
    }
    DefaultRemoteMemorySpace().fence();
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::endDeepCopy();
    }
#endif
    return;
#else
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    for (int r = 0; r < dst_type::Rank - 1; r++) {
      message += std::to_string(dst.extent(r));
      message += ",";
    }
    message += std::to_string(dst.extent(dst_type::Rank - 1));
    message += ") ";
    message += src.label();
    message += "(";
    for (int r = 0; r < src_type::Rank - 1; r++) {
      message += std::to_string(src.extent(r));
      message += ",";
    }
    message += std::to_string(src.extent(src_type::Rank - 1));
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
#endif
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
      (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
       (dst_type::rank == 1 && src_type::rank == 1)) &&
      dst.span_is_contiguous() && src.span_is_contiguous() &&
      ((dst_type::rank < 1) || (dst.stride_0() == src.stride_0())) &&
      ((dst_type::rank < 2) || (dst.stride_1() == src.stride_1())) &&
      ((dst_type::rank < 3) || (dst.stride_2() == src.stride_2())) &&
      ((dst_type::rank < 4) || (dst.stride_3() == src.stride_3())) &&
      ((dst_type::rank < 5) || (dst.stride_4() == src.stride_4())) &&
      ((dst_type::rank < 6) || (dst.stride_5() == src.stride_5())) &&
      ((dst_type::rank < 7) || (dst.stride_6() == src.stride_6())) &&
      ((dst_type::rank < 8) || (dst.stride_7() == src.stride_7()))) {
    const size_t nbytes = sizeof(typename dst_type::value_type) * dst.span();
    DefaultRemoteMemorySpace().fence();
    if ((void *)dst.data() != (void *)src.data()) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
          dst.data(), src.data(), nbytes);
    }
    DefaultRemoteMemorySpace().fence();
  } else {
    DefaultRemoteMemorySpace().fence();
    // Kokkos::Impl::view_copy(dst, src); //not implemented
    Kokkos::abort("Error: Not implemented.");
    DefaultRemoteMemorySpace().fence();
  }
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endDeepCopy();
  }
#endif
}

} // namespace Experimental
} // namespace Kokkos

#endif // KOKKOS_REMOTESPACES_DEEPCOPY_HPP
