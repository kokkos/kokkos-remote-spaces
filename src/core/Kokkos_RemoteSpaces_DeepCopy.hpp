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

namespace Impl {

/*REMOVE THIS LATER*/

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace, int,
          typename iType>
struct ViewCopy_ {};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 1, iType> {
  ViewTypeA a;
  ViewTypeB b;

  using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;
  using value_type  = typename ViewTypeA::value_type;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy_-1D",
                         policy_type(space, 0, a.extent(0)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0) const {
    a(i0) = static_cast<value_type>(b(i0));
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 2, iType> {
  ViewTypeA a;
  ViewTypeB b;
  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;
  using value_type = typename ViewTypeA::value_type;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy_-2D",
                         policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1) const {
    a(i0, i1) = static_cast<value_type>(b(i0, i1));
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 3, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;
  using value_type = typename ViewTypeA::value_type;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for(
        "Kokkos::ViewCopy_-3D",
        policy_type(space, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2) const {
    a(i0, i1, i2) = static_cast<value_type>(b(i0, i1, i2));
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 4, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<4, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for(
        "Kokkos::ViewCopy_-4D",
        policy_type(space, {0, 0, 0, 0},
                    {a.extent(0), a.extent(1), a.extent(2), a.extent(3)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3) const {
    a(i0, i1, i2, i3) = b(i0, i1, i2, i3);
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 5, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<5, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy_-5D",
                         policy_type(space, {0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4) const {
    a(i0, i1, i2, i3, i4) = b(i0, i1, i2, i3, i4);
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 6, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy_-6D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4), a.extent(5)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4, const iType& i5) const {
    a(i0, i1, i2, i3, i4, i5) = b(i0, i1, i2, i3, i4, i5);
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 7, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy_-7D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(3),
                                      a.extent(4), a.extent(5), a.extent(6)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i4, const iType& i5, const iType& i6) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      a(i0, i1, i2, i3, i4, i5, i6) = b(i0, i1, i2, i3, i4, i5, i6);
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_<ViewTypeA, ViewTypeB, Layout, ExecSpace, 8, iType> {
  ViewTypeA a;
  ViewTypeB b;

  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
  using iterate_type =
      Kokkos::Rank<6, outer_iteration_pattern, inner_iteration_pattern>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewCopy_(const ViewTypeA& a_, const ViewTypeB& b_,
            const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy_-8D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(3),
                                      a.extent(5), a.extent(6), a.extent(7)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i5, const iType& i6, const iType& i7) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      for (iType i4 = 0; i4 < iType(a.extent(4)); i4++)
        a(i0, i1, i2, i3, i4, i5, i6, i7) = b(i0, i1, i2, i3, i4, i5, i6, i7);
  };
};

template <class DstType, class SrcType/*, typename std::enable_if<
  std::is_same<typename DstType::traits::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value ||
  std::is_same<typename SrcType::traits::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type*/>
void view_copy_(const DstType& dst, const SrcType& src) {
  using dst_execution_space = typename DstType::execution_space;
  using src_execution_space = typename SrcType::execution_space;
  using dst_memory_space    = typename DstType::memory_space;
  using src_memory_space    = typename SrcType::memory_space;

  enum {
    DstExecCanAccessSrc =
        Kokkos::SpaceAccessibility<dst_execution_space,
                                   src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::SpaceAccessibility<src_execution_space,
                                   dst_memory_space>::accessible
  };

  if (!DstExecCanAccessSrc && !SrcExecCanAccessDst) {
    std::string message(
        "Error: Kokkos::deep_copy with no available copy mechanism: ");
    message += src.label();
    message += " to ";
    message += dst.label();
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Figure out iteration order in case we need it
  int64_t strides[DstType::Rank + 1];
  dst.stride(strides);
  Kokkos::Iterate iterate;
  if (Kokkos::is_layouttiled<typename DstType::array_layout>::value) {
    iterate = Kokkos::layout_iterate_type_selector<
        typename DstType::array_layout>::outer_iteration_pattern;
  } else if (std::is_same<typename DstType::array_layout,
                          Kokkos::PartitionedLayoutRight>::value ||
             std::is_same<typename DstType::array_layout,
                          Kokkos::LayoutRight>::value) {
    iterate = Kokkos::Iterate::Right;
  } else if (std::is_same<typename DstType::array_layout,
                          Kokkos::PartitionedLayoutLeft>::value ||
             std::is_same<typename DstType::array_layout,
                          Kokkos::LayoutLeft>::value) {
    iterate = Kokkos::Iterate::Left;
  } else if (std::is_same<typename DstType::array_layout,
                          Kokkos::PartitionedLayoutStride>::value ||
             std::is_same<typename DstType::array_layout,
                          Kokkos::LayoutStride>::value) {
    if (strides[0] > strides[DstType::Rank - 1])
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if (std::is_same<typename DstType::array_layout,
                     Kokkos::PartitionedLayoutRight>::value ||
        std::is_same<typename DstType::array_layout,
                     Kokkos::LayoutRight>::value)
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }

  if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
      (src.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (DstExecCanAccessSrc) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutRight,
                                dst_execution_space, DstType::Rank, int64_t>(
            dst, src);
      else
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutLeft,
                                dst_execution_space, DstType::Rank, int64_t>(
            dst, src);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutRight,
                                src_execution_space, DstType::Rank, int64_t>(
            dst, src);
      else
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutLeft,
                                src_execution_space, DstType::Rank, int64_t>(
            dst, src);
    }
  } else {
    if (DstExecCanAccessSrc) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutRight,
                                dst_execution_space, DstType::Rank, int>(dst,
                                                                         src);
      else
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutLeft,
                                dst_execution_space, DstType::Rank, int>(dst,
                                                                         src);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutRight,
                                src_execution_space, DstType::Rank, int>(dst,
                                                                         src);
      else
        Kokkos::Impl::ViewCopy_<DstType, SrcType, Kokkos::LayoutLeft,
                                src_execution_space, DstType::Rank, int>(dst,
                                                                         src);
    }
  }
}

}  // namespace  Impl

// namespace Experimental {

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
typedef Kokkos::Experimental::NVSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_SHMEMSPACE
typedef Kokkos::Experimental::SHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_MPISPACE
typedef Kokkos::Experimental::MPISpace DefaultRemoteMemorySpace;
#endif
#endif
#endif

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    typename std::enable_if<(
        (std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value ||
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value) &&
        (unsigned(ViewTraits<DT, DP...>::rank) == unsigned(0) &&
         unsigned(ViewTraits<ST, SP...>::rank) == unsigned(0)))>::type* =
        nullptr) {
  using dst_type = View<DT, DP...>;
  using src_type = View<ST, SP...>;

  using value_type       = typename dst_type::value_type;
  using dst_memory_space = typename dst_type::memory_space;
  using src_memory_space = typename src_type::memory_space;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename src_type::non_const_value_type>::value,
                "deep_copy requires matching non-const destination type");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }

  if (dst.data() == nullptr && src.data() == nullptr) {
    Kokkos::fence(
        "Kokkos::deep_copy: scalar to scalar copy, both pointers null");

    DefaultRemoteMemorySpace().fence();

    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::fence("Kokkos::deep_copy: scalar to scalar copy, pre copy fence");

  DefaultRemoteMemorySpace().fence();

  if (dst.data() != src.data()) {
    Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
        dst.data(), src.data(), sizeof(value_type));
    Kokkos::fence("Kokkos::deep_copy: scalar to scalar copy, post copy fence");

    DefaultRemoteMemorySpace().fence();
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible
 * type, same non-zero rank, same contiguous layout.
 */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    typename std::enable_if<(
        (std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value ||
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value) &&
        (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
         unsigned(ViewTraits<ST, SP...>::rank) != 0))>::type* = nullptr) {
  using dst_type            = View<DT, DP...>;
  using src_type            = View<ST, SP...>;
  using dst_execution_space = typename dst_type::execution_space;
  using src_execution_space = typename src_type::execution_space;
  using dst_memory_space    = typename dst_type::memory_space;
  using src_memory_space    = typename src_type::memory_space;
  using dst_value_type      = typename dst_type::value_type;
  using src_value_type      = typename src_type::value_type;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename dst_type::value_type));
  }

  if (dst.data() == nullptr || src.data() == nullptr) {
    // throw if dimension mismatch
    if (/*(src.extent(0) != dst.extent(0)) || */ (src.extent(1) !=
                                                  dst.extent(1)) ||
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
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, fence due to null "
        "argument");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  enum {
    DstExecCanAccessSrc =
        Kokkos::SpaceAccessibility<dst_execution_space,
                                   src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::SpaceAccessibility<src_execution_space,
                                   dst_memory_space>::accessible
  };

  // Checking for Overlapping Views.
  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();
  if (((std::ptrdiff_t)dst_start == (std::ptrdiff_t)src_start) &&
      ((std::ptrdiff_t)dst_end == (std::ptrdiff_t)src_end) &&
      (dst.span_is_contiguous() && src.span_is_contiguous())) {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, fence due to same "
        "spans");

    DefaultRemoteMemorySpace().fence();

    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
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
  if (/*(src.extent(0) != dst.extent(0)) ||*/ (src.extent(1) !=
                                               dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
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
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy
  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
          (std::is_same<typename dst_type::array_layout,
                        typename src_type::array_layout>::value ||
           ((std::is_same<typename dst_type::array_layout,
                          typename Kokkos::PartitionedLayoutRight>::value &&
             std::is_same<typename src_type::array_layout,
                          typename Kokkos::LayoutRight>::value) ||
            (std::is_same<typename dst_type::array_layout,
                          typename Kokkos::PartitionedLayoutLeft>::value &&
             std::is_same<typename src_type::array_layout,
                          typename Kokkos::LayoutLeft>::value) ||
            (std::is_same<typename src_type::array_layout,
                          typename Kokkos::PartitionedLayoutRight>::value &&
             std::is_same<typename dst_type::array_layout,
                          typename Kokkos::LayoutRight>::value) ||
            (std::is_same<typename src_type::array_layout,
                          typename Kokkos::PartitionedLayoutLeft>::value &&
             std::is_same<typename dst_type::array_layout,
                          typename Kokkos::LayoutLeft>::value))) ||
      (dst_type::rank == 1 && src_type::rank == 1) &&
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
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre deep copy "
        "fence");

    DefaultRemoteMemorySpace().fence();

    if ((void*)dst.data() != (void*)src.data()) {
      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
          dst.data(), src.data(), nbytes);
      Kokkos::fence(
          "Kokkos::deep_copy: copy between contiguous views, post deep copy "
          "fence");

      DefaultRemoteMemorySpace().fence();
    }
  } else {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre copy fence");

    DefaultRemoteMemorySpace().fence();

    Impl::view_copy_(dst, src);
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, post copy fence");

    DefaultRemoteMemorySpace().fence();
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible
 * type, same non-zero rank
 */
template <class ExecSpace, class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const ExecSpace& exec_space, const View<DT, DP...>& dst,
    const View<ST, SP...>& src,
    typename std::enable_if<(
        Kokkos::is_execution_space<ExecSpace>::value &&
        (std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value ||
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value) &&
        (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
         unsigned(ViewTraits<ST, SP...>::rank) != 0))>::type* = nullptr) {
  using dst_type            = View<DT, DP...>;
  using src_type            = View<ST, SP...>;
  using dst_execution_space = typename dst_type::execution_space;
  using src_execution_space = typename src_type::execution_space;
  using dst_memory_space    = typename dst_type::memory_space;
  using src_memory_space    = typename src_type::memory_space;
  using dst_value_type      = typename dst_type::value_type;
  using src_value_type      = typename src_type::value_type;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(dst_memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(), dst.span() * sizeof(dst_value_type));
  }

  dst_value_type* dst_start = dst.data();
  dst_value_type* dst_end   = dst.data() + dst.span();
  src_value_type* src_start = src.data();
  src_value_type* src_end   = src.data() + src.span();

  // Early dropout if identical range
  if ((dst_start == nullptr || src_start == nullptr) ||
      ((std::ptrdiff_t(dst_start) == std::ptrdiff_t(src_start)) &&
       (std::ptrdiff_t(dst_end) == std::ptrdiff_t(src_end)))) {
    // throw if dimension mismatch
    if (/*(src.extent(0) != dst.extent(0)) ||*/ (src.extent(1) !=
                                                 dst.extent(1)) ||
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
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  enum {
    ExecCanAccessSrcDst =
        Kokkos::SpaceAccessibility<ExecSpace, dst_memory_space>::accessible &&
        Kokkos::SpaceAccessibility<ExecSpace, src_memory_space>::accessible
  };
  enum {
    DstExecCanAccessSrc =
        Kokkos::SpaceAccessibility<dst_execution_space,
                                   src_memory_space>::accessible
  };

  enum {
    SrcExecCanAccessDst =
        Kokkos::SpaceAccessibility<src_execution_space,
                                   dst_memory_space>::accessible
  };

  // Error out for non-identical overlapping views.
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
  if (/*(src.extent(0) != dst.extent(0)) ||*/ (src.extent(1) !=
                                               dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
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
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy
  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
          (std::is_same<typename dst_type::array_layout,
                        typename src_type::array_layout>::value ||
           ((std::is_same<typename dst_type::array_layout,
                          typename Kokkos::PartitionedLayoutRight>::value &&
             std::is_same<typename src_type::array_layout,
                          typename Kokkos::LayoutRight>::value) ||
            (std::is_same<typename dst_type::array_layout,
                          typename Kokkos::PartitionedLayoutLeft>::value &&
             std::is_same<typename src_type::array_layout,
                          typename Kokkos::LayoutLeft>::value) ||
            (std::is_same<typename src_type::array_layout,
                          typename Kokkos::PartitionedLayoutRight>::value &&
             std::is_same<typename dst_type::array_layout,
                          typename Kokkos::LayoutRight>::value) ||
            (std::is_same<typename src_type::array_layout,
                          typename Kokkos::PartitionedLayoutLeft>::value &&
             std::is_same<typename dst_type::array_layout,
                          typename Kokkos::LayoutLeft>::value))) ||
      (dst_type::rank == 1 && src_type::rank == 1) &&
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
    if ((void*)dst.data() != (void*)src.data()) {
      DefaultRemoteMemorySpace().fence();

      Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space, ExecSpace>(
          exec_space, dst.data(), src.data(), nbytes);

      DefaultRemoteMemorySpace().fence();
    }
  } else {
    // Copying data between views in accessible memory spaces and either
    // non-contiguous or incompatible shape.
    if (ExecCanAccessSrcDst) {
      Impl::view_copy(exec_space, dst, src);
    } else if (DstExecCanAccessSrc || SrcExecCanAccessDst) {
      using cpy_exec_space =
          typename std::conditional<DstExecCanAccessSrc, dst_execution_space,
                                    src_execution_space>::type;
      exec_space.fence(
          "Kokkos::deep_copy: view-to-view noncontiguous copy on space, pre "
          "copy");

      DefaultRemoteMemorySpace().fence();

      Impl::view_copy(cpy_exec_space(), dst, src);
      cpy_exec_space().fence(
          "Kokkos::deep_copy: view-to-view noncontiguous copy on space, post "
          "copy");

      DefaultRemoteMemorySpace().fence();

    } else {
      Kokkos::Impl::throw_runtime_exception(
          "deep_copy given views that would require a temporary allocation");
    }
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_DEEPCOPY_HPP
