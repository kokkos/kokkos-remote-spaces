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

#ifndef KOKKOS_REMOTESPACES_DEEPCOPY_HPP
#define KOKKOS_REMOTESPACES_DEEPCOPY_HPP

#include <Kokkos_RemoteSpaces.hpp>

namespace Kokkos {
namespace Impl {

using namespace Kokkos::Experimental::Impl;

template <class ViewType, class Layout, class ExecSpace, int, typename iType>
struct ViewFill_RemoteSpaces;

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace, int,
          typename iType>
struct ViewCopy_RemoteSpaces;

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 0, iType> {
  ViewType a;
  typename ViewType::const_value_type val;
  using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill_RemoteSpaces-Scalar",
                         policy_type(space, 0, a.extent(0)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i) = val;
    else
      a(i) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 1, iType> {
  ViewType a;
  typename ViewType::const_value_type val;
  using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill_RemoteSpaces-1D",
                         policy_type(space, 0, a.extent(0)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i) = val;
    else
      a(i) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 2, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<2, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill_RemoteSpaces-2D",
                         policy_type(space, {0, 0}, {a.extent(0), a.extent(1)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i0, i1) = val;
    else
      a(i0, i1) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 3, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<3, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for(
        "Kokkos::ViewFill_RemoteSpaces-3D",
        policy_type(space, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i0, i1, i2) = val;
    else
      a(i0, i1, i2) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 4, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<4, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for(
        "Kokkos::ViewFill_RemoteSpaces-4D",
        policy_type(space, {0, 0, 0, 0},
                    {a.extent(0), a.extent(1), a.extent(2), a.extent(3)}),
        *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i0, i1, i2, i3) = val;
    else
      a(i0, i1, i2, i3) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 5, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<5, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill_RemoteSpaces-5D",
                         policy_type(space, {0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i0, i1, i2, i3, i4) = val;
    else
      a(i0, i1, i2, i3, i4) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 6, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<6, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill_RemoteSpaces-6D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(4), a.extent(5)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i2,
                  const iType& i3, const iType& i4, const iType& i5) const {
    if constexpr (Is_Partitioned_Layout<ViewType>::value)
      a(0, i0, i1, i2, i3, i4, i5) = val;
    else
      a(i0, i1, i2, i3, i4, i5) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 7, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<6, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::ViewFill_RemoteSpaces-7D",
                         policy_type(space, {0, 0, 0, 0, 0, 0},
                                     {a.extent(0), a.extent(1), a.extent(2),
                                      a.extent(3), a.extent(5), a.extent(6)}),
                         *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0, const iType& i1, const iType& i3,
                  const iType& i4, const iType& i5, const iType& i6) const {
    for (iType i2 = 0; i2 < iType(a.extent(2)); i2++)
      if constexpr (Is_Partitioned_Layout<ViewType>::value)
        a(0, i0, i1, i2, i3, i4, i5, i6) = val;
      else
        a(i0, i1, i2, i3, i4, i5, i6) = val;
  };
};

template <class ViewType, class Layout, class ExecSpace, typename iType>
struct ViewFill_RemoteSpaces<ViewType, Layout, ExecSpace, 8, iType> {
  ViewType a;
  typename ViewType::const_value_type val;

  using iterate_type = Kokkos::Rank<6, ViewFillLayoutSelector<Layout>::iterate,
                                    ViewFillLayoutSelector<Layout>::iterate>;
  using policy_type =
      Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

  ViewFill_RemoteSpaces(const ViewType& a_,
                        typename ViewType::const_value_type& val_,
                        const ExecSpace& space)
      : a(a_), val(val_) {
    Kokkos::parallel_for("Kokkos::VieViewFill_RemoteSpaceswFill-8D",
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
        if constexpr (Is_Partitioned_Layout<ViewType>::value)
          a(0, i0, i1, i2, i3, i4, i5, i6, i7) = val;
        else
          a(i0, i1, i2, i3, i4, i5, i6, i7) = val;
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 1,
                             iType> {
  ViewTypeA a;
  ViewTypeB b;

  using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;
  using value_type  = typename ViewTypeA::value_type;

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-1D",
                         policy_type(space, 0, a.extent(0)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType& i0) const {
    a(i0) = static_cast<value_type>(b(i0));
  };
};

template <class ViewTypeA, class ViewTypeB, class Layout, class ExecSpace,
          typename iType>
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 2,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-2D",
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
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 3,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for(
        "Kokkos::ViewCopy-3D",
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
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 4,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for(
        "Kokkos::ViewCopy-4D",
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
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 5,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-5D",
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
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 6,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-6D",
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
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 7,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-7D",
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
struct ViewCopy_RemoteSpaces<ViewTypeA, ViewTypeB, Layout, ExecSpace, 8,
                             iType> {
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

  ViewCopy_RemoteSpaces(const ViewTypeA& a_, const ViewTypeB& b_,
                        const ExecSpace space = ExecSpace())
      : a(a_), b(b_) {
    Kokkos::parallel_for("Kokkos::ViewCopy-8D",
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

template <class ExecutionSpace, class DstType, class SrcType>
void view_copy_RemoteSpaces(
    const ExecutionSpace& space, const DstType& dst, const SrcType& src,
    typename std::enable_if_t<(
        std::is_same<typename SrcType::traits::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value ||
        std::is_same<typename SrcType::traits::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>* =
        nullptr) {
  using dst_memory_space = typename DstType::memory_space;
  using src_memory_space = typename SrcType::memory_space;

  enum {
    ExecCanAccessSrc =
        Kokkos::SpaceAccessibility<ExecutionSpace, src_memory_space>::accessible
  };
  enum {
    ExecCanAccessDst =
        Kokkos::SpaceAccessibility<ExecutionSpace, dst_memory_space>::accessible
  };

  if (!(ExecCanAccessSrc && ExecCanAccessDst)) {
    Kokkos::Impl::throw_runtime_exception(
        "Kokkos::Impl::view_copy_RemoteSpaces called with invalid execution "
        "space");
  } else {
    // Figure out iteration order in case we need it
    int64_t strides[DstType::rank + 1];
    dst.stride(strides);
    Kokkos::Iterate iterate;
    if (Kokkos::is_layouttiled<typename DstType::array_layout>::value) {
      iterate = Kokkos::layout_iterate_type_selector<
          typename DstType::array_layout>::outer_iteration_pattern;
    } else if (std::is_same<typename DstType::array_layout,
                            Kokkos::LayoutRight>::value) {
      iterate = Kokkos::Iterate::Right;
    } else if (std::is_same<typename DstType::array_layout,
                            Kokkos::LayoutLeft>::value) {
      iterate = Kokkos::Iterate::Left;
    } else if (std::is_same<typename DstType::array_layout,
                            Kokkos::LayoutStride>::value) {
      if (strides[0] > strides[DstType::rank - 1])
        iterate = Kokkos::Iterate::Right;
      else
        iterate = Kokkos::Iterate::Left;
    } else {
      if (std::is_same<typename DstType::execution_space::array_layout,
                       Kokkos::LayoutRight>::value)
        iterate = Kokkos::Iterate::Right;
      else
        iterate = Kokkos::Iterate::Left;
    }

    if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
        (src.span() >= size_t(std::numeric_limits<int>::max()))) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_RemoteSpaces<DstType, SrcType,
                                            Kokkos::LayoutRight, ExecutionSpace,
                                            DstType::rank, int64_t>(dst, src,
                                                                    space);
      else
        Kokkos::Impl::ViewCopy_RemoteSpaces<DstType, SrcType,
                                            Kokkos::LayoutLeft, ExecutionSpace,
                                            DstType::rank, int64_t>(dst, src,
                                                                    space);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_RemoteSpaces<DstType, SrcType,
                                            Kokkos::LayoutRight, ExecutionSpace,
                                            DstType::rank, int>(dst, src,
                                                                space);
      else
        Kokkos::Impl::ViewCopy_RemoteSpaces<DstType, SrcType,
                                            Kokkos::LayoutLeft, ExecutionSpace,
                                            DstType::rank, int>(dst, src,
                                                                space);
    }
  }
}

template <class DstType, class SrcType>
void view_copy_RemoteSpaces(
    const DstType& dst, const SrcType& src,
    typename std::enable_if_t<(Is_View_Of_Type_RemoteSpaces<DstType>::value ||
                               Is_View_Of_Type_RemoteSpaces<SrcType>::value)>* =
        nullptr) {
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

  if (!(DstExecCanAccessSrc || SrcExecCanAccessDst)) {
    std::string message(
        "Error: Kokkos::deep_copy with no available copy mechanism: ");
    message += src.label();
    message += " to ";
    message += dst.label();
    Kokkos::Impl::throw_runtime_exception(message);
  }

  // Figure out iteration order in case we need it
  int64_t strides[DstType::rank + 1];
  dst.stride(strides);
  Kokkos::Iterate iterate;
  if (Kokkos::is_layouttiled<typename DstType::array_layout>::value) {
    iterate = Kokkos::layout_iterate_type_selector<
        typename DstType::array_layout>::outer_iteration_pattern;
  } else if (std::is_same<typename DstType::array_layout,
                          Kokkos::LayoutRight>::value) {
    iterate = Kokkos::Iterate::Right;
  } else if (std::is_same<typename DstType::array_layout,
                          Kokkos::LayoutLeft>::value) {
    iterate = Kokkos::Iterate::Left;
  } else if (std::is_same<typename DstType::array_layout,
                          Kokkos::LayoutStride>::value) {
    if (strides[0] > strides[DstType::rank - 1])
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if (std::is_same<typename DstType::execution_space::array_layout,
                     Kokkos::LayoutRight>::value)
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }

  if ((dst.span() >= size_t(std::numeric_limits<int>::max())) ||
      (src.span() >= size_t(std::numeric_limits<int>::max()))) {
    if (DstExecCanAccessSrc) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutRight, dst_execution_space,
            DstType::rank, int64_t>(dst, src);
      else
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutLeft, dst_execution_space,
            DstType::rank, int64_t>(dst, src);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutRight, src_execution_space,
            DstType::rank, int64_t>(dst, src);
      else
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutLeft, src_execution_space,
            DstType::rank, int64_t>(dst, src);
    }
  } else {
    if (DstExecCanAccessSrc) {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutRight, dst_execution_space,
            DstType::rank, int>(dst, src);
      else
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutLeft, dst_execution_space,
            DstType::rank, int>(dst, src);
    } else {
      if (iterate == Kokkos::Iterate::Right)
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutRight, src_execution_space,
            DstType::rank, int>(dst, src);
      else
        Kokkos::Impl::ViewCopy_RemoteSpaces<
            DstType, SrcType, Kokkos::LayoutLeft, src_execution_space,
            DstType::rank, int>(dst, src);
    }
  }
}

template <typename ExecutionSpace, class DT, class... DP>
inline void contiguous_fill(
    const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    typename std::enable_if_t<
        (Is_View_Of_Type_RemoteSpaces<View<DT, DP...>>::value)>* = nullptr) {
  using ViewType     = View<DT, DP...>;
  using ViewTypeFlat = Kokkos::View<
      typename ViewType::value_type*, Kokkos::LayoutRight,
      Kokkos::Device<typename ViewType::execution_space,
                     std::conditional_t<ViewType::rank == 0,
                                        typename ViewType::memory_space,
                                        Kokkos::AnonymousSpace>>,
      Kokkos::MemoryTraits<0>>;

  ViewTypeFlat dst_flat(dst.data(), dst.size());
  if (dst.span() < static_cast<size_t>(std::numeric_limits<int>::max())) {
    Kokkos::Impl::ViewFill_RemoteSpaces<ViewTypeFlat, Kokkos::LayoutRight,
                                        ExecutionSpace, ViewTypeFlat::rank,
                                        int>(dst_flat, value, exec_space);
  } else
    Kokkos::Impl::ViewFill_RemoteSpaces<ViewTypeFlat, Kokkos::LayoutRight,
                                        ExecutionSpace, ViewTypeFlat::rank,
                                        int64_t>(dst_flat, value, exec_space);
}

// Default implementation for execution spaces that don't provide a definition
template <typename ExecutionSpace, class ViewType>
struct ZeroMemset_RemoteSpaces {
  ZeroMemset_RemoteSpaces(
      const ExecutionSpace& exec_space, const ViewType& dst,
      typename ViewType::const_value_type& value,
      typename std::enable_if_t<
          Is_View_Of_Type_RemoteSpaces<ViewType>::value>* = nullptr) {
    contiguous_fill(exec_space, dst, value);
  }

  ZeroMemset_RemoteSpaces(const ViewType& dst,
                          typename ViewType::const_value_type& value) {
    contiguous_fill(ExecutionSpace(), dst, value);
  }
};

template <typename ExecutionSpace, class DT, class... DP>
inline std::enable_if_t<
    std::is_trivial<typename ViewTraits<DT, DP...>::value_type>::value &&
    std::is_trivially_copy_assignable<
        typename ViewTraits<DT, DP...>::value_type>::value>
contiguous_fill_or_memset(
    const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    typename std::enable_if_t<
        Is_View_Of_Type_RemoteSpaces<View<DT, DP...>>::value>* = nullptr) {
// On A64FX memset seems to do the wrong thing with regards to first touch
// leading to the significant performance issues
#ifndef KOKKOS_ARCH_A64FX
  if (Impl::is_zero_byte(value))
    ZeroMemset<ExecutionSpace, View<DT, DP...>>(exec_space, dst, value);
  else
#endif
    contiguous_fill(exec_space, dst, value);
}

template <typename ExecutionSpace, class DT, class... DP>
inline std::enable_if_t<
    !(std::is_trivial<typename ViewTraits<DT, DP...>::value_type>::value &&
      std::is_trivially_copy_assignable<
          typename ViewTraits<DT, DP...>::value_type>::value)>
contiguous_fill_or_memset(
    const ExecutionSpace& exec_space, const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    typename std::enable_if_t<
        Is_View_Of_Type_RemoteSpaces<View<DT, DP...>>::value>* = nullptr) {
  contiguous_fill(exec_space, dst, value);
}

template <class DT, class... DP>
inline std::enable_if_t<
    std::is_trivial<typename ViewTraits<DT, DP...>::value_type>::value &&
    std::is_trivially_copy_assignable<
        typename ViewTraits<DT, DP...>::value_type>::value>
contiguous_fill_or_memset(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    typename std::enable_if_t<
        Is_View_Of_Type_RemoteSpaces<View<DT, DP...>>::value>* = nullptr) {
  using ViewType        = View<DT, DP...>;
  using exec_space_type = typename ViewType::execution_space;

// On A64FX memset seems to do the wrong thing with regards to first touch
// leading to the significant performance issues
#ifndef KOKKOS_ARCH_A64FX
  if (Impl::is_zero_byte(value))
    ZeroMemset<exec_space_type, View<DT, DP...>>(dst, value);
  else
#endif
    contiguous_fill(exec_space_type(), dst, value);
}

template <class DT, class... DP>
inline std::enable_if_t<
    !(std::is_trivial<typename ViewTraits<DT, DP...>::value_type>::value &&
      std::is_trivially_copy_assignable<
          typename ViewTraits<DT, DP...>::value_type>::value)>
contiguous_fill_or_memset(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    typename std::enable_if_t<
        Is_View_Of_Type_RemoteSpaces<View<DT, DP...>>::value>* = nullptr) {
  using ViewType        = View<DT, DP...>;
  using exec_space_type = typename ViewType::execution_space;

  contiguous_fill(exec_space_type(), dst, value);
}
}  // namespace Impl

/** \brief  Deep copy a value from Host memory into a view.  */
template <class DT, class... DP>
inline void deep_copy(
    const View<DT, DP...>& dst,
    typename ViewTraits<DT, DP...>::const_value_type& value,
    typename std::enable_if_t<
        Kokkos::Experimental::Impl::Is_View_Of_Type_RemoteSpaces<
            View<DT, DP...>>::value>* = nullptr) {
  using ViewType        = View<DT, DP...>;
  using exec_space_type = typename ViewType::execution_space;

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(ViewType::memory_space::name()),
        dst.label(), dst.data(),
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "Scalar", &value, dst.span() * sizeof(typename ViewType::value_type));
  }

  if (dst.data() == nullptr) {
    Kokkos::fence(
        "Kokkos::deep_copy: scalar copy, fence because destination is null");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::fence("Kokkos::deep_copy: scalar copy, pre copy fence");
  static_assert(std::is_same<typename ViewType::non_const_value_type,
                             typename ViewType::value_type>::value,
                "deep_copy requires non-const type");

  // If contiguous we can simply do a 1D flat loop or use memset
  if (dst.span_is_contiguous()) {
    Impl::contiguous_fill_or_memset(dst, value);
    Kokkos::fence("Kokkos::deep_copy: scalar copy, post copy fence");
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  // Figure out iteration order to do the ViewFill
  int64_t strides[ViewType::rank + 1];
  dst.stride(strides);
  Kokkos::Iterate iterate;
  if (std::is_same<typename ViewType::array_layout,
                   Kokkos::LayoutRight>::value) {
    iterate = Kokkos::Iterate::Right;
  } else if (std::is_same<typename ViewType::array_layout,
                          Kokkos::LayoutLeft>::value) {
    iterate = Kokkos::Iterate::Left;
  } else if (std::is_same<typename ViewType::array_layout,
                          Kokkos::LayoutStride>::value) {
    if (strides[0] > strides[ViewType::rank > 0 ? ViewType::rank - 1 : 0])
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  } else {
    if (std::is_same<typename ViewType::execution_space::array_layout,
                     Kokkos::LayoutRight>::value)
      iterate = Kokkos::Iterate::Right;
    else
      iterate = Kokkos::Iterate::Left;
  }

  // Lets call the right ViewFill functor based on integer space needed and
  // iteration type
  using ViewTypeUniform =
      std::conditional_t<ViewType::rank == 0,
                         typename ViewType::uniform_runtime_type,
                         typename ViewType::uniform_runtime_nomemspace_type>;
  if (dst.span() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    if (iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight,
                             exec_space_type, ViewType::rank, int64_t>(
          dst, value, exec_space_type());
    else
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft,
                             exec_space_type, ViewType::rank, int64_t>(
          dst, value, exec_space_type());
  } else {
    if (iterate == Kokkos::Iterate::Right)
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutRight,
                             exec_space_type, ViewType::rank, int>(
          dst, value, exec_space_type());
    else
      Kokkos::Impl::ViewFill<ViewTypeUniform, Kokkos::LayoutLeft,
                             exec_space_type, ViewType::rank, int>(
          dst, value, exec_space_type());
  }
  Kokkos::fence("Kokkos::deep_copy: scalar copy, post copy fence");

  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template <class ST, class... SP>
inline void deep_copy(
    typename ViewTraits<ST, SP...>::non_const_value_type& dst,
    const View<ST, SP...>& src,
    std::enable_if_t<Kokkos::Experimental::Impl::Is_View_Of_Type_RemoteSpaces<
        View<ST, SP...>>::value>* = nullptr) {
  using src_traits       = ViewTraits<ST, SP...>;
  using src_memory_space = typename src_traits::memory_space;

  static_assert(src_traits::rank == 0,
                "ERROR: Non-rank-zero view in deep_copy( value , View )");

  if (Kokkos::Tools::Experimental::get_callbacks().begin_deep_copy != nullptr) {
    Kokkos::Profiling::beginDeepCopy(
        Kokkos::Profiling::make_space_handle(Kokkos::HostSpace::name()),
        "Scalar", &dst,
        Kokkos::Profiling::make_space_handle(src_memory_space::name()),
        src.label(), src.data(),
        src.span() * sizeof(typename src_traits::value_type));
  }

  if (src.data() == nullptr) {
    Kokkos::fence("Kokkos::deep_copy: copy into scalar, src is null");
  } else {
    Kokkos::fence("Kokkos::deep_copy: copy into scalar, pre copy fence");
    Kokkos::Impl::DeepCopy<HostSpace, src_memory_space>(&dst, src.data(),
                                                        sizeof(ST));
    Kokkos::fence("Kokkos::deep_copy: copy into scalar, post copy fence");
  }

  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<(Kokkos::Experimental::Impl::Is_View_Of_Type_RemoteSpaces<
                          View<ST, SP...>>::value &&
                      Kokkos::Experimental::Impl::Is_View_Of_Type_RemoteSpaces<
                          View<DT, DP...>>::value &&
                      (unsigned(ViewTraits<DT, DP...>::rank) == unsigned(0) &&
                       unsigned(ViewTraits<ST, SP...>::rank) ==
                           unsigned(0)))>* = nullptr) {
  using dst_type = View<DT, DP...>;
  using src_type = View<ST, SP...>;

  using value_type       = typename dst_type::value_type;
  using dst_memory_space = typename dst_type::memory_space;
  using src_memory_space = typename src_type::memory_space;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename src_type::non_const_value_type>::value,
                "deep_copy requires matching non-const destination type");

  static_assert(((Kokkos::Experimental::Impl::Is_Partitioned_Layout<
                      View<DT, DP...>>::value &&
                  Kokkos::Experimental::Impl::Is_Partitioned_Layout<
                      View<DT, DP...>>::value) ||
                 (!Kokkos::Experimental::Impl::Is_Partitioned_Layout<
                      View<DT, DP...>>::value &&
                  !Kokkos::Experimental::Impl::Is_Partitioned_Layout<
                      View<DT, DP...>>::value)),
                "ERROR: deep_copy requires compatible view types");

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
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  Kokkos::fence("Kokkos::deep_copy: scalar to scalar copy, pre copy fence");
  if (dst.data() != src.data()) {
    Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
        dst.data(), src.data(), sizeof(value_type));
    Kokkos::fence("Kokkos::deep_copy: scalar to scalar copy, post copy fence");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of RemoteSpaces specialization, compatible
 * type, same non-zero rank, same contiguous layout.
 */
template <class DT, class... DP, class ST, class... SP>
inline void deep_copy(
    const View<DT, DP...>& dst, const View<ST, SP...>& src,
    std::enable_if_t<((Kokkos::Experimental::Impl::Is_View_Of_Type_RemoteSpaces<
                           View<ST, SP...>>::value ||
                       Kokkos::Experimental::Impl::Is_View_Of_Type_RemoteSpaces<
                           View<DT, DP...>>::value) &&
                      (unsigned(ViewTraits<DT, DP...>::rank) != 0 ||
                       unsigned(ViewTraits<ST, SP...>::rank) != 0))>* =
        nullptr) {
  using dst_type = View<DT, DP...>;
  using src_type = View<ST, SP...>;

  static_assert(std::is_same<typename dst_type::value_type,
                             typename dst_type::non_const_value_type>::value,
                "deep_copy requires non-const destination type");

  static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "deep_copy requires Views of equal rank");

  using dst_execution_space = typename dst_type::execution_space;
  using src_execution_space = typename src_type::execution_space;
  using dst_memory_space    = typename dst_type::memory_space;
  using src_memory_space    = typename src_type::memory_space;
  using dst_value_type      = typename dst_type::value_type;
  using src_value_type      = typename src_type::value_type;
  using namespace Kokkos::Experimental::Impl;

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
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message(
          "Deprecation Error: Kokkos::deep_copy extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      message += std::to_string(dst.extent(0));
      for (size_t r = 1; r < dst_type::rank; r++) {
        message += ",";
        message += std::to_string(dst.extent(r));
      }
      message += ") ";
      message += src.label();
      message += "(";
      message += std::to_string(src.extent(0));
      for (size_t r = 1; r < src_type::rank; r++) {
        message += ",";
        message += std::to_string(src.extent(r));
      }
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
    if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
      Kokkos::Profiling::endDeepCopy();
    }
    return;
  }

  // Check for same extents
  if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
      (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
      (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
      (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
    std::string message(
        "Deprecation Error: Kokkos::deep_copy extents of views don't match: ");
    message += dst.label();
    message += "(";
    message += std::to_string(dst.extent(0));
    for (size_t r = 1; r < dst_type::rank; r++) {
      message += ",";
      message += std::to_string(dst.extent(r));
    }
    message += ") ";
    message += src.label();
    message += "(";
    message += std::to_string(src.extent(0));
    for (size_t r = 1; r < src_type::rank; r++) {
      message += ",";
      message += std::to_string(src.extent(r));
    }
    message += ") ";

    Kokkos::Impl::throw_runtime_exception(message);
  }

  // If same type, equal layout, equal dimensions, equal span, and contiguous
  // memory then can byte-wise copy

  if (std::is_same<typename dst_type::value_type,
                   typename src_type::non_const_value_type>::value &&
      (std::is_same<typename dst_type::array_layout,
                    typename src_type::array_layout>::value ||
       (Is_Partitioned_Layout<src_type>::value &&
        !Is_Partitioned_Layout<dst_type>::value) ||
       (Is_Partitioned_Layout<dst_type>::value &&
        !Is_Partitioned_Layout<src_type>::value) ||
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
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre view equality "
        "check");
    if ((void*)dst.data() != (void*)src.data() && 0 < nbytes) {
      // If both view are local, use data ptr
      if (is_local_view(src) && is_local_view(dst)) {
        Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
            dst.data(), src.data(), nbytes);
        Kokkos::fence(
            "Kokkos::deep_copy: copy between contiguous views, post deep copy "
            "fence");
      } else {
        if (!is_local_view(src)) {
          std::string message(
              "Error: Kokkos::deep_copy with no available copy mechanism for "
              "remote view: ");
          message += src.label();
          message += " to ";
          message += dst.label();
          Kokkos::Impl::throw_runtime_exception(message);

        } else /*(!is_local_view(dst))*/
        {
          std::string message(
              "Error: Kokkos::deep_copy with no available copy mechanism for "
              "remote view: ");
          message += dst.label();
          message += " to ";
          message += src.label();
          Kokkos::Impl::throw_runtime_exception(message);
        }
      }
    }
  } else {
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, pre copy fence");
    Impl::view_copy_RemoteSpaces(dst, src);
    Kokkos::fence(
        "Kokkos::deep_copy: copy between contiguous views, post copy fence");
  }
  if (Kokkos::Tools::Experimental::get_callbacks().end_deep_copy != nullptr) {
    Kokkos::Profiling::endDeepCopy();
  }
}

}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_DEEPCOPY_HPP
