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

#ifndef KOKKOS_REMOTESPACES_VIEWMAPPING_HPP
#define KOKKOS_REMOTESPACES_VIEWMAPPING_HPP

#include <type_traits>

#define ENABLE_IF_GLOBAL_LAYOUT \
  std::enable_if_t<!Is_Partitioned_Layout<T>::value> * = nullptr
#define ENABLE_IF_PARTITIONED_LAYOUT \
  std::enable_if_t<Is_Partitioned_Layout<T>::value> * = nullptr

#define USING_GLOBAL_INDEXING !remote_view_props.using_local_indexing
#define USING_LOCAL_INDEXING remote_view_props.using_local_indexing

//----------------------------------------------------------------------------
/** \brief  View mapping for specialized data type */

namespace Kokkos {
namespace Impl {

using namespace Kokkos::Experimental::Impl;

/*
 * Type used for subview creation with non-standard Layouts and default (void)
 * view specialization.
 */

template <class SrcTraits, class... Args>
class ViewMapping<
    std::enable_if_t<(std::is_void<typename SrcTraits::specialize>::value &&
                      (std::is_same<typename SrcTraits::array_layout,
                                    Kokkos::PartitionedLayoutLeft>::value ||
                       std::is_same<typename SrcTraits::array_layout,
                                    Kokkos::PartitionedLayoutRight>::value ||
                       std::is_same<typename SrcTraits::array_layout,
                                    Kokkos::PartitionedLayoutStride>::value))>,
    SrcTraits, Args...> {
 private:
  static_assert(SrcTraits::rank == sizeof...(Args),
                "Subview mapping requires one argument for each dimension of "
                "source View");

  enum {
    RZ = false,
    R0 = bool(is_integral_extent<0, Args...>::value),
    R1 = bool(is_integral_extent<1, Args...>::value),
    R2 = bool(is_integral_extent<2, Args...>::value),
    R3 = bool(is_integral_extent<3, Args...>::value),
    R4 = bool(is_integral_extent<4, Args...>::value),
    R5 = bool(is_integral_extent<5, Args...>::value),
    R6 = bool(is_integral_extent<6, Args...>::value),
    R7 = bool(is_integral_extent<7, Args...>::value)
  };

  enum {
    rank = unsigned(R0) + unsigned(R1) + unsigned(R2) + unsigned(R3) +
           unsigned(R4) + unsigned(R5) + unsigned(R6) + unsigned(R7)
  };

  // Whether right-most rank is a range.
  enum {
    R0_rev =
        (0 == SrcTraits::rank
             ? RZ
             : (1 == SrcTraits::rank
                    ? R0
                    : (2 == SrcTraits::rank
                           ? R1
                           : (3 == SrcTraits::rank
                                  ? R2
                                  : (4 == SrcTraits::rank
                                         ? R3
                                         : (5 == SrcTraits::rank
                                                ? R4
                                                : (6 == SrcTraits::rank
                                                       ? R5
                                                       : (7 == SrcTraits::rank
                                                              ? R6
                                                              : R7))))))))
  };

  // Subview's layout
  using array_layout = std::conditional_t<
      (            /* Same array layout IF */
       (rank == 0) /* output rank zero */
       || SubviewLegalArgsCompileTime<typename SrcTraits::array_layout,
                                      typename SrcTraits::array_layout, rank,
                                      SrcTraits::rank, 0, Args...>::value ||
       // OutputRank 1 or 2, InputLayout Left, Interval 0
       // because single stride one or second index has a stride.
       (rank <= 2 && R0 &&
        std::is_same<typename SrcTraits::array_layout,
                     Kokkos::LayoutLeft>::value)  // replace with input rank
       ||
       // OutputRank 1 or 2, InputLayout Right, Interval [InputRank-1]
       // because single stride one or second index has a stride.
       (rank <= 2 && R0_rev &&
        std::is_same<typename SrcTraits::array_layout,
                     Kokkos::LayoutRight>::value)  // replace input rank
       ),
      typename SrcTraits::array_layout, Kokkos::LayoutStride>;

  using value_type = typename SrcTraits::value_type;

  using data_type =
      typename SubViewDataType<value_type,
                               typename Kokkos::Impl::ParseViewExtents<
                                   typename SrcTraits::data_type>::type,
                               Args...>::type;

 public:
  using traits_type = Kokkos::ViewTraits<data_type, array_layout,
                                         typename SrcTraits::device_type,
                                         typename SrcTraits::memory_traits>;

  using type =
      Kokkos::View<data_type, array_layout, typename SrcTraits::device_type,
                   typename SrcTraits::memory_traits>;

  template <class MemoryTraits>
  struct apply {
    static_assert(Kokkos::is_memory_traits<MemoryTraits>::value, "");

    using traits_type =
        Kokkos::ViewTraits<data_type, array_layout,
                           typename SrcTraits::device_type, MemoryTraits>;

    using type = Kokkos::View<data_type, array_layout,
                              typename SrcTraits::device_type, MemoryTraits>;
  };

  // The presumed type is 'ViewMapping< traits_type , void >'
  // However, a compatible ViewMapping is acceptable.
  template <class DstTraits>
  KOKKOS_INLINE_FUNCTION static void assign(
      ViewMapping<DstTraits, void> &dst,
      ViewMapping<SrcTraits, void> const &src, Args... args) {
    static_assert(ViewMapping<DstTraits, traits_type, void>::is_assignable,
                  "Subview destination type must be compatible with subview "
                  "derived type");

    using DstType = ViewMapping<DstTraits, void>;

    using dst_offset_type = typename DstType::offset_type;

    const SubviewExtents<SrcTraits::rank, rank> extents(src.m_impl_offset.m_dim,
                                                        args...);

    dst.m_impl_offset = dst_offset_type(src.m_impl_offset, extents);

    dst.m_impl_handle = ViewDataHandle<DstTraits>::assign(
        src.m_impl_handle,
        src.m_impl_offset(extents.domain_offset(0), extents.domain_offset(1),
                          extents.domain_offset(2), extents.domain_offset(3),
                          extents.domain_offset(4), extents.domain_offset(5),
                          extents.domain_offset(6), extents.domain_offset(7)));
  }
};

/*
 * ViewMapping type used by View copy-ctr and subview() to specialize new
 * (sub-) view type
 */

template <class SrcTraits, class... Args>
class ViewMapping<
    typename std::enable_if<(
        std::is_same<typename SrcTraits::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type,
    SrcTraits, Args...> {
 private:
  static_assert(SrcTraits::rank == sizeof...(Args),
                "Subview mapping requires one argument for each dimension of "
                "source View");
  enum {
    RZ = false,
    R0 = bool(is_integral_extent<0, Args...>::value),
    R1 = bool(is_integral_extent<1, Args...>::value),
    R2 = bool(is_integral_extent<2, Args...>::value),
    R3 = bool(is_integral_extent<3, Args...>::value),
    R4 = bool(is_integral_extent<4, Args...>::value),
    R5 = bool(is_integral_extent<5, Args...>::value),
    R6 = bool(is_integral_extent<6, Args...>::value),
    R7 = bool(is_integral_extent<7, Args...>::value)
  };

  enum {
    rank = unsigned(R0) + unsigned(R1) + unsigned(R2) + unsigned(R3) +
           unsigned(R4) + unsigned(R5) + unsigned(R6) + unsigned(R7)
  };

  // Whether right-most rank is a range.
  enum {
    R0_rev =
        (0 == SrcTraits::rank
             ? RZ
             : (1 == SrcTraits::rank
                    ? R0
                    : (2 == SrcTraits::rank
                           ? R1
                           : (3 == SrcTraits::rank
                                  ? R2
                                  : (4 == SrcTraits::rank
                                         ? R3
                                         : (5 == SrcTraits::rank
                                                ? R4
                                                : (6 == SrcTraits::rank
                                                       ? R5
                                                       : (7 == SrcTraits::rank
                                                              ? R6
                                                              : R7))))))))
  };

  // Subview's layout
  using array_layout_candidate =

      typename std::conditional<
          (            /* Same array layout IF */
           (rank == 0) /* output rank zero */
           || SubviewLegalArgsCompileTime<
                  typename SrcTraits::array_layout,
                  typename SrcTraits::array_layout, rank, SrcTraits::rank, 0,
                  Args...>::value ||  // OutputRank 1 or 2, InputLayout Left,
                                      // Interval 0 because single stride one or
                                      // second index has a stride.
           (rank <= 2 && R0 &&
            (std::is_same<typename SrcTraits::array_layout,
                          Kokkos::LayoutLeft>::value ||
             std::is_same<typename SrcTraits::array_layout,
                          Kokkos::PartitionedLayoutLeft>::value))  // replace
                                                                   // with input
                                                                   // rank
           ||
           // OutputRank 1 or 2, InputLayout Right, Interval [InputRank-1]
           // because single stride one or second index has a stride.
           (rank <= 2 && R0_rev &&
            (std::is_same<typename SrcTraits::array_layout,
                          Kokkos::LayoutRight>::value ||
             std::is_same<typename SrcTraits::array_layout,
                          Kokkos::PartitionedLayoutRight>::value)  // replace
                                                                   // input rank
            )),
          typename SrcTraits::array_layout, Kokkos::LayoutStride>::type;

  // Check if Kokkos::LayoutStride should become PartitionedLayoutStride
  using array_layout = typename std::conditional<
      std::is_same<array_layout_candidate, Kokkos::LayoutStride>::value &&
          Is_Partitioned_Layout<SrcTraits>::value,
      Kokkos::PartitionedLayoutStride, array_layout_candidate>::type;

  using value_type = typename SrcTraits::value_type;

  using data_type =
      typename SubViewDataType<value_type,
                               typename Kokkos::Impl::ParseViewExtents<
                                   typename SrcTraits::data_type>::type,
                               Args...>::type;

  // If dim0 is range and PartitionedLayout, dim0 is PE
  // We compute the offset to that subview during assign
  enum { require_R0 = R0 };

 public:
  using memory_traits = typename std::conditional<
      require_R0,
      Kokkos::MemoryTraits<
          RemoteSpaces_MemoryTraits<typename SrcTraits::memory_traits>::state |
          RemoteSpaces_MemoryTraitFlags::Dim0IsPE> /*Remove as obsolete*/,
      typename SrcTraits::memory_traits>::type;

  using traits_type =
      Kokkos::ViewTraits<data_type, array_layout,
                         typename SrcTraits::memory_space, memory_traits>;
  using view_type =
      typename Kokkos::View<data_type, array_layout,
                            typename SrcTraits::memory_space, memory_traits>;

  using type = view_type;

  static_assert(
      std::is_same<typename SrcTraits::specialize,
                   Kokkos::Experimental::RemoteSpaceSpecializeTag>::value,

      "Remote memory space copy-construction with incorrect specialization.");

  template <class MemoryTraits>
  struct apply {
    static_assert(Kokkos::is_memory_traits<MemoryTraits>::value, "");

    using traits_type =
        Kokkos::ViewTraits<data_type, array_layout,
                           typename SrcTraits::memory_space, MemoryTraits>;
    using type = Kokkos::View<data_type, array_layout,
                              typename SrcTraits::memory_space, MemoryTraits>;
  };

  template <class DstTraits>
  KOKKOS_INLINE_FUNCTION static void assign(
      ViewMapping<DstTraits, Kokkos::Experimental::RemoteSpaceSpecializeTag>
          &dst,
      ViewMapping<SrcTraits,
                  Kokkos::Experimental::RemoteSpaceSpecializeTag> const &src,
      Args... args) {
    static_assert(
        ViewMapping<
            DstTraits, traits_type,
            Kokkos::Experimental::RemoteSpaceSpecializeTag>::is_assignable,
        "Subview destination type must be compatible with subview "
        "derived type");

    using SrcType =
        ViewMapping<SrcTraits, Kokkos::Experimental::RemoteSpaceSpecializeTag>;
    using DstType =
        ViewMapping<DstTraits, Kokkos::Experimental::RemoteSpaceSpecializeTag>;
    using dst_offset_type = typename DstType::offset_type;

    const SubviewExtents<SrcTraits::rank, rank> extents(src.m_offset.m_dim,
                                                        args...);
    dst.m_offset                  = dst_offset_type(src.m_offset, extents);
    dst.remote_view_props         = src.remote_view_props;
    bool switch_to_local_indexing = false;

    /*We currently support only subviews of subviews where the first subview is
     created with a scalar over the leading dim*/
    /*Subviews that span across multiple nodes cannot have subviews in this
     * version
     */
    if (!src.remote_view_props.using_local_indexing) {
      dst.remote_view_props.using_local_indexing = !R0 ? true : false;
      dst.remote_view_props.R0_offset            = extents.domain_offset(0);
    } else
      switch_to_local_indexing = true;

    typename view_type::size_type offset;
    offset =
        switch_to_local_indexing
            ? src.m_offset(extents.domain_offset(0), extents.domain_offset(1),
                           extents.domain_offset(2), extents.domain_offset(3),
                           extents.domain_offset(4), extents.domain_offset(5),
                           extents.domain_offset(6), extents.domain_offset(7))
            : src.m_offset(
                  0 /*Global indexing uses R0_offset for this dim offset*/,
                  extents.domain_offset(1), extents.domain_offset(2),
                  extents.domain_offset(3), extents.domain_offset(4),
                  extents.domain_offset(5), extents.domain_offset(6),
                  extents.domain_offset(7));

#ifdef KRS_ENABLE_MPISPACE
    // Subviews propagate MPI_Window of the original view
    dst.m_handle = ViewDataHandle<DstTraits>::assign(
        src.m_handle, src.m_handle.loc.win, offset);
#else
    dst.m_handle = ViewDataHandle<DstTraits>::assign(src.m_handle, offset);
#endif
  }
};

/*
 * ViewMapping class used by View specialization
 */

template <class Traits>
class ViewMapping<Traits, Kokkos::Experimental::RemoteSpaceSpecializeTag> {
 private:
  template <class, class...>
  friend class ViewMapping;
  template <class, class...>
  friend class Kokkos::View;

  using layout = typename Traits::array_layout;

  typedef typename ViewDataHandle<Traits>::handle_type handle_type;
  typedef typename ViewDataHandle<Traits>::return_type reference_type;
  typedef typename Traits::value_type *pointer_type;

  // Add here a std::conditional to differentiate between a subview remote
  // offset. A subview ViewOffset can handle dim0 accesses correctly (offsets)
  typedef ViewOffset<typename Traits::dimension, typename Traits::array_layout,
                     void>
      offset_type;
  RemoteSpaces_View_Properties<typename Traits::size_type> remote_view_props;

 public:
  offset_type m_offset;
  handle_type m_handle;

  typedef void printable_label_typedef;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

  KOKKOS_INLINE_FUNCTION
  int get_PE() const { return remote_view_props.my_PE; }

  template <typename T = Traits>
  KOKKOS_INLINE_FUNCTION int get_logical_PE(ENABLE_IF_GLOBAL_LAYOUT) const {
    // If View is subview, compute owning PE of index R0_offset
    if (USING_GLOBAL_INDEXING && remote_view_props.R0_offset != 0)
      return compute_dim0_offsets(remote_view_props.R0_offset).PE;
    // Else, return my_PE
    return remote_view_props.my_PE;
  }

  template <typename T = Traits>
  KOKKOS_INLINE_FUNCTION int get_logical_PE(
      ENABLE_IF_PARTITIONED_LAYOUT) const {
    if (USING_GLOBAL_INDEXING && remote_view_props.R0_offset != 0)
      return remote_view_props.R0_offset;
    return remote_view_props.my_PE;
  }

  template <typename iType, typename T = Traits>
  KOKKOS_INLINE_FUNCTION constexpr size_t extent(const iType &r) const {
    if (r == 0) return dimension_0();
    return m_offset.m_dim.extent(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr typename Traits::array_layout get_layout()
      const {
    return m_offset.layout();
  }

  template <typename T = Traits>
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0(
      ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_GLOBAL_INDEXING)
      return remote_view_props.R0_size;
    else
      return m_offset.dimension_0();
  }

  template <typename T = Traits>
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0(
      ENABLE_IF_PARTITIONED_LAYOUT) const {
    return m_offset.dimension_0();
  }

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() const {
    return m_offset.dimension_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() const {
    return m_offset.dimension_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() const {
    return m_offset.dimension_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() const {
    return m_offset.dimension_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() const {
    return m_offset.dimension_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() const {
    return m_offset.dimension_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() const {
    return m_offset.dimension_7();
  }

  // Is a regular layout with uniform striding for each index.
  using is_regular = typename offset_type::is_regular;

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const {
    return m_offset.stride_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const {
    return m_offset.stride_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const {
    return m_offset.stride_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const {
    return m_offset.stride_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const {
    return m_offset.stride_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const {
    return m_offset.stride_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const {
    return m_offset.stride_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const {
    return m_offset.stride_7();
  }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    m_offset.stride(s);
  }

  //----------------------------------------
  // Range span

  /** \brief  Span of the mapped range */
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const {
    return m_offset.span();
  }

  /** \brief  Is the mapped range span contiguous */
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return m_offset.span_is_contiguous();
  }

  /** \brief  Query raw pointer to memory */
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const {
    return m_handle.ptr;
  }

  /** \brief  Query raw pointer to memory */
  KOKKOS_INLINE_FUNCTION handle_type handle() const { return m_handle; }

  //----------------------------------------
  // The View class performs all rank and bounds checking before
  // calling these element reference methods.

  KOKKOS_INLINE_FUNCTION
  reference_type reference() const { return m_handle[0]; }

  // PartitionedLayout{Left,Right,Strided} access operators

  template <typename I0, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, ENABLE_IF_PARTITIONED_LAYOUT) const {
    // We need this dynamic check as we do not derive the
    // type specialization at view construction through the
    // view ctr (only through Kokkos::subview(...)). This adds support
    // for auto sub_v = View_t(old_sub_v,...).
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element = m_handle(dim0_offset, m_offset(_i0));
    return element;
  }

  template <typename I0, typename I1, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element = m_handle(dim0_offset, m_offset(_i0, i1));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2,
            ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element = m_handle(dim0_offset, m_offset(_i0, i1, i2));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element =
        m_handle(dim0_offset, m_offset(_i0, i1, i2, i3));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element =
        m_handle(dim0_offset, m_offset(_i0, i1, i2, i3, i4));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element =
        m_handle(dim0_offset, m_offset(_i0, i1, i2, i3, i4, i5));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3, const I4 &i4,
      const I5 &i5, const I6 &i6, ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element =
        m_handle(dim0_offset, m_offset(_i0, i1, i2, i3, i4, i5, i6));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7,
            ENABLE_IF_PARTITIONED_LAYOUT) const {
    auto dim0_offset = remote_view_props.R0_offset;
    auto _i0         = i0;

    if (USING_GLOBAL_INDEXING) {
      dim0_offset += i0;
      _i0 = 0;
    }
    const reference_type element =
        m_handle(dim0_offset, m_offset(_i0, i1, i2, i3, i4, i5, i6, i7));
    return element;
  }

  //----------------------------------------
  // Layout{Left,Right,Stride} access operators
  // Implements global views

  template <class T>
  struct Dim0_IndexOffset {
    int PE;
    T offset;
  };

  template <typename I0>
  KOKKOS_INLINE_FUNCTION Dim0_IndexOffset<I0> compute_dim0_offsets(
      const I0 &_i0) const {
    assert(remote_view_props.R0_size);
    auto local_size = static_cast<I0>(remote_view_props.R0_size);
    auto target_pe  = static_cast<int>(_i0 / local_size);
    auto dim0_mod   = static_cast<I0>(_i0 % local_size);
    return {target_pe, dim0_mod};
  }

  template <typename I0, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element = m_handle(0, m_offset(i0));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element    = m_handle(new_offset.PE, m_offset(i0));
      return element;
    }
    if (remote_view_props.num_PEs <= 1) {
      const reference_type element = m_handle(0, m_offset(i0));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element =
        m_handle(new_offset.PE, m_offset(new_offset.offset));
    return element;
  }

  template <typename I0, typename I1, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element = m_handle(0, m_offset(i0, i1));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element = m_handle(new_offset.PE, m_offset(i0, i1));
      return element;
    }
    if (remote_view_props.num_PEs <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element =
        m_handle(new_offset.PE, m_offset(new_offset.offset, i1));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1, const I2 &i2, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element = m_handle(0, m_offset(i0, i1, i2));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element =
          m_handle(new_offset.PE, m_offset(i0, i1, i2));
      return element;
    }

    if (remote_view_props.num_PEs <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1, i2));
      return element;
    }
    auto dim0_offset = remote_view_props.R0_offset + i0;

    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element =
        m_handle(new_offset.PE, m_offset(new_offset.offset, i1, i2));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element = m_handle(0, m_offset(i0, i1, i2, i3));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element =
          m_handle(new_offset.PE, m_offset(i0, i1, i2, i3));
      return element;
    }

    if (remote_view_props.num_PEs <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1, i2, i3));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element =
        m_handle(new_offset.PE, m_offset(new_offset.offset, i1, i2, i3));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element =
            m_handle(0, m_offset(i0, i1, i2, i3, i4));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element =
          m_handle(new_offset.PE, m_offset(i0, i1, i2, i3, i4));
      return element;
    }

    if (remote_view_props.num_PEs <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1, i2, i3, i4));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element =
        m_handle(new_offset.PE, m_offset(new_offset.offset, i1, i2, i3, i4));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element =
            m_handle(0, m_offset(i0, i1, i2, i3, i4, i5));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element =
          m_handle(new_offset.PE, m_offset(i0, i1, i2, i3, i4, i5));
      return element;
    }

    if (remote_view_props.num_PEs <= 1) {
      const reference_type element =
          m_handle(0, m_offset(i0, i1, i2, i3, i4, i5));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element    = m_handle(
        new_offset.PE, m_offset(new_offset.offset, i1, i2, i3, i4, i5));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3, const I4 &i4,
      const I5 &i5, const I6 &i6, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element =
            m_handle(0, m_offset(i0, i1, i2, i3, i4, i5, i6));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element =
          m_handle(new_offset.PE, m_offset(i0, i1, i2, i3, i4, i5, i6));
      return element;
    }
    if (remote_view_props.num_PEs <= 1) {
      const reference_type element =
          m_handle(0, m_offset(i0, i1, i2, i3, i4, i5, i6));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element    = m_handle(
        new_offset.PE, m_offset(new_offset.offset, i1, i2, i3, i4, i5, i6));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3, const I4 &i4,
      const I5 &i5, const I6 &i6, const I7 &i7, ENABLE_IF_GLOBAL_LAYOUT) const {
    if (USING_LOCAL_INDEXING) {
      if (remote_view_props.num_PEs <= 1) {
        const reference_type element =
            m_handle(0, m_offset(i0, i1, i2, i3, i4, i5, i6, i7));
        return element;
      }
      auto dim0_offset                = remote_view_props.R0_offset;
      Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
      const reference_type element =
          m_handle(new_offset.PE, m_offset(i0, i1, i2, i3, i4, i5, i6, i7));
      return element;
    }
    if (remote_view_props.num_PEs <= 1) {
      const reference_type element =
          m_handle(0, m_offset(i0, i1, i2, i3, i4, i5, i6, i7));
      return element;
    }
    auto dim0_offset                = remote_view_props.R0_offset + i0;
    Dim0_IndexOffset<I0> new_offset = compute_dim0_offsets<I0>(dim0_offset);
    const reference_type element    = m_handle(
        new_offset.PE, m_offset(new_offset.offset, i1, i2, i3, i4, i5, i6, i7));
    return element;
  }

 private:
  enum { MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
  enum { MemorySpanSize = sizeof(typename Traits::value_type) };

 public:
  /** \brief  Span, in bytes, of the referenced memory */
  KOKKOS_INLINE_FUNCTION constexpr size_t memory_span() const {
    return (m_offset.span() * sizeof(typename Traits::value_type) +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  /**\brief  Span, in bytes, of the required memory */
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t memory_span(
      typename Traits::array_layout const &arg_layout) {
    typedef std::integral_constant<unsigned, 0> padding;
    return (offset_type(padding(), arg_layout).span() * MemorySpanSize +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  KOKKOS_INLINE_FUNCTION ~ViewMapping() {}
  KOKKOS_INLINE_FUNCTION ViewMapping()
      : m_handle(), m_offset(), remote_view_props() {}

  KOKKOS_INLINE_FUNCTION ViewMapping(const ViewMapping &rhs)
      : m_handle(rhs.m_handle),
        m_offset(rhs.m_offset),
        remote_view_props(rhs.remote_view_props) {}

  KOKKOS_INLINE_FUNCTION ViewMapping &operator=(const ViewMapping &rhs) {
    m_handle          = rhs.m_handle;
    m_offset          = rhs.m_offset;
    remote_view_props = rhs.remote_view_props;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION ViewMapping(ViewMapping &&rhs)
      : m_handle(rhs.m_handle),
        m_offset(rhs.m_offset),
        remote_view_props(rhs.remote_view_props) {}

  KOKKOS_INLINE_FUNCTION ViewMapping &operator=(ViewMapping &&rhs) {
    m_handle          = rhs.m_handle;
    m_offset          = rhs.m_offset;
    remote_view_props = rhs.remote_view_props;
    return *this;
  }

  /**\brief  Wrap a span of memory */
  template <class... P>
  KOKKOS_INLINE_FUNCTION ViewMapping(
      Kokkos::Impl::ViewCtorProp<P...> const &arg_prop,
      typename Traits::array_layout const &arg_layout)
      : remote_view_props(),
        m_handle(
            ((Kokkos::Impl::ViewCtorProp<void, pointer_type> const &)arg_prop)
                .value)

  {
    typedef typename Traits::value_type value_type;
    typedef std::integral_constant<
        unsigned, Kokkos::Impl::ViewCtorProp<P...>::allow_padding
                      ? sizeof(value_type)
                      : 0>
        padding;

    typename Traits::array_layout layout;

    // Copy layout properties
    set_layout(arg_layout, layout, remote_view_props);
    m_offset = offset_type(padding(), layout);
  }

  /**\brief  Assign data */
  KOKKOS_FUNCTION
  void assign_data(pointer_type arg_ptr) { m_handle = handle_type(arg_ptr); }

 private:
  template <typename T = Traits>
  KOKKOS_FUNCTION typename std::enable_if_t<!Is_Partitioned_Layout<T>::value>
  set_layout(typename T::array_layout const &arg_layout,
             typename T::array_layout &layout,
             RemoteSpaces_View_Properties<typename T::size_type> &view_props) {
    for (int i = 0; i < T::rank; i++)
      layout.dimension[i] = arg_layout.dimension[i];
    view_props.R0_size =
        Kokkos::Experimental::get_indexing_block_size(arg_layout.dimension[0]);
    layout.dimension[0] = view_props.R0_size;
  }

  template <typename T = Traits>
  KOKKOS_FUNCTION typename std::enable_if_t<Is_Partitioned_Layout<T>::value>
  set_layout(typename T::array_layout const &arg_layout,
             typename T::array_layout &layout,
             RemoteSpaces_View_Properties<typename T::size_type> &view_props) {
    for (int i = 0; i < T::rank; i++)
      layout.dimension[i] = arg_layout.dimension[i];
    layout.dimension[0] = 1;
    view_props.R0_size  = 0;
  }

 public:
  //----------------------------------------
  /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */
  template <class... P, typename T = Traits>
  Kokkos::Impl::SharedAllocationRecord<> *allocate_shared(
      Kokkos::Impl::ViewCtorProp<P...> const &arg_prop,
      typename Traits::array_layout const &arg_layout,
      bool execution_space_specified) {
    using alloc_prop = Kokkos::Impl::ViewCtorProp<P...>;

    using execution_space = typename alloc_prop::execution_space;
    using memory_space    = typename Traits::memory_space;
    static_assert(
        SpaceAccessibility<execution_space, memory_space>::accessible);
    using value_type = typename Traits::value_type;
    using functor_type =
        ViewValueFunctor<Kokkos::Device<execution_space, memory_space>,
                         value_type>;
    using record_type =
        Kokkos::Impl::SharedAllocationRecord<memory_space, functor_type>;

    // Copy layout properties
    typename T::array_layout layout;
    set_layout(arg_layout, layout, remote_view_props);

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    typedef std::integral_constant<
        unsigned, alloc_prop::allow_padding ? sizeof(value_type) : 0>
        padding;

    m_offset = offset_type(padding(), layout);

    const size_t alloc_size =
        (m_offset.span() * MemorySpanSize + MemorySpanMask) &
        ~size_t(MemorySpanMask);
    const std::string &alloc_name =
        Impl::get_property<Impl::LabelTag>(arg_prop);
    const execution_space &exec_space =
        Impl::get_property<Impl::ExecutionSpaceTag>(arg_prop);
    const memory_space &mem_space =
        Impl::get_property<Impl::MemorySpaceTag>(arg_prop);

    // Create shared memory tracking record with allocate memory from the
    // memory space
    record_type *const record = record_type::allocate(
        ((Kokkos::Impl::ViewCtorProp<void, memory_space> const &)arg_prop)
            .value,
        ((Kokkos::Impl::ViewCtorProp<void, std::string> const &)arg_prop).value,
        alloc_size);

#ifdef KRS_ENABLE_MPISPACE
    if (alloc_size) {
      m_handle = handle_type(reinterpret_cast<pointer_type>(record->data()),
                             record->win);
    }
#else
    if (alloc_size) {
      m_handle = handle_type(reinterpret_cast<pointer_type>(record->data()));
    }
#endif

    functor_type functor =
        execution_space_specified
            ? functor_type(exec_space, (value_type *)m_handle.ptr,
                           m_offset.span(), alloc_name)
            : functor_type((value_type *)m_handle.ptr, m_offset.span(),
                           alloc_name);

    //  Only initialize if the allocation is non-zero.
    //  May be zero if one of the dimensions is zero.
    if constexpr (alloc_prop::initialize)
      if (alloc_size) {
        // Assume destruction is only required when construction is requested.
        // The ViewValueFunctor has both value construction and destruction
        // operators.
        record->m_destroy = std::move(functor);
        // Construct values
        record->m_destroy.construct_shared_allocation();
      }
    return record;
  }
};  // namespace Impl

template <class DstTraits, class SrcTraits>
class ViewMapping<DstTraits, SrcTraits,
                  Kokkos::Experimental::RemoteSpaceSpecializeTag> {
 private:
  enum {
    is_assignable_space = Kokkos::Impl::MemorySpaceAccess<
        typename DstTraits::memory_space,
        typename SrcTraits::memory_space>::assignable
  };

  enum {
    is_assignable_value_type =
        std::is_same<typename DstTraits::value_type,
                     typename SrcTraits::value_type>::value ||
        std::is_same<typename DstTraits::value_type,
                     typename SrcTraits::const_value_type>::value
  };

  enum {
    is_assignable_dimension =
        ViewDimensionAssignable<typename DstTraits::dimension,
                                typename SrcTraits::dimension>::value
  };

 public:
  enum {
    is_assignable_data_type =
        is_assignable_value_type && is_assignable_dimension
  };
  enum {
    is_assignable = is_assignable_space && is_assignable_value_type &&
                    is_assignable_dimension
  };

  using TrackType = Kokkos::Impl::SharedAllocationTracker;
  using DstType =
      ViewMapping<DstTraits, Kokkos::Experimental::RemoteSpaceSpecializeTag>;
  using SrcType =
      ViewMapping<SrcTraits, Kokkos::Experimental::RemoteSpaceSpecializeTag>;

  KOKKOS_INLINE_FUNCTION
  static bool assignable_layout_check(DstType &,
                                      const SrcType &src)  // Runtime check
  {
    size_t strides[9];
    bool assignable = true;
    src.stride(strides);
    size_t exp_stride = 1;
    if (std::is_same<typename DstTraits::array_layout,
                     Kokkos::LayoutLeft>::value) {
      for (int i = 0; i < src.Rank; i++) {
        if (i > 0) exp_stride *= src.extent(i - 1);
        if (strides[i] != exp_stride) {
          assignable = false;
          break;
        }
      }
    } else if (std::is_same<typename DstTraits::array_layout,
                            Kokkos::LayoutRight>::value) {
      for (int i = src.Rank - 1; i >= 0; i--) {
        if (i < src.Rank - 1) exp_stride *= src.extent(i + 1);
        if (strides[i] != exp_stride) {
          assignable = false;
          break;
        }
      }
    }
    return assignable;
  }

  KOKKOS_INLINE_FUNCTION
  static void assign(DstType &dst, const SrcType &src,
                     const TrackType &src_track) {
    static_assert(is_assignable_space,
                  "View assignment must have compatible spaces");

    static_assert(
        is_assignable_value_type,
        "View assignment must have same value type or const = non-const");

    static_assert(is_assignable_dimension,
                  "View assignment must have compatible dimensions");

    bool assignable_layout = assignable_layout_check(dst, src);  // Runtime
                                                                 // check
    if (!assignable_layout)
      Kokkos::abort("View assignment must have compatible layouts\n");

    using dst_offset_type = typename DstType::offset_type;

    if (size_t(DstTraits::dimension::rank_dynamic) <
        size_t(SrcTraits::dimension::rank_dynamic)) {
      using dst_dim   = typename DstTraits::dimension;
      bool assignable = ((1 > DstTraits::dimension::rank_dynamic &&
                          1 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN0 == src.dimension_0()
                             : true) &&
                        ((2 > DstTraits::dimension::rank_dynamic &&
                          2 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN1 == src.dimension_1()
                             : true) &&
                        ((3 > DstTraits::dimension::rank_dynamic &&
                          3 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN2 == src.dimension_2()
                             : true) &&
                        ((4 > DstTraits::dimension::rank_dynamic &&
                          4 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN3 == src.dimension_3()
                             : true) &&
                        ((5 > DstTraits::dimension::rank_dynamic &&
                          5 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN4 == src.dimension_4()
                             : true) &&
                        ((6 > DstTraits::dimension::rank_dynamic &&
                          6 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN5 == src.dimension_5()
                             : true) &&
                        ((7 > DstTraits::dimension::rank_dynamic &&
                          7 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN6 == src.dimension_6()
                             : true) &&
                        ((8 > DstTraits::dimension::rank_dynamic &&
                          8 <= SrcTraits::dimension::rank_dynamic)
                             ? dst_dim::ArgN7 == src.dimension_7()
                             : true);
      if (!assignable)
        Kokkos::abort(
            "View Assignment: trying to assign runtime dimension to non "
            "matching compile time dimension.");
    }
    dst.m_offset = dst_offset_type(src.m_offset);
    dst.m_handle = Kokkos::Impl::ViewDataHandle<DstTraits>::assign(src.m_handle,
                                                                   src_track);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#undef ENABLE_IF_GLOBAL_LAYOUT
#undef ENABLE_IF_PARTITIONED_LAYOUT

#endif  // KOKKOS_REMOTESPACES_VIEWMAPPING_HPP
