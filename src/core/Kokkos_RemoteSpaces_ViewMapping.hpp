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

#ifndef KOKKOS_REMOTESPACES_VIEWMAPPING_HPP
#define KOKKOS_REMOTESPACES_VIEWMAPPING_HPP

#include <type_traits>

//----------------------------------------------------------------------------
/** \brief  View mapping for non-specialized data type and standard layout */

namespace Kokkos {

namespace Experimental {

extern size_t get_my_pe();

template <typename T>
std::pair<size_t, size_t> get_range(
    T &v, size_t pe,
    typename std::enable_if<!std::is_integral<T>::value>::type * = nullptr) {
  static_assert(!(std::is_same<typename T::traits::array_layout,
                               Kokkos::PartitionedLayoutRight>::value ||
                  std::is_same<typename T::traits::array_layout,
                               Kokkos::PartitionedLayoutLeft>::value ||
                  std::is_same<typename T::traits::array_layout,
                               Kokkos::PartitionedLayoutStride>::value),
                "get_local_range over partitioned layouts are not allowed");

  // JC: Error out also in this case as we need to access the original dim0 of
  // the View and not the rounded dim0 of the View. Fix would need to add
  // get_mapping to View
  static_assert((std::is_same<typename T::traits::array_layout,
                              Kokkos::PartitionedLayoutRight>::value ||
                 std::is_same<typename T::traits::array_layout,
                              Kokkos::PartitionedLayoutLeft>::value ||
                 std::is_same<typename T::traits::array_layout,
                              Kokkos::PartitionedLayoutStride>::value),
                "get_local_range overload currently unsupported");

  size_t extent_dim0 = v.extent(0);
  return getRange(extent_dim0, pe);
}

template <typename T>
std::pair<size_t, size_t> get_local_range(
    T &v,
    typename std::enable_if<!std::is_integral<T>::value>::type * = nullptr) {
  static_assert(!(std::is_same<typename T::traits::array_layout,
                               Kokkos::PartitionedLayoutRight>::value ||
                  std::is_same<typename T::traits::array_layout,
                               Kokkos::PartitionedLayoutLeft>::value ||
                  std::is_same<typename T::traits::array_layout,
                               Kokkos::PartitionedLayoutStride>::value),
                "get_local_range over partitioned layouts are not allowed");

  // JC: Error out also in this case as we need to access the original dim0 of
  // the View and not the rounded dim0 of the View. Fix would need to add
  // get_mapping to View
  static_assert((std::is_same<typename T::traits::array_layout,
                              Kokkos::PartitionedLayoutRight>::value ||
                 std::is_same<typename T::traits::array_layout,
                              Kokkos::PartitionedLayoutLeft>::value ||
                 std::is_same<typename T::traits::array_layout,
                              Kokkos::PartitionedLayoutStride>::value),
                "get_local_range overload currently unsupported");

  size_t pe          = get_my_pe();
  size_t extent_dim0 = v.extent(0);
  return getRange(extent_dim0, pe);
}

template <typename T>
std::pair<size_t, size_t> get_range(
    T size, size_t pe,
    typename std::enable_if<std::is_integral<T>::value>::type * = nullptr) {
  return getRange(size, pe);
}

template <typename T>
std::pair<size_t, size_t> get_local_range(
    T size,
    typename std::enable_if<std::is_integral<T>::value>::type * = nullptr) {
  size_t pe = get_my_pe();
  return getRange(size, pe);
}

}  // namespace Experimental

namespace Impl {

/*
 * ViewMapping class used by View copy-ctr and subview() to specialize new
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
           ||
           SubviewLegalArgsCompileTime<typename SrcTraits::array_layout,
                                       typename SrcTraits::array_layout, rank,
                                       SrcTraits::rank, 0, Args...>::value ||
           // OutputRank 1 or 2, InputLayout Left, Interval 0
           // because single stride one or second index has a stride.
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
          (std::is_same<typename SrcTraits::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename SrcTraits::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename SrcTraits::array_layout,
                        Kokkos::PartitionedLayoutStride>::value),
      Kokkos::PartitionedLayoutStride, array_layout_candidate>::type;

  using value_type = typename SrcTraits::value_type;

  using data_type =
      typename SubViewDataType<value_type,
                               typename Kokkos::Impl::ParseViewExtents<
                                   typename SrcTraits::data_type>::type,
                               Args...>::type;

  // If dim0 is range and PartitionedLayout, dim0 is PE
  // We compute the offset to that subview during assign
  enum { is_required_Dim0IsPE = R0 };

 public:
  using memory_traits = typename std::conditional<
      is_required_Dim0IsPE,
      Kokkos::MemoryTraits<
          RemoteSpaces_MemoryTraits<typename SrcTraits::memory_traits>::state |
          RemoteSpaces_MemoryTraitsFlags::Dim0IsPE>,
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

    using DstType =
        ViewMapping<DstTraits, Kokkos::Experimental::RemoteSpaceSpecializeTag>;
    using dst_offset_type = typename DstType::offset_type;

    const SubviewExtents<SrcTraits::rank, rank> extents(src.m_offset.m_dim,
                                                        args...);

    dst.m_offset     = dst_offset_type(src.m_offset, extents);
    dst.m_local_dim0 = src.m_local_dim0;

    // Set offset for dim0 manually in order to support remote copy-ctr'ed views
    // and subviews
    dst.m_offset_remote_dim = extents.domain_offset(0);
    dst.dim0_is_pe          = R0;

    dst.m_handle = ViewDataHandle<DstTraits>::assign(
        src.m_handle,
        src.m_offset(0, extents.domain_offset(1), extents.domain_offset(2),
                     extents.domain_offset(3), extents.domain_offset(4),
                     extents.domain_offset(5), extents.domain_offset(6),
                     extents.domain_offset(7)));
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

  handle_type m_handle;
  offset_type m_offset;

  size_t m_offset_remote_dim;
  size_t m_local_dim0;

  // We need this dynamic property as we do not derive the
  // type specialization at view construction through the
  // subview ctr. Default is set to 1 as a direct view construction
  // with a partitioned layout always expects dim0 to be rank id
  size_t dim0_is_pe;

  int m_num_pes;
  int pe;

 public:
  typedef void printable_label_typedef;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

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
      typename std::enable_if<
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout,
                       Kokkos::LayoutStride>::value>::type * = nullptr) const {
    return m_local_dim0;
  }

  template <typename T = Traits>
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0(
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    return m_num_pes;
  }

  template <typename T = Traits>
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0(
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    return m_offset.m_dim.extent(0);
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

  //----------------------------------------
  // The View class performs all rank and bounds checking before
  // calling these element reference methods.

  KOKKOS_INLINE_FUNCTION
  reference_type reference() const { return m_handle[0]; }

  //----------------------------------------
  // PartitionedLayout{Left,Right,Strided} access operators where dim0 is PE

  template <typename I0, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0,
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    const reference_type element = m_handle(m_offset_remote_dim + i0, 0);
    return element;
  }

  template <typename I0, typename I1, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1,
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    const reference_type element =
        m_handle(m_offset_remote_dim + i0, m_offset(0, i1));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1, const I2 &i2,
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    const reference_type element =
        m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3) const {
    const reference_type element =
        m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4) const {
    const reference_type element =
        m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5) const {
    const reference_type element =
        m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4, i5));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6) const {
    const reference_type element =
        m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4, i5, i6));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7) const {
    const reference_type element = m_handle(
        m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4, i5, i6, i7));
    return element;
  }

  //----------------------------------------
  // PartitionedLayout{Left,Right,Strided} access operators where dim0 is not PE
  // This occurs on subiew creation
  template <typename I0, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0,
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    // We need this dynamic check as we do not derive the
    // type specialization at view construction through the
    // subview ctr (only through Kokkos::subview(...)). This adds support
    // for auto sub_v = View_t(v,...).

    if (dim0_is_pe) {
      const reference_type element = m_handle(m_offset_remote_dim + i0, 0);
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(i0));
      return element;
    }
  }

  template <typename I0, typename I1, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1,
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    if (dim0_is_pe) {
      const reference_type element =
          m_handle(m_offset_remote_dim + i0, m_offset(0, i1));
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(i0, i1));
      return element;
    }
  }

  template <typename I0, typename I1, typename I2, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0, const I1 &i1, const I2 &i2,
      typename std::enable_if<
          (std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutLeft>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutRight>::value ||
           std::is_same<typename T::array_layout,
                        Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe>::
          type * = nullptr) const {
    if (dim0_is_pe) {
      const reference_type element =
          m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2));
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(0, i1, i2));
      return element;
    }
  }

  template <typename I0, typename I1, typename I2, typename I3,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3) const {
    if (dim0_is_pe) {
      const reference_type element =
          m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3));
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(0, i1, i2, i3));
      return element;
    }
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4) const {
    if (dim0_is_pe) {
      const reference_type element =
          m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4));
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(0, i1, i2, i3, i4));
      return element;
    }
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5) const {
    if (dim0_is_pe) {
      const reference_type element =
          m_handle(m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4, i5));
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(0, i1, i2, i3, i4, i5));
      return element;
    }
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6) const {
    if (dim0_is_pe) {
      const reference_type element = m_handle(
          m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4, i5, i6));
      return element;
    } else {
      const reference_type element =
          m_handle(m_offset_remote_dim, m_offset(0, i1, i2, i3, i4, i5, i6));
      return element;
    }
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value) &&
          !RemoteSpaces_MemoryTraits<typename T::memory_traits>::dim0_is_pe,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7) const {
    if (dim0_is_pe) {
      const reference_type element = m_handle(
          m_offset_remote_dim + i0, m_offset(0, i1, i2, i3, i4, i5, i6, i7));
      return element;
    } else {
      const reference_type element = m_handle(
          m_offset_remote_dim, m_offset(0, i1, i2, i3, i4, i5, i6, i7));
      return element;
    }
  }

  //----------------------------------------
  // Layout{Left,Right,Stride} access operators
  // Implements global views

  struct dim0_offsets {
    size_t pe, offset;
  };

  // TODO: move this to kokkos::view_offset (new template specialization
  // on RemoteSpace space type for all default layouts and also one for
  // all partitioned laytouts. Wait for mdspan.)
  template <typename I0, typename T = Traits>
  KOKKOS_INLINE_FUNCTION dim0_offsets
  compute_dim0_offsets(const I0 &_i0) const {
    size_t target_pe, dim0_mod, i0;
    i0 = static_cast<size_t>(_i0);
    assert(m_local_dim0);
    target_pe = i0 / m_local_dim0;
    dim0_mod  = i0 % m_local_dim0;
    return {target_pe, dim0_mod};
  }

  template <typename I0, typename T = Traits>

  KOKKOS_INLINE_FUNCTION const reference_type reference(
      const I0 &i0,
      typename std::enable_if<
          std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout,
                       Kokkos::LayoutStride>::value>::type * = nullptr) const {
    if (m_num_pes <= 1) {
      const reference_type element = m_handle(0, m_offset(i0));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element =
        m_handle(_dim0_offset.pe, m_offset(_dim0_offset.offset));
    return element;
  }

  template <typename I0, typename I1, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1) const {
    if (m_num_pes <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element =
        m_handle(_dim0_offset.pe, m_offset(_dim0_offset.offset, i1));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value,
      reference_type>::type
  reference(
      const I0 &i0, const I1 &i1, const I2 &i2,
      typename std::enable_if<
          std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout,
                       Kokkos::LayoutRight>::value>::type * = nullptr) const {
    if (m_num_pes <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1, i2));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element =
        m_handle(_dim0_offset.pe, m_offset(_dim0_offset.offset, i1, i2));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3) const {
    if (m_num_pes <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1, i2, i3));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element =
        m_handle(_dim0_offset.pe, m_offset(_dim0_offset.offset, i1, i2, i3));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4) const {
    if (m_num_pes <= 1) {
      const reference_type element = m_handle(0, m_offset(i0, i1, i2, i3, i4));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element = m_handle(
        _dim0_offset.pe, m_offset(_dim0_offset.offset, i1, i2, i3, i4));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5) const {
    if (m_num_pes <= 1) {
      const reference_type element =
          m_handle(0, m_offset(i0, i1, i2, i3, i4, i5));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element = m_handle(
        _dim0_offset.pe, m_offset(_dim0_offset.offset, i1, i2, i3, i4, i5));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6) const {
    if (m_num_pes <= 1) {
      const reference_type element =
          m_handle(0, m_offset(i0, i1, i2, i3, i4, i5, i6));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element = m_handle(
        _dim0_offset.pe, m_offset(_dim0_offset.offset, i1, i2, i3, i4, i5, i6));
    return element;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename T = Traits>
  KOKKOS_INLINE_FUNCTION const typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value ||
          std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value,
      reference_type>::type
  reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
            const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7) const {
    if (m_num_pes <= 1) {
      const reference_type element =
          m_handle(0, m_offset(i0, i1, i2, i3, i4, i5, i6, i7));
      return element;
    }
    dim0_offsets _dim0_offset = compute_dim0_offsets(m_offset_remote_dim + i0);
    const reference_type element =
        m_handle(_dim0_offset.pe,
                 m_offset(_dim0_offset.offset, i1, i2, i3, i4, i5, i6, i7));
    return element;
  }

  //----------------------------------------

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

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION ~ViewMapping() {}
  KOKKOS_INLINE_FUNCTION ViewMapping()
      : m_handle(),
        m_offset(),
        m_offset_remote_dim(0),
        m_local_dim0(0),
        dim0_is_pe(1) {
    m_num_pes = Kokkos::Experimental::get_num_pes();
    pe        = Kokkos::Experimental::get_my_pe();
  }

  KOKKOS_INLINE_FUNCTION ViewMapping(const ViewMapping &rhs)
      : m_handle(rhs.m_handle),
        m_offset(rhs.m_offset),
        m_num_pes(rhs.m_num_pes),
        pe(rhs.pe),
        m_offset_remote_dim(rhs.m_offset_remote_dim),
        m_local_dim0(rhs.m_local_dim0),
        dim0_is_pe(rhs.dim0_is_pe) {}

  KOKKOS_INLINE_FUNCTION ViewMapping &operator=(const ViewMapping &rhs) {
    m_handle            = rhs.m_handle;
    m_offset            = rhs.m_offset;
    m_num_pes           = rhs.m_num_pes;
    m_offset_remote_dim = rhs.m_offset_remote_dim;
    m_local_dim0        = rhs.m_local_dim0;
    dim0_is_pe          = rhs.dim0_is_pe;
    pe                  = rhs.pe;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION ViewMapping(ViewMapping &&rhs)
      : m_handle(rhs.m_handle),
        m_offset(rhs.m_offset),
        m_num_pes(rhs.m_num_pes),
        pe(rhs.pe),
        m_offset_remote_dim(rhs.m_offset_remote_dim),
        m_local_dim0(rhs.m_local_dim0),
        dim0_is_pe(0) {}

  KOKKOS_INLINE_FUNCTION ViewMapping &operator=(ViewMapping &&rhs) {
    m_handle            = rhs.m_handle;
    m_offset            = rhs.m_offset;
    m_num_pes           = rhs.m_num_pes;
    pe                  = rhs.pe;
    m_offset_remote_dim = rhs.m_offset_remote_dim;
    m_local_dim0        = rhs.m_local_dim0;
    dim0_is_pe          = rhs.dim0_is_pe;
    return *this;
  }

  //----------------------------------------

  /**\brief  Wrap a span of memory */
  template <class... P>
  KOKKOS_INLINE_FUNCTION ViewMapping(
      Kokkos::Impl::ViewCtorProp<P...> const &arg_prop,
      typename Traits::array_layout const &arg_layout)
      : m_offset_remote_dim(0),
        m_handle(
            ((Kokkos::Impl::ViewCtorProp<void, pointer_type> const &)arg_prop)
                .value) {
    typedef typename Traits::value_type value_type;
    typedef std::integral_constant<
        unsigned, Kokkos::Impl::ViewCtorProp<P...>::allow_padding
                      ? sizeof(value_type)
                      : 0>
        padding;

    typename Traits::array_layout layout;

    // Copy layout properties
    set_layout(arg_layout, layout, m_local_dim0);

    m_offset  = offset_type(padding(), layout);
    m_num_pes = Kokkos::Experimental::get_num_pes();
    pe        = Kokkos::Experimental::get_my_pe();
  }

  /**\brief  Assign data */
  KOKKOS_FUNCTION
  void assign_data(pointer_type arg_ptr) { m_handle = handle_type(arg_ptr); }

 private:
  template <typename T = Traits>
  KOKKOS_FUNCTION typename std::enable_if<
      std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value ||
      std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
      std::is_same<typename T::array_layout, Kokkos::LayoutStride>::value>::type
  set_layout(typename T::array_layout const &arg_layout,
             typename T::array_layout &layout, size_t &local_dim0) {
    for (int i = 0; i < T::rank; i++)
      layout.dimension[i] = arg_layout.dimension[i];

    local_dim0 =
        Kokkos::Experimental::get_indexing_block_size(arg_layout.dimension[0]);
    // We overallocate potentially in favor of symmetric memory allocation
    layout.dimension[0] = local_dim0;
  }

  template <typename T = Traits>
  KOKKOS_FUNCTION typename std::enable_if<
      (std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutLeft>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutRight>::value ||
       std::is_same<typename T::array_layout,
                    Kokkos::PartitionedLayoutStride>::value)>::type
  set_layout(typename T::array_layout const &arg_layout,
             typename T::array_layout &layout, size_t &local_dim0) {
    for (int i = 0; i < T::rank; i++)
      layout.dimension[i] = arg_layout.dimension[i];

    // Override
    layout.dimension[0] = 1;
    local_dim0          = 0;
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
      typename Traits::array_layout const &arg_layout) {
    typedef Kokkos::Impl::ViewCtorProp<P...> alloc_prop;

    typedef typename alloc_prop::execution_space execution_space;
    typedef typename T::memory_space memory_space;
    typedef typename T::value_type value_type;
    typedef ViewValueFunctor<execution_space, value_type> functor_type;
    typedef Kokkos::Impl::SharedAllocationRecord<memory_space, functor_type>
        record_type;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    typedef std::integral_constant<
        unsigned, alloc_prop::allow_padding ? sizeof(value_type) : 0>
        padding;
    typename T::array_layout layout;

    // Copy layout properties
    set_layout(arg_layout, layout, m_local_dim0);

    m_num_pes = Kokkos::Experimental::get_num_pes();
    pe        = Kokkos::Experimental::get_my_pe();

    m_offset = offset_type(padding(), layout);

    const size_t alloc_size = memory_span();

    // Create shared memory tracking record with allocate memory from the memory
    // space
    record_type *const record = record_type::allocate(
        ((Kokkos::Impl::ViewCtorProp<void, memory_space> const &)arg_prop)
            .value,
        ((Kokkos::Impl::ViewCtorProp<void, std::string> const &)arg_prop).value,
        alloc_size);

#ifdef KOKKOS_ENABLE_MPISPACE
    if (alloc_size) {
      m_handle = handle_type(reinterpret_cast<pointer_type>(record->data()),
                             record->win);
    }
#else
    if (alloc_size) {
      m_handle = handle_type(reinterpret_cast<pointer_type>(record->data()));
    }
#endif

    const std::string &alloc_name =
        static_cast<Kokkos::Impl::ViewCtorProp<void, std::string> const &>(
            arg_prop)
            .value;

    //  Only initialize if the allocation is non-zero.
    //  May be zero if one of the dimensions is zero.
    if (alloc_size && alloc_prop::initialize) {
      // Assume destruction is only required when construction is requested.
      // The ViewValueFunctor has both value construction and destruction
      // operators.
      record->m_destroy = functor_type(
          static_cast<Kokkos::Impl::ViewCtorProp<void, execution_space> const
                          &>(arg_prop)
              .value,
          (value_type *)m_handle.ptr, m_offset.span(), alloc_name);
    }

    return record;
  }
};

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

  enum {
    is_assignable_layout =
        std::is_same<typename DstTraits::array_layout,
                     typename SrcTraits::array_layout>::value ||
        std::is_same<typename DstTraits::array_layout,
                     Kokkos::LayoutStride>::value ||
        (DstTraits::dimension::rank == 0) ||
        (DstTraits::dimension::rank == 1 &&
         DstTraits::dimension::rank_dynamic == 1)
  };

 public:
  enum {
    is_assignable_data_type =
        is_assignable_value_type && is_assignable_dimension
  };
  enum {
    is_assignable = is_assignable_space && is_assignable_value_type &&
                    is_assignable_layout && is_assignable_dimension
  };
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_VIEWMAPPING_HPP
