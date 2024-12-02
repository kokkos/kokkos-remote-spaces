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

#ifndef KOKKOS_REMOTESPACES_VIEWOFFSET_HPP
#define KOKKOS_REMOTESPACES_VIEWOFFSET_HPP

#include <type_traits>

//----------------------------------------------------------------------------
/** \brief  View mapping for non-specialized data type and standard layout */

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
// LayoutLeft AND ( 1 >= rank OR 0 == rank_dynamic ) : no padding / no striding
template <class Dimension>
struct ViewOffset<
    Dimension, Kokkos::PartitionedLayoutLeft,
    typename std::enable_if<(1 >= Dimension::rank ||
                             0 == Dimension::rank_dynamic)>::type> {
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::true_type;

  using size_type      = size_t;
  using dimension_type = Dimension;
  using array_layout   = Kokkos::PartitionedLayoutLeft;

  dimension_type m_dim;

  //----------------------------------------

  // rank 1
  template <typename I0>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
    return i0;
  }

  // rank 2
  template <typename I0, typename I1>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1) const {
    return i0 + m_dim.N0 * i1;
  }

  // rank 3
  template <typename I0, typename I1, typename I2>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2) const {
    return i0 + m_dim.N0 * (i1 + m_dim.N1 * i2);
  }

  // rank 4
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3) const {
    return i0 + m_dim.N0 * (i1 + m_dim.N1 * (i2 + m_dim.N2 * i3));
  }

  // rank 5
  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3,
                                                        I4 const &i4) const {
    return i0 +
           m_dim.N0 * (i1 + m_dim.N1 * (i2 + m_dim.N2 * (i3 + m_dim.N3 * i4)));
  }

  // rank 6
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5) const {
    return i0 +
           m_dim.N0 *
               (i1 +
                m_dim.N1 *
                    (i2 + m_dim.N2 * (i3 + m_dim.N3 * (i4 + m_dim.N4 * i5))));
  }

  // rank 7
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6) const {
    return i0 +
           m_dim.N0 *
               (i1 + m_dim.N1 *
                         (i2 + m_dim.N2 *
                                   (i3 + m_dim.N3 *
                                             (i4 + m_dim.N4 *
                                                       (i5 + m_dim.N5 * i6)))));
  }

  // rank 8
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6, I7 const &i7) const {
    return i0 +
           m_dim.N0 *
               (i1 +
                m_dim.N1 *
                    (i2 + m_dim.N2 *
                              (i3 + m_dim.N3 *
                                        (i4 + m_dim.N4 *
                                                  (i5 + m_dim.N5 *
                                                            (i6 + m_dim.N6 *
                                                                      i7))))));
  }

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const {
    constexpr auto r = dimension_type::rank;
    return array_layout((r > 0 ? m_dim.N0 : KOKKOS_INVALID_INDEX),
                        (r > 1 ? m_dim.N1 : KOKKOS_INVALID_INDEX),
                        (r > 2 ? m_dim.N2 : KOKKOS_INVALID_INDEX),
                        (r > 3 ? m_dim.N3 : KOKKOS_INVALID_INDEX),
                        (r > 4 ? m_dim.N4 : KOKKOS_INVALID_INDEX),
                        (r > 5 ? m_dim.N5 : KOKKOS_INVALID_INDEX),
                        (r > 6 ? m_dim.N6 : KOKKOS_INVALID_INDEX),
                        (r > 7 ? m_dim.N7 : KOKKOS_INVALID_INDEX));
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
    return m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
    return m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
    return m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
    return m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
    return m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
    return m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
    return m_dim.N7;
  }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return true;
  }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return 1; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const {
    return m_dim.N0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const {
    return m_dim.N0 * m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6;
  }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    s[0] = 1;
    if (0 < dimension_type::rank) {
      s[1] = m_dim.N0;
    }
    if (1 < dimension_type::rank) {
      s[2] = s[1] * m_dim.N1;
    }
    if (2 < dimension_type::rank) {
      s[3] = s[2] * m_dim.N2;
    }
    if (3 < dimension_type::rank) {
      s[4] = s[3] * m_dim.N3;
    }
    if (4 < dimension_type::rank) {
      s[5] = s[4] * m_dim.N4;
    }
    if (5 < dimension_type::rank) {
      s[6] = s[5] * m_dim.N5;
    }
    if (6 < dimension_type::rank) {
      s[7] = s[6] * m_dim.N6;
    }
    if (7 < dimension_type::rank) {
      s[8] = s[7] * m_dim.N7;
    }
  }

  //----------------------------------------

  // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
  // correct and errors out during compilation. Same for the other places where
  // I changed this.
#ifdef KOKKOS_IMPL_WINDOWS_CUDA
  KOKKOS_FUNCTION ViewOffset() : m_dim(dimension_type()) {}
  KOKKOS_FUNCTION ViewOffset(const ViewOffset &src) { m_dim = src.m_dim; }
  KOKKOS_FUNCTION ViewOffset &operator=(const ViewOffset &src) {
    m_dim = src.m_dim;
    return *this;
  }
#else
  ViewOffset()                              = default;
  ViewOffset(const ViewOffset &)            = default;
  ViewOffset &operator=(const ViewOffset &) = default;
#endif

  template <unsigned TrivialScalarSize>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      std::integral_constant<unsigned, TrivialScalarSize> const &,
      Kokkos::PartitionedLayoutLeft const &arg_layout)
      : m_dim(arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0) {}

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutLeft, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7) {
    static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                  "ViewOffset assignment requires equal rank");
    // Also requires equal static dimensions ...
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutRight, void> &rhs)
      : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {
    static_assert((DimRHS::rank == 0 && dimension_type::rank == 0) ||
                      (DimRHS::rank == 1 && dimension_type::rank == 1 &&
                       dimension_type::rank_dynamic == 1),
                  "ViewOffset LayoutLeft and LayoutRight are only compatible "
                  "when rank <= 1");
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION ViewOffset(
      const ViewOffset<DimRHS, Kokkos::LayoutStride, void> &rhs)
      : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {
    if (rhs.m_stride.S0 != 1) {
      Kokkos::abort(
          "Kokkos::Impl::ViewOffset assignment of LayoutLeft from LayoutStride "
          " requires stride == 1");
    }
  }
  //----------------------------------------
  // Subview construction

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutLeft, void> &,
      const SubviewExtents<DimRHS::rank, dimension_type::rank> &sub)
      : m_dim(sub.range_extent(0), 0, 0, 0, 0, 0, 0, 0) {
    static_assert((0 == dimension_type::rank_dynamic) ||
                      (1 == dimension_type::rank &&
                       1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank),
                  "ViewOffset subview construction requires compatible rank");
  }
};

//----------------------------------------------------------------------------
// LayoutLeft AND ( 1 < rank AND 0 < rank_dynamic ) : has padding / striding
template <class Dimension>
struct ViewOffset<
    Dimension, Kokkos::PartitionedLayoutLeft,
    typename std::enable_if<(1 < Dimension::rank &&
                             0 < Dimension::rank_dynamic)>::type> {
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::true_type;

  using size_type      = size_t;
  using dimension_type = Dimension;
  using array_layout   = Kokkos::PartitionedLayoutLeft;

  dimension_type m_dim;
  size_type m_stride;

  //----------------------------------------

  // rank 1
  template <typename I0>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
    return i0;
  }

  // rank 2
  template <typename I0, typename I1>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1) const {
    return i0 + m_stride * i1;
  }

  // rank 3
  template <typename I0, typename I1, typename I2>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2) const {
    return i0 + m_stride * (i1 + m_dim.N1 * i2);
  }

  // rank 4
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3) const {
    return i0 + m_stride * (i1 + m_dim.N1 * (i2 + m_dim.N2 * i3));
  }

  // rank 5
  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3,
                                                        I4 const &i4) const {
    return i0 +
           m_stride * (i1 + m_dim.N1 * (i2 + m_dim.N2 * (i3 + m_dim.N3 * i4)));
  }

  // rank 6
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5) const {
    return i0 +
           m_stride *
               (i1 +
                m_dim.N1 *
                    (i2 + m_dim.N2 * (i3 + m_dim.N3 * (i4 + m_dim.N4 * i5))));
  }

  // rank 7
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6) const {
    return i0 +
           m_stride *
               (i1 + m_dim.N1 *
                         (i2 + m_dim.N2 *
                                   (i3 + m_dim.N3 *
                                             (i4 + m_dim.N4 *
                                                       (i5 + m_dim.N5 * i6)))));
  }

  // rank 8
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6, I7 const &i7) const {
    return i0 +
           m_stride *
               (i1 +
                m_dim.N1 *
                    (i2 + m_dim.N2 *
                              (i3 + m_dim.N3 *
                                        (i4 + m_dim.N4 *
                                                  (i5 + m_dim.N5 *
                                                            (i6 + m_dim.N6 *
                                                                      i7))))));
  }

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const {
    constexpr auto r = dimension_type::rank;
    return array_layout((r > 0 ? m_dim.N0 : KOKKOS_INVALID_INDEX),
                        (r > 1 ? m_dim.N1 : KOKKOS_INVALID_INDEX),
                        (r > 2 ? m_dim.N2 : KOKKOS_INVALID_INDEX),
                        (r > 3 ? m_dim.N3 : KOKKOS_INVALID_INDEX),
                        (r > 4 ? m_dim.N4 : KOKKOS_INVALID_INDEX),
                        (r > 5 ? m_dim.N5 : KOKKOS_INVALID_INDEX),
                        (r > 6 ? m_dim.N6 : KOKKOS_INVALID_INDEX),
                        (r > 7 ? m_dim.N7 : KOKKOS_INVALID_INDEX));
  }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
    return m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
    return m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
    return m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
    return m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
    return m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
    return m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
    return m_dim.N7;
  }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const {
    return (m_dim.N0 > size_type(0) ? m_stride : size_type(0)) * m_dim.N1 *
           m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7;
  }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return m_stride == m_dim.N0;
  }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return 1; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const {
    return m_stride;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const {
    return m_stride * m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const {
    return m_stride * m_dim.N1 * m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const {
    return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const {
    return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const {
    return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const {
    return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6;
  }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    s[0] = 1;
    if (0 < dimension_type::rank) {
      s[1] = m_stride;
    }
    if (1 < dimension_type::rank) {
      s[2] = s[1] * m_dim.N1;
    }
    if (2 < dimension_type::rank) {
      s[3] = s[2] * m_dim.N2;
    }
    if (3 < dimension_type::rank) {
      s[4] = s[3] * m_dim.N3;
    }
    if (4 < dimension_type::rank) {
      s[5] = s[4] * m_dim.N4;
    }
    if (5 < dimension_type::rank) {
      s[6] = s[5] * m_dim.N5;
    }
    if (6 < dimension_type::rank) {
      s[7] = s[6] * m_dim.N6;
    }
    if (7 < dimension_type::rank) {
      s[8] = s[7] * m_dim.N7;
    }
  }

  //----------------------------------------

 private:
  template <unsigned TrivialScalarSize>
  struct Padding {
    enum {
      div = TrivialScalarSize == 0
                ? 0
                : Kokkos::Impl::MEMORY_ALIGNMENT /
                      (TrivialScalarSize ? TrivialScalarSize : 1)
    };
    enum {
      mod = TrivialScalarSize == 0
                ? 0
                : Kokkos::Impl::MEMORY_ALIGNMENT %
                      (TrivialScalarSize ? TrivialScalarSize : 1)
    };

    // If memory alignment is a multiple of the trivial scalar size then attempt
    // to align.
    enum { align = 0 != TrivialScalarSize && 0 == mod ? div : 0 };
    enum {
      div_ok = (div != 0) ? div : 1
    };  // To valid modulo zero in constexpr

    KOKKOS_INLINE_FUNCTION
    static constexpr size_t stride(size_t const N) {
      return ((align != 0) &&
              ((static_cast<int>(Kokkos::Impl::MEMORY_ALIGNMENT_THRESHOLD) *
                static_cast<int>(align)) < N) &&
              ((N % div_ok) != 0))
                 ? N + align - (N % div_ok)
                 : N;
    }
  };

 public:
  // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
  // correct and errors out during compilation. Same for the other places where
  // I changed this.
#ifdef KOKKOS_IMPL_WINDOWS_CUDA
  KOKKOS_FUNCTION ViewOffset() : m_dim(dimension_type()), m_stride(0) {}
  KOKKOS_FUNCTION ViewOffset(const ViewOffset &src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
  }
  KOKKOS_FUNCTION ViewOffset &operator=(const ViewOffset &src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
    return *this;
  }
#else

  ViewOffset()                              = default;
  ViewOffset(const ViewOffset &)            = default;
  ViewOffset &operator=(const ViewOffset &) = default;
#endif

  /* Enable padding for trivial scalar types with non-zero trivial scalar size
   */
  template <unsigned TrivialScalarSize>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      std::integral_constant<unsigned, TrivialScalarSize> const &,
      Kokkos::PartitionedLayoutLeft const &arg_layout)
      : m_dim(arg_layout.dimension[0], arg_layout.dimension[1],
              arg_layout.dimension[2], arg_layout.dimension[3],
              arg_layout.dimension[4], arg_layout.dimension[5],
              arg_layout.dimension[6], arg_layout.dimension[7]),
        m_stride(Padding<TrivialScalarSize>::stride(arg_layout.dimension[0])) {}

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutLeft, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
        m_stride(rhs.stride_1()) {
    static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                  "ViewOffset assignment requires equal rank");
    // Also requires equal static dimensions ...
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION ViewOffset(
      const ViewOffset<DimRHS, Kokkos::LayoutStride, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
        m_stride(rhs.stride_1()) {
    if (rhs.m_stride.S0 != 1) {
      Kokkos::abort(
          "Kokkos::Impl::ViewOffset assignment of LayoutLeft from LayoutStride "
          "requires stride == 1");
    }
  }

  //----------------------------------------
  // Subview construction
  // This subview must be 2 == rank and 2 == rank_dynamic
  // due to only having stride #0.
  // The source dimension #0 must be non-zero for stride-one leading dimension.
  // At most subsequent dimension can be non-zero.

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutLeft, void> &rhs,
      const SubviewExtents<DimRHS::rank, dimension_type::rank> &sub)
      : m_dim(sub.range_extent(0), sub.range_extent(1), sub.range_extent(2),
              sub.range_extent(3), sub.range_extent(4), sub.range_extent(5),
              sub.range_extent(6), sub.range_extent(7)),
        m_stride(
            (1 == sub.range_index(1)
                 ? rhs.stride_1()
                 : (2 == sub.range_index(1)
                        ? rhs.stride_2()
                        : (3 == sub.range_index(1)
                               ? rhs.stride_3()
                               : (4 == sub.range_index(1)
                                      ? rhs.stride_4()
                                      : (5 == sub.range_index(1)
                                             ? rhs.stride_5()
                                             : (6 == sub.range_index(1)
                                                    ? rhs.stride_6()
                                                    : (7 == sub.range_index(1)
                                                           ? rhs.stride_7()
                                                           : 0)))))))) {}
};

//----------------------------------------------------------------------------
// LayoutRight AND ( 1 >= rank OR 0 == rank_dynamic ) : no padding / striding
template <class Dimension>
struct ViewOffset<
    Dimension, Kokkos::PartitionedLayoutRight,
    typename std::enable_if<(1 >= Dimension::rank ||
                             0 == Dimension::rank_dynamic)>::type> {
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::true_type;

  using size_type      = size_t;
  using dimension_type = Dimension;
  using array_layout   = Kokkos::PartitionedLayoutRight;

  dimension_type m_dim;

  //----------------------------------------

  // rank 1
  template <typename I0>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
    return i0;
  }

  // rank 2
  template <typename I0, typename I1>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1) const {
    return i1 + m_dim.N1 * i0;
  }

  // rank 3
  template <typename I0, typename I1, typename I2>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2) const {
    return i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0));
  }

  // rank 4
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3) const {
    return i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0)));
  }

  // rank 5
  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3,
                                                        I4 const &i4) const {
    return i4 + m_dim.N4 *
                    (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0))));
  }

  // rank 6
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5) const {
    return i5 +
           m_dim.N5 *
               (i4 +
                m_dim.N4 *
                    (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0)))));
  }

  // rank 7
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6) const {
    return i6 +
           m_dim.N6 *
               (i5 +
                m_dim.N5 *
                    (i4 +
                     m_dim.N4 *
                         (i3 + m_dim.N3 *
                                   (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0))))));
  }

  // rank 8
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6, I7 const &i7) const {
    return i7 +
           m_dim.N7 *
               (i6 +
                m_dim.N6 *
                    (i5 +
                     m_dim.N5 *
                         (i4 +
                          m_dim.N4 *
                              (i3 +
                               m_dim.N3 *
                                   (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0)))))));
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const {
    constexpr auto r = dimension_type::rank;
    return array_layout((r > 0 ? m_dim.N0 : KOKKOS_INVALID_INDEX),
                        (r > 1 ? m_dim.N1 : KOKKOS_INVALID_INDEX),
                        (r > 2 ? m_dim.N2 : KOKKOS_INVALID_INDEX),
                        (r > 3 ? m_dim.N3 : KOKKOS_INVALID_INDEX),
                        (r > 4 ? m_dim.N4 : KOKKOS_INVALID_INDEX),
                        (r > 5 ? m_dim.N5 : KOKKOS_INVALID_INDEX),
                        (r > 6 ? m_dim.N6 : KOKKOS_INVALID_INDEX),
                        (r > 7 ? m_dim.N7 : KOKKOS_INVALID_INDEX));
  }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
    return m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
    return m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
    return m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
    return m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
    return m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
    return m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
    return m_dim.N7;
  }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return true;
  }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return 1; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const {
    return m_dim.N7;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const {
    return m_dim.N7 * m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2 *
           m_dim.N1;
  }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    size_type n = 1;
    if (7 < dimension_type::rank) {
      s[7] = n;
      n *= m_dim.N7;
    }
    if (6 < dimension_type::rank) {
      s[6] = n;
      n *= m_dim.N6;
    }
    if (5 < dimension_type::rank) {
      s[5] = n;
      n *= m_dim.N5;
    }
    if (4 < dimension_type::rank) {
      s[4] = n;
      n *= m_dim.N4;
    }
    if (3 < dimension_type::rank) {
      s[3] = n;
      n *= m_dim.N3;
    }
    if (2 < dimension_type::rank) {
      s[2] = n;
      n *= m_dim.N2;
    }
    if (1 < dimension_type::rank) {
      s[1] = n;
      n *= m_dim.N1;
    }
    if (0 < dimension_type::rank) {
      s[0] = n;
    }
    s[dimension_type::rank] = n * m_dim.N0;
  }

  //----------------------------------------
  // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
  // correct and errors out during compilation. Same for the other places where
  // I changed this.

#ifdef KOKKOS_IMPL_WINDOWS_CUDA
  KOKKOS_FUNCTION ViewOffset() : m_dim(dimension_type()) {}
  KOKKOS_FUNCTION ViewOffset(const ViewOffset &src) { m_dim = src.m_dim; }
  KOKKOS_FUNCTION ViewOffset &operator=(const ViewOffset &src) {
    m_dim = src.m_dim;
    return *this;
  }
#else

  ViewOffset()                              = default;
  ViewOffset(const ViewOffset &)            = default;
  ViewOffset &operator=(const ViewOffset &) = default;
#endif

  template <unsigned TrivialScalarSize>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      std::integral_constant<unsigned, TrivialScalarSize> const &,
      Kokkos::PartitionedLayoutRight const &arg_layout)
      : m_dim(arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0) {}

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutRight, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7) {
    static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                  "ViewOffset assignment requires equal rank");
    // Also requires equal static dimensions ...
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutLeft, void> &rhs)
      : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {
    static_assert((DimRHS::rank == 0 && dimension_type::rank == 0) ||
                      (DimRHS::rank == 1 && dimension_type::rank == 1 &&
                       dimension_type::rank_dynamic == 1),
                  "ViewOffset LayoutRight and LayoutLeft are only compatible "
                  "when rank <= 1");
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION ViewOffset(
      const ViewOffset<DimRHS, Kokkos::LayoutStride, void> &rhs)
      : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {}

  //----------------------------------------
  // Subview construction

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutRight, void> &,
      const SubviewExtents<DimRHS::rank, dimension_type::rank> &sub)
      : m_dim(sub.range_extent(0), 0, 0, 0, 0, 0, 0, 0) {
    static_assert((0 == dimension_type::rank_dynamic) ||
                      (1 == dimension_type::rank &&
                       1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank),
                  "ViewOffset subview construction requires compatible rank");
  }
};

//----------------------------------------------------------------------------
// LayoutRight AND ( 1 < rank AND 0 < rank_dynamic ) : has padding / striding
template <class Dimension>
struct ViewOffset<
    Dimension, Kokkos::PartitionedLayoutRight,
    typename std::enable_if<(1 < Dimension::rank &&
                             0 < Dimension::rank_dynamic)>::type> {
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::true_type;

  using size_type      = size_t;
  using dimension_type = Dimension;
  using array_layout   = Kokkos::PartitionedLayoutRight;

  dimension_type m_dim;
  size_type m_stride;

  //----------------------------------------

  // rank 1
  template <typename I0>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
    return i0;
  }

  // rank 2
  template <typename I0, typename I1>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1) const {
    return i1 + i0 * m_stride;
  }

  // rank 3
  template <typename I0, typename I1, typename I2>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2) const {
    return i2 + m_dim.N2 * (i1) + i0 * m_stride;
  }

  // rank 4
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3) const {
    return i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1)) + i0 * m_stride;
  }

  // rank 5
  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3,
                                                        I4 const &i4) const {
    return i4 + m_dim.N4 * (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1))) +
           i0 * m_stride;
  }

  // rank 6
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5) const {
    return i5 +
           m_dim.N5 *
               (i4 + m_dim.N4 * (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1)))) +
           i0 * m_stride;
  }

  // rank 7
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6) const {
    return i6 +
           m_dim.N6 *
               (i5 + m_dim.N5 *
                         (i4 + m_dim.N4 *
                                   (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1))))) +
           i0 * m_stride;
  }

  // rank 8
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6, I7 const &i7) const {
    return i7 +
           m_dim.N7 *
               (i6 +
                m_dim.N6 *
                    (i5 +
                     m_dim.N5 *
                         (i4 + m_dim.N4 *
                                   (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1)))))) +
           i0 * m_stride;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const {
    constexpr auto r = dimension_type::rank;
    return array_layout((r > 0 ? m_dim.N0 : KOKKOS_INVALID_INDEX),
                        (r > 1 ? m_dim.N1 : KOKKOS_INVALID_INDEX),
                        (r > 2 ? m_dim.N2 : KOKKOS_INVALID_INDEX),
                        (r > 3 ? m_dim.N3 : KOKKOS_INVALID_INDEX),
                        (r > 4 ? m_dim.N4 : KOKKOS_INVALID_INDEX),
                        (r > 5 ? m_dim.N5 : KOKKOS_INVALID_INDEX),
                        (r > 6 ? m_dim.N6 : KOKKOS_INVALID_INDEX),
                        (r > 7 ? m_dim.N7 : KOKKOS_INVALID_INDEX));
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
    return m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
    return m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
    return m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
    return m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
    return m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
    return m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
    return m_dim.N7;
  }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const {
    return size() > 0 ? m_dim.N0 * m_stride : 0;
  }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return m_stride == m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 *
                           m_dim.N2 * m_dim.N1;
  }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return 1; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const {
    return m_dim.N7;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const {
    return m_dim.N7 * m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const {
    return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const {
    return m_stride;
  }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    size_type n = 1;
    if (7 < dimension_type::rank) {
      s[7] = n;
      n *= m_dim.N7;
    }
    if (6 < dimension_type::rank) {
      s[6] = n;
      n *= m_dim.N6;
    }
    if (5 < dimension_type::rank) {
      s[5] = n;
      n *= m_dim.N5;
    }
    if (4 < dimension_type::rank) {
      s[4] = n;
      n *= m_dim.N4;
    }
    if (3 < dimension_type::rank) {
      s[3] = n;
      n *= m_dim.N3;
    }
    if (2 < dimension_type::rank) {
      s[2] = n;
      n *= m_dim.N2;
    }
    if (1 < dimension_type::rank) {
      s[1] = n;
    }
    if (0 < dimension_type::rank) {
      s[0] = m_stride;
    }
    s[dimension_type::rank] = m_stride * m_dim.N0;
  }

  //----------------------------------------

 private:
  template <unsigned TrivialScalarSize>
  struct Padding {
    enum {
      div = TrivialScalarSize == 0
                ? 0
                : Kokkos::Impl::MEMORY_ALIGNMENT /
                      (TrivialScalarSize ? TrivialScalarSize : 1)
    };
    enum {
      mod = TrivialScalarSize == 0
                ? 0
                : Kokkos::Impl::MEMORY_ALIGNMENT %
                      (TrivialScalarSize ? TrivialScalarSize : 1)
    };

    // If memory alignment is a multiple of the trivial scalar size then attempt
    // to align.
    enum { align = 0 != TrivialScalarSize && 0 == mod ? div : 0 };
    enum {
      div_ok = (div != 0) ? div : 1
    };  // To valid modulo zero in constexpr

    KOKKOS_INLINE_FUNCTION
    static constexpr size_t stride(size_t const N) {
      return ((align != 0) &&
              ((static_cast<int>(Kokkos::Impl::MEMORY_ALIGNMENT_THRESHOLD) *
                static_cast<int>(align)) < N) &&
              ((N % div_ok) != 0))
                 ? N + align - (N % div_ok)
                 : N;
    }
  };

 public:
  // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
  // correct and errors out during compilation. Same for the other places where
  // I changed this.

#ifdef KOKKOS_IMPL_WINDOWS_CUDA
  KOKKOS_FUNCTION ViewOffset() : m_dim(dimension_type()), m_stride(0) {}
  KOKKOS_FUNCTION ViewOffset(const ViewOffset &src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
  }
  KOKKOS_FUNCTION ViewOffset &operator=(const ViewOffset &src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
    return *this;
  }
#else

  ViewOffset()                              = default;
  ViewOffset(const ViewOffset &)            = default;
  ViewOffset &operator=(const ViewOffset &) = default;
#endif

  /* Enable padding for trivial scalar types with non-zero trivial scalar size.
   */
  template <unsigned TrivialScalarSize>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      std::integral_constant<unsigned, TrivialScalarSize> const &,
      Kokkos::PartitionedLayoutRight const &arg_layout)
      : m_dim(arg_layout.dimension[0], arg_layout.dimension[1],
              arg_layout.dimension[2], arg_layout.dimension[3],
              arg_layout.dimension[4], arg_layout.dimension[5],
              arg_layout.dimension[6], arg_layout.dimension[7]),
        m_stride(
            Padding<TrivialScalarSize>::
                stride(/* 2 <= rank */
                       m_dim.N1 *
                       (dimension_type::rank == 2
                            ? 1
                            : m_dim.N2 *
                                  (dimension_type::rank == 3
                                       ? 1
                                       : m_dim.N3 *
                                             (dimension_type::rank == 4
                                                  ? 1
                                                  : m_dim.N4 *
                                                        (dimension_type::rank ==
                                                                 5
                                                             ? 1
                                                             : m_dim.N5 *
                                                                   (dimension_type::
                                                                                rank ==
                                                                            6
                                                                        ? 1
                                                                        : m_dim.N6 *
                                                                              (dimension_type::
                                                                                           rank ==
                                                                                       7
                                                                                   ? 1
                                                                                   : m_dim
                                                                                         .N7)))))))) {
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutRight, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
        m_stride(rhs.stride_0()) {
    static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                  "ViewOffset assignment requires equal rank");
    // Also requires equal static dimensions ...
  }

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION ViewOffset(
      const ViewOffset<DimRHS, Kokkos::LayoutStride, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
        m_stride(rhs.stride_0()) {
    if (((dimension_type::rank == 2)
             ? rhs.m_stride.S1
             : ((dimension_type::rank == 3)
                    ? rhs.m_stride.S2
                    : ((dimension_type::rank == 4)
                           ? rhs.m_stride.S3
                           : ((dimension_type::rank == 5)
                                  ? rhs.m_stride.S4
                                  : ((dimension_type::rank == 6)
                                         ? rhs.m_stride.S5
                                         : ((dimension_type::rank == 7)
                                                ? rhs.m_stride.S6
                                                : rhs.m_stride.S7)))))) != 1) {
      Kokkos::abort(
          "Kokkos::Impl::ViewOffset assignment of LayoutRight from "
          "LayoutStride requires right-most stride == 1");
    }
  }

  //----------------------------------------
  // Subview construction
  // Last dimension must be non-zero

  template <class DimRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, Kokkos::PartitionedLayoutRight, void> &rhs,
      const SubviewExtents<DimRHS::rank, dimension_type::rank> &sub)
      : m_dim(sub.range_extent(0), sub.range_extent(1), sub.range_extent(2),
              sub.range_extent(3), sub.range_extent(4), sub.range_extent(5),
              sub.range_extent(6), sub.range_extent(7)),
        m_stride(
            0 == sub.range_index(0)
                ? rhs.stride_0()
                : (1 == sub.range_index(0)
                       ? rhs.stride_1()
                       : (2 == sub.range_index(0)
                              ? rhs.stride_2()
                              : (3 == sub.range_index(0)
                                     ? rhs.stride_3()
                                     : (4 == sub.range_index(0)
                                            ? rhs.stride_4()
                                            : (5 == sub.range_index(0)
                                                   ? rhs.stride_5()
                                                   : (6 == sub.range_index(0)
                                                          ? rhs.stride_6()
                                                          : 0))))))) {}
};

template <class Dimension>
struct ViewOffset<Dimension, Kokkos::PartitionedLayoutStride, void> {
 private:
  using stride_type = ViewStride<Dimension::rank>;

 public:
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::true_type;

  using size_type      = size_t;
  using dimension_type = Dimension;
  using array_layout   = Kokkos::PartitionedLayoutStride;

  dimension_type m_dim;
  stride_type m_stride;

  //----------------------------------------

  // rank 1
  template <typename I0>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
    return i0 * m_stride.S0;
  }

  // rank 2
  template <typename I0, typename I1>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1;
  }

  // rank 3
  template <typename I0, typename I1, typename I2>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2;
  }

  // rank 4
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
           i3 * m_stride.S3;
  }

  // rank 5
  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                        I1 const &i1,
                                                        I2 const &i2,
                                                        I3 const &i3,
                                                        I4 const &i4) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
           i3 * m_stride.S3 + i4 * m_stride.S4;
  }

  // rank 6
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
           i3 * m_stride.S3 + i4 * m_stride.S4 + i5 * m_stride.S5;
  }

  // rank 7
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
           i3 * m_stride.S3 + i4 * m_stride.S4 + i5 * m_stride.S5 +
           i6 * m_stride.S6;
  }

  // rank 8
  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7>
  KOKKOS_INLINE_FUNCTION constexpr size_type operator()(
      I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
      I5 const &i5, I6 const &i6, I7 const &i7) const {
    return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
           i3 * m_stride.S3 + i4 * m_stride.S4 + i5 * m_stride.S5 +
           i6 * m_stride.S6 + i7 * m_stride.S7;
  }

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const {
    constexpr auto r = dimension_type::rank;
    return array_layout((r > 0 ? m_dim.N0 : KOKKOS_INVALID_INDEX),
                        (r > 1 ? m_dim.N1 : KOKKOS_INVALID_INDEX),
                        (r > 2 ? m_dim.N2 : KOKKOS_INVALID_INDEX),
                        (r > 3 ? m_dim.N3 : KOKKOS_INVALID_INDEX),
                        (r > 4 ? m_dim.N4 : KOKKOS_INVALID_INDEX),
                        (r > 5 ? m_dim.N5 : KOKKOS_INVALID_INDEX),
                        (r > 6 ? m_dim.N6 : KOKKOS_INVALID_INDEX),
                        (r > 7 ? m_dim.N7 : KOKKOS_INVALID_INDEX));
  }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
    return m_dim.N1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
    return m_dim.N2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
    return m_dim.N3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
    return m_dim.N4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
    return m_dim.N5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
    return m_dim.N6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
    return m_dim.N7;
  }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const {
    return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
           m_dim.N6 * m_dim.N7;
  }

 private:
  KOKKOS_INLINE_FUNCTION
  static constexpr size_type Max(size_type lhs, size_type rhs) {
    return lhs < rhs ? rhs : lhs;
  }

 public:
  /* Span of the range space, largest stride * dimension */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const {
    return size() == size_type(0)
               ? size_type(0)
               : Max(m_dim.N0 * m_stride.S0,
                     Max(m_dim.N1 * m_stride.S1,
                         Max(m_dim.N2 * m_stride.S2,
                             Max(m_dim.N3 * m_stride.S3,
                                 Max(m_dim.N4 * m_stride.S4,
                                     Max(m_dim.N5 * m_stride.S5,
                                         Max(m_dim.N6 * m_stride.S6,
                                             m_dim.N7 * m_stride.S7)))))));
  }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return span() == size();
  }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const {
    return m_stride.S0;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const {
    return m_stride.S1;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const {
    return m_stride.S2;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const {
    return m_stride.S3;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const {
    return m_stride.S4;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const {
    return m_stride.S5;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const {
    return m_stride.S6;
  }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const {
    return m_stride.S7;
  }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType *const s) const {
    if (0 < dimension_type::rank) {
      s[0] = m_stride.S0;
    }
    if (1 < dimension_type::rank) {
      s[1] = m_stride.S1;
    }
    if (2 < dimension_type::rank) {
      s[2] = m_stride.S2;
    }
    if (3 < dimension_type::rank) {
      s[3] = m_stride.S3;
    }
    if (4 < dimension_type::rank) {
      s[4] = m_stride.S4;
    }
    if (5 < dimension_type::rank) {
      s[5] = m_stride.S5;
    }
    if (6 < dimension_type::rank) {
      s[6] = m_stride.S6;
    }
    if (7 < dimension_type::rank) {
      s[7] = m_stride.S7;
    }
    s[dimension_type::rank] = span();
  }

  //----------------------------------------
  // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
  // correct and errors out during compilation. Same for the other places where
  // I changed this.

#ifdef KOKKOS_IMPL_WINDOWS_CUDA
  KOKKOS_FUNCTION ViewOffset()
      : m_dim(dimension_type()), m_stride(stride_type()) {}
  KOKKOS_FUNCTION ViewOffset(const ViewOffset &src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
  }
  KOKKOS_FUNCTION ViewOffset &operator=(const ViewOffset &src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
    return *this;
  }
#else

  ViewOffset()                              = default;
  ViewOffset(const ViewOffset &)            = default;
  ViewOffset &operator=(const ViewOffset &) = default;
#endif

  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset(std::integral_constant<unsigned, 0> const &,
                       Kokkos::PartitionedLayoutStride const &rhs)
      : m_dim(rhs.dimension[0], rhs.dimension[1], rhs.dimension[2],
              rhs.dimension[3], rhs.dimension[4], rhs.dimension[5],
              rhs.dimension[6], rhs.dimension[7]),
        m_stride(rhs.stride[0], rhs.stride[1], rhs.stride[2], rhs.stride[3],
                 rhs.stride[4], rhs.stride[5], rhs.stride[6], rhs.stride[7]) {}

  template <class DimRHS, class LayoutRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, LayoutRHS, void> &rhs)
      : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
              rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
        m_stride(rhs.stride_0(), rhs.stride_1(), rhs.stride_2(), rhs.stride_3(),
                 rhs.stride_4(), rhs.stride_5(), rhs.stride_6(),
                 rhs.stride_7()) {
    static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                  "ViewOffset assignment requires equal rank");
    // Also requires equal static dimensions ...
  }

  //----------------------------------------
  // Subview construction

 private:
  template <class DimRHS, class LayoutRHS>
  KOKKOS_INLINE_FUNCTION static constexpr size_t stride(
      unsigned r, const ViewOffset<DimRHS, LayoutRHS, void> &rhs) {
    return r > 7
               ? 0
               : (r == 0
                      ? rhs.stride_0()
                      : (r == 1
                             ? rhs.stride_1()
                             : (r == 2
                                    ? rhs.stride_2()
                                    : (r == 3
                                           ? rhs.stride_3()
                                           : (r == 4
                                                  ? rhs.stride_4()
                                                  : (r == 5
                                                         ? rhs.stride_5()
                                                         : (r == 6
                                                                ? rhs.stride_6()
                                                                : rhs.stride_7())))))));
  }

 public:
  template <class DimRHS, class LayoutRHS>
  KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
      const ViewOffset<DimRHS, LayoutRHS, void> &rhs,
      const SubviewExtents<DimRHS::rank, dimension_type::rank> &sub)
      // range_extent(r) returns 0 when dimension_type::rank <= r
      : m_dim(sub.range_extent(0), sub.range_extent(1), sub.range_extent(2),
              sub.range_extent(3), sub.range_extent(4), sub.range_extent(5),
              sub.range_extent(6), sub.range_extent(7))
        // range_index(r) returns ~0u when dimension_type::rank <= r
        ,
        m_stride(
            stride(sub.range_index(0), rhs), stride(sub.range_index(1), rhs),
            stride(sub.range_index(2), rhs), stride(sub.range_index(3), rhs),
            stride(sub.range_index(4), rhs), stride(sub.range_index(5), rhs),
            stride(sub.range_index(6), rhs), stride(sub.range_index(7), rhs)) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_VIEWOFFSET_HPP
