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

#ifndef KOKKOS_REMOTESPACES_VIEWLAYOUT_HPP
#define KOKKOS_REMOTESPACES_VIEWMLAYOUT_HPP

#include <type_traits>

//----------------------------------------------------------------------------
/** \brief  View mapping for non-specialized data type and standard layout */

namespace Kokkos {

struct PartitionedLayoutLeft {
  //! Tag this class as a kokkos array layout
  using array_layout = PartitionedLayoutLeft;

  size_t dimension[ARRAY_LAYOUT_MAX_RANK];

  enum : bool { is_extent_constructible = true };

  PartitionedLayoutLeft(PartitionedLayoutLeft const &) = default;
  PartitionedLayoutLeft(PartitionedLayoutLeft &&) = default;
  PartitionedLayoutLeft &operator=(PartitionedLayoutLeft const &) = default;
  PartitionedLayoutLeft &operator=(PartitionedLayoutLeft &&) = default;

  KOKKOS_INLINE_FUNCTION
  explicit constexpr PartitionedLayoutLeft(size_t N0 = 0, size_t N1 = 0,
                                      size_t N2 = 0, size_t N3 = 0,
                                      size_t N4 = 0, size_t N5 = 0,
                                      size_t N6 = 0, size_t N7 = 0)
      : dimension{N0, N1, N2, N3, N4, N5, N6, N7} {}
};

struct PartitionedLayoutRight {
  //! Tag this class as a kokkos array layout
  using array_layout = PartitionedLayoutRight;

  size_t dimension[ARRAY_LAYOUT_MAX_RANK];

  enum : bool { is_extent_constructible = true };

  PartitionedLayoutRight(PartitionedLayoutRight const &) = default;
  PartitionedLayoutRight(PartitionedLayoutRight &&) = default;
  PartitionedLayoutRight &operator=(PartitionedLayoutRight const &) = default;
  PartitionedLayoutRight &operator=(PartitionedLayoutRight &&) = default;

  KOKKOS_INLINE_FUNCTION
  explicit constexpr PartitionedLayoutRight(size_t N0 = 0, size_t N1 = 0,
                                       size_t N2 = 0, size_t N3 = 0,
                                       size_t N4 = 0, size_t N5 = 0,
                                       size_t N6 = 0, size_t N7 = 0)
      : dimension{N0, N1, N2, N3, N4, N5, N6, N7} {}
};

/// \struct PartitionedLayoutStride
/// \brief  Memory layout tag indicated arbitrarily strided
///         multi-index mapping into contiguous memory.
struct PartitionedLayoutStride {
  //! Tag this class as a kokkos array layout
  using array_layout = PartitionedLayoutStride;

  size_t dimension[ARRAY_LAYOUT_MAX_RANK];
  size_t stride[ARRAY_LAYOUT_MAX_RANK];

  enum : bool { is_extent_constructible = false };

  PartitionedLayoutStride(PartitionedLayoutStride const &) = default;
  PartitionedLayoutStride(PartitionedLayoutStride &&) = default;
  PartitionedLayoutStride &operator=(PartitionedLayoutStride const &) = default;
  PartitionedLayoutStride &operator=(PartitionedLayoutStride &&) = default;

  /** \brief  Compute strides from ordered dimensions.
   *
   *  Values of order uniquely form the set [0..rank)
   *  and specify ordering of the dimensions.
   *  Order = {0,1,2,...} is LayoutLeft
   *  Order = {...,2,1,0} is LayoutRight
   */
  template <typename iTypeOrder, typename iTypeDimen>
  KOKKOS_INLINE_FUNCTION static PartitionedLayoutStride
  order_dimensions(int const rank, iTypeOrder const *const order,
                   iTypeDimen const *const dimen) {
    PartitionedLayoutStride tmp;
    // Verify valid rank order:
    int check_input = ARRAY_LAYOUT_MAX_RANK < rank ? 0 : int(1 << rank) - 1;
    for (int r = 0; r < ARRAY_LAYOUT_MAX_RANK; ++r) {
      tmp.dimension[r] = 0;
      tmp.stride[r] = 0;
    }
    for (int r = 0; r < rank; ++r) {
      check_input &= ~int(1 << order[r]);
    }
    if (0 == check_input) {
      size_t n = 1;
      for (int r = 0; r < rank; ++r) {
        tmp.stride[order[r]] = n;
        n *= (dimen[order[r]]);
        tmp.dimension[r] = dimen[r];
      }
    }
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  explicit constexpr PartitionedLayoutStride(
      size_t N0 = 0, size_t S0 = 0, size_t N1 = 0, size_t S1 = 0, size_t N2 = 0,
      size_t S2 = 0, size_t N3 = 0, size_t S3 = 0, size_t N4 = 0, size_t S4 = 0,
      size_t N5 = 0, size_t S5 = 0, size_t N6 = 0, size_t S6 = 0, size_t N7 = 0,
      size_t S7 = 0)
      : dimension{N0, N1, N2, N3, N4, N5, N6, N7}, stride{S0, S1, S2, S3,
                                                          S4, S5, S6, S7} {}
};

namespace Impl {

// Rules for subview arguments and global layouts matching
// Rules which allow LayoutLeft to LayoutLeft assignment

template <int RankDest, int RankSrc, int CurrentArg, class Arg,
          class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutLeft,
                                   Kokkos::PartitionedLayoutLeft, RankDest, RankSrc,
                                   CurrentArg, Arg, SubViewArgs...> {
  enum {
    value = (((CurrentArg == RankDest - 1) &&
              (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
             ((CurrentArg >= RankDest) && (std::is_integral<Arg>::value)) ||
             ((CurrentArg < RankDest) &&
              (std::is_same<Arg, Kokkos::Impl::ALL_t>::value)) ||
             ((CurrentArg == 0) &&
              (Kokkos::Impl::is_integral_extent_type<Arg>::value))) &&
            (SubviewLegalArgsCompileTime<
                Kokkos::PartitionedLayoutLeft, Kokkos::PartitionedLayoutLeft, RankDest,
                RankSrc, CurrentArg + 1, SubViewArgs...>::value)
  };
};

template <int RankDest, int RankSrc, int CurrentArg, class Arg>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutLeft,
                                   Kokkos::PartitionedLayoutLeft, RankDest, RankSrc,
                                   CurrentArg, Arg> {
  enum {
    value = ((CurrentArg == RankDest - 1) || (std::is_integral<Arg>::value)) &&
            (CurrentArg == RankSrc - 1)
  };
};

// Rules which allow LayoutRight to LayoutRight assignment

template <int RankDest, int RankSrc, int CurrentArg, class Arg,
          class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutRight,
                                   Kokkos::PartitionedLayoutRight, RankDest, RankSrc,
                                   CurrentArg, Arg, SubViewArgs...> {
  enum {
    value = (((CurrentArg == RankSrc - RankDest) &&
              (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
             ((CurrentArg < RankSrc - RankDest) &&
              (std::is_integral<Arg>::value)) ||
             ((CurrentArg >= RankSrc - RankDest) &&
              (std::is_same<Arg, Kokkos::Impl::ALL_t>::value))) &&
            (SubviewLegalArgsCompileTime<
                Kokkos::PartitionedLayoutRight, Kokkos::PartitionedLayoutRight, RankDest,
                RankSrc, CurrentArg + 1, SubViewArgs...>::value)
  };
};

template <int RankDest, int RankSrc, int CurrentArg, class Arg>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutRight,
                                   Kokkos::PartitionedLayoutRight, RankDest, RankSrc,
                                   CurrentArg, Arg> {
  enum {
    value = ((CurrentArg == RankSrc - 1) &&
             (std::is_same<Arg, Kokkos::Impl::ALL_t>::value))
  };
};

// Rules which allow assignment to LayoutStride

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride,
                                   Kokkos::PartitionedLayoutLeft, RankDest, RankSrc,
                                   CurrentArg, SubViewArgs...> {
  enum : bool { value = false };
};

// We probably need to update / refine this

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutRight,
                                   Kokkos::PartitionedLayoutRight, RankDest, RankSrc,
                                   CurrentArg, SubViewArgs...> {
  enum : bool { value = true };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutLeft,
                                   Kokkos::PartitionedLayoutLeft, RankDest, RankSrc,
                                   CurrentArg, SubViewArgs...> {
  enum : bool { value = true };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutRight,
                                   Kokkos::LayoutRight, RankDest, RankSrc,
                                   CurrentArg, SubViewArgs...> {
  enum : bool { value = true };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::PartitionedLayoutLeft, Kokkos::LayoutLeft,
                                   RankDest, RankSrc, CurrentArg,
                                   SubViewArgs...> {
  enum : bool { value = true };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutRight,
                                   Kokkos::PartitionedLayoutRight, RankDest, RankSrc,
                                   CurrentArg, SubViewArgs...> {
  enum : bool { value = false };
};

template <int RankDest, int RankSrc, int CurrentArg, class... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::PartitionedLayoutLeft,
                                   RankDest, RankSrc, CurrentArg,
                                   SubViewArgs...> {
  enum : bool { value = false };
};

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_REMOTESPACES_VIEWLAYOUT_HPP
