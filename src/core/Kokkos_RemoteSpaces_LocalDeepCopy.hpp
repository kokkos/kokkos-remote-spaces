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

#ifndef KOKKOS_REMOTESPACES_LOCALDEEPCOPY_HPP
#define KOKKOS_REMOTESPACES_LOCALDEEPCOPY_HPP

#include <Kokkos_RemoteSpaces.hpp>

namespace Kokkos {
namespace Experimental {
namespace RemoteSpaces {

#ifdef KRS_ENABLE_NVSHMEMSPACE
typedef NVSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KRS_ENABLE_SHMEMSPACE
typedef SHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KRS_ENABLE_MPISPACE
typedef MPISpace DefaultRemoteMemorySpace;
#endif
#endif
#endif

/** \brief  A local deep copy between views of the default specialization,
 * compatible type, same non-zero rank.
 */
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  int src_rank = src.impl_map().get_owning_pe();
  int dst_rank = dst.impl_map().get_owning_pe();
  int my_rank  = get_my_pe();

  if (src_rank != my_rank && dst_rank != my_rank)
    static_assert(
        "local_deep_copy allows only one view with remote data access");

  //We use the data ptr explicitly thus expecting that a subview starts at the 
  //beginning of the local allocaton. We need to add the offset = sum of offsets 
  //in all non-leading dimenions to the ptr to support the generic case. 
  using src_data_block_t = Kokkos::Impl::NVSHMEMBlockDataHandle<
      typename ViewTraits<ST, SP...>::value_type, ViewTraits<ST, SP...>>;
  using dst_data_block_t = Kokkos::Impl::NVSHMEMBlockDataHandle<
      typename ViewTraits<DT, DP...>::value_type, ViewTraits<DT, DP...>>;
  if (src_rank != my_rank) {
    team.team_barrier();
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      src_data_block_t src_data =
          src_data_block_t(dst.data(), src.data(), src.span(), src_rank);
      src_data.get();
    });
  } else if (dst_rank != my_rank) {
    team.team_barrier();
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      dst_data_block_t dst_data =
          dst_data_block_t(dst.data(), src.data(), dst.span(), dst_rank);
      dst_data.put();
    });
  } else {
    // Data resides within the node, copy as usual
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, src.span()),
                         [&](const int &i) { dst.data()[i] = src.data()[i]; });
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  int src_rank = src.impl_map().get_owning_pe();
  int dst_rank = dst.impl_map().get_owning_pe();
  int my_rank  = get_my_pe();

  if (src_rank != my_rank && dst_rank != my_rank)
    static_assert(
        "local_deep_copy allows only one view with remote data access");

  using src_data_block_t = Kokkos::Impl::NVSHMEMBlockDataHandle<
      typename ViewTraits<ST, SP...>::value_type, ViewTraits<ST, SP...>>;
  using dst_data_block_t = Kokkos::Impl::NVSHMEMBlockDataHandle<
      typename ViewTraits<DT, DP...>::value_type, ViewTraits<DT, DP...>>;

  //We use the data ptr explicitly thus expecting that a subview starts at the 
  //beginning of the local allocaton. We need to add the offset = sum of offsets 
  //in all non-leading dimenions to the ptr to support the generic case. 
  if (src_rank != my_rank) {
    src_data_block_t src_data =
        src_data_block_t(dst.data(), src.data(), src.span(), src_rank);
    src_data.get();
  } else if (dst_rank != my_rank) {
    dst_data_block_t dst_data =
        dst_data_block_t(dst.data(), src.data(), dst.span(), dst_rank);
    dst_data.put();
  } else {
    // Data resides within the node, copy as usual
    for (size_t i = 0; i < src.span(); ++i) dst.data()[i] = src.data()[i];
  }
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<std::is_same<
        typename ViewTraits<DT, DP...>::specialize,
        Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type * =
        nullptr

) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dst.span()),
                       [&](const int &i) { dst.data()[i] = value; });
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy_contiguous(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<std::is_same<
        typename ViewTraits<DT, DP...>::specialize,
        Kokkos::Experimental::RemoteSpaceSpecializeTag>::value>::type * =
        nullptr) {
  for (size_t i = 0; i < dst.span(); ++i) {
    dst.data()[i] = value;
  }
}

// Accepts (team, src_view, dst_view)
template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 1 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N),
                       [&](const int &i) { dst(i) = src(i); });
  team.team_barrier();
}

template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 2 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0      = i % dst.extent(0);
      int i1      = i / dst.extent(0);
      dst(i0, i1) = src(i0, i1);
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 3 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0          = i % dst.extent(0);
      int itmp        = i / dst.extent(0);
      int i1          = itmp % dst.extent(1);
      int i2          = itmp / dst.extent(1);
      dst(i0, i1, i2) = src(i0, i1, i2);
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 4 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N =
      dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0              = i % dst.extent(0);
      int itmp            = i / dst.extent(0);
      int i1              = itmp % dst.extent(1);
      itmp                = itmp / dst.extent(1);
      int i2              = itmp % dst.extent(2);
      int i3              = itmp / dst.extent(2);
      dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 5 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0                  = i % dst.extent(0);
      int itmp                = i / dst.extent(0);
      int i1                  = itmp % dst.extent(1);
      itmp                    = itmp / dst.extent(1);
      int i2                  = itmp % dst.extent(2);
      itmp                    = itmp / dst.extent(2);
      int i3                  = itmp % dst.extent(3);
      int i4                  = itmp / dst.extent(3);
      dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 6 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0                      = i % dst.extent(0);
      int itmp                    = i / dst.extent(0);
      int i1                      = itmp % dst.extent(1);
      itmp                        = itmp / dst.extent(1);
      int i2                      = itmp % dst.extent(2);
      itmp                        = itmp / dst.extent(2);
      int i3                      = itmp % dst.extent(3);
      itmp                        = itmp / dst.extent(3);
      int i4                      = itmp % dst.extent(4);
      int i5                      = itmp / dst.extent(4);
      dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 7 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5) *
                   dst.extent(6);

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   src);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0                          = i % dst.extent(0);
      int itmp                        = i / dst.extent(0);
      int i1                          = itmp % dst.extent(1);
      itmp                            = itmp / dst.extent(1);
      int i2                          = itmp % dst.extent(2);
      itmp                            = itmp / dst.extent(2);
      int i3                          = itmp % dst.extent(3);
      itmp                            = itmp / dst.extent(3);
      int i4                          = itmp % dst.extent(4);
      itmp                            = itmp / dst.extent(4);
      int i5                          = itmp % dst.extent(5);
      int i6                          = itmp / dst.extent(5);
      dst(i0, i1, i2, i3, i4, i5, i6) = src(i0, i1, i2, i3, i4, i5, i6);
    });
    team.team_barrier();
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 1 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }
  Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 2 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = src(i0, i1);
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 3 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          dst(i0, i1, i2) = src(i0, i1, i2);
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 4 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            dst(i0, i1, i2, i3) = src(i0, i1, i2, i3);
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 5 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              dst(i0, i1, i2, i3, i4) = src(i0, i1, i2, i3, i4);
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 6 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                dst(i0, i1, i2, i3, i4, i5) = src(i0, i1, i2, i3, i4, i5);
  }
}

template <class DT, class... DP, class ST, class... SP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst, const View<ST, SP...> &src,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
         unsigned(ViewTraits<ST, SP...>::rank) == 7 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value &&
         std::is_same<typename ViewTraits<ST, SP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous() && src.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, src);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                  dst(i0, i1, i2, i3, i4, i5, i6) =
                      src(i0, i1, i2, i3, i4, i5, i6);
  }
}

// Accepts (team, src_view, value)

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N),
                       [&](const int &i) { dst(i) = value; });
  team.team_barrier();
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0      = i % dst.extent(0);
      int i1      = i / dst.extent(0);
      dst(i0, i1) = value;
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0          = i % dst.extent(0);
      int itmp        = i / dst.extent(0);
      int i1          = itmp % dst.extent(1);
      int i2          = itmp / dst.extent(1);
      dst(i0, i1, i2) = value;
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N =
      dst.extent(0) * dst.extent(1) * dst.extent(2) * dst.extent(3);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0              = i % dst.extent(0);
      int itmp            = i / dst.extent(0);
      int i1              = itmp % dst.extent(1);
      itmp                = itmp / dst.extent(1);
      int i2              = itmp % dst.extent(2);
      int i3              = itmp / dst.extent(2);
      dst(i0, i1, i2, i3) = value;
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0                  = i % dst.extent(0);
      int itmp                = i / dst.extent(0);
      int i1                  = itmp % dst.extent(1);
      itmp                    = itmp / dst.extent(1);
      int i2                  = itmp % dst.extent(2);
      itmp                    = itmp / dst.extent(2);
      int i3                  = itmp % dst.extent(3);
      int i4                  = itmp / dst.extent(3);
      dst(i0, i1, i2, i3, i4) = value;
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0                      = i % dst.extent(0);
      int itmp                    = i / dst.extent(0);
      int i1                      = itmp % dst.extent(1);
      itmp                        = itmp / dst.extent(1);
      int i2                      = itmp % dst.extent(2);
      itmp                        = itmp / dst.extent(2);
      int i3                      = itmp % dst.extent(3);
      itmp                        = itmp / dst.extent(3);
      int i4                      = itmp % dst.extent(4);
      int i5                      = itmp / dst.extent(4);
      dst(i0, i1, i2, i3, i4, i5) = value;
    });
    team.team_barrier();
  }
}

template <class TeamType, class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const TeamType &team, const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0) * dst.extent(1) * dst.extent(2) *
                   dst.extent(3) * dst.extent(4) * dst.extent(5) *
                   dst.extent(6);

  if (dst.span_is_contiguous()) {
    team.team_barrier();
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(team, dst,
                                                                   value);
    team.team_barrier();
  } else {
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const int &i) {
      int i0                          = i % dst.extent(0);
      int itmp                        = i / dst.extent(0);
      int i1                          = itmp % dst.extent(1);
      itmp                            = itmp / dst.extent(1);
      int i2                          = itmp % dst.extent(2);
      itmp                            = itmp / dst.extent(2);
      int i3                          = itmp % dst.extent(3);
      itmp                            = itmp / dst.extent(3);
      int i4                          = itmp % dst.extent(4);
      itmp                            = itmp / dst.extent(4);
      int i5                          = itmp % dst.extent(5);
      int i6                          = itmp / dst.extent(5);
      dst(i0, i1, i2, i3, i4, i5, i6) = value;
    });
    team.team_barrier();
  }
}

// Accepts (src_view, value)

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 1 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  const size_t N = dst.extent(0);

  for (size_t i = 0; i < N; ++i) {
    dst(i) = value;
  }
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 2 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1) dst(i0, i1) = value;
  }
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 3 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2) dst(i0, i1, i2) = value;
  }
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 4 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            dst(i0, i1, i2, i3) = value;
  }
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 5 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              dst(i0, i1, i2, i3, i4) = value;
  }
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 6 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                dst(i0, i1, i2, i3, i4, i5) = value;
  }
}

template <class DT, class... DP>
void KOKKOS_INLINE_FUNCTION local_deep_copy(
    const View<DT, DP...> &dst,
    typename ViewTraits<DT, DP...>::const_value_type &value,
    typename std::enable_if<
        (unsigned(ViewTraits<DT, DP...>::rank) == 7 &&
         std::is_same<typename ViewTraits<DT, DP...>::specialize,
                      Kokkos::Experimental::RemoteSpaceSpecializeTag>::value)>::
        type * = nullptr) {
  if (dst.data() == nullptr) {
    return;
  }

  if (dst.span_is_contiguous()) {
    Kokkos::Experimental::RemoteSpaces::local_deep_copy_contiguous(dst, value);
  } else {
    for (size_t i0 = 0; i0 < dst.extent(0); ++i0)
      for (size_t i1 = 0; i1 < dst.extent(1); ++i1)
        for (size_t i2 = 0; i2 < dst.extent(2); ++i2)
          for (size_t i3 = 0; i3 < dst.extent(3); ++i3)
            for (size_t i4 = 0; i4 < dst.extent(4); ++i4)
              for (size_t i5 = 0; i5 < dst.extent(5); ++i5)
                for (size_t i6 = 0; i6 < dst.extent(6); ++i6)
                  dst(i0, i1, i2, i3, i4, i5, i6) = value;
  }
}

}  // namespace RemoteSpaces
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_LOCALDEEPCOPY_HPP