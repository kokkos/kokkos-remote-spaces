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
namespace Impl {

template <class T, class P>
auto KOKKOS_INLINE_FUNCTION get_local_subview(T view, P r) {
  if constexpr (T::traits::dimension::rank == 0) {
    return view;
  } else if constexpr (T::traits::dimension::rank == 1) {
    return Kokkos::subview(view, r);
  } else if constexpr (T::traits::dimension::rank == 2) {
    return Kokkos::subview(view, r, Kokkos::ALL);
  } else if constexpr (T::traits::dimension::rank == 3) {
    return Kokkos::subview(view, r, Kokkos::ALL, Kokkos::ALL);
  } else if constexpr (T::traits::dimension::rank == 4) {
    return Kokkos::subview(view, r, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  } else if constexpr (T::traits::dimension::rank == 5) {
    return Kokkos::subview(view, r, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  } else if constexpr (T::traits::dimension::rank == 6) {
    return Kokkos::subview(view, r, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  } else if constexpr (T::traits::dimension::rank == 7) {
    return Kokkos::subview(view, r, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  } else {
    static_assert("Unsupported view type");
  }
}

template <class T>
auto KOKKOS_INLINE_FUNCTION get_view_adr(T view) {
  return view.impl_map().handle().ptr;
}
}  // namespace Impl

namespace Experimental {
namespace RemoteSpaces {

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
  int src_rank = src.impl_map().get_logical_PE();
  int dst_rank = dst.impl_map().get_logical_PE();
  int my_rank  = get_my_pe();

  if (src_rank != my_rank && dst_rank != my_rank) {
    // Both views are remote, copy through view accessor (TODO)
    static_assert("local_deep_copy for provided views is not supported");
  }

  if (dst_rank == my_rank && src_rank == my_rank) {
    // Both views are local, copy as array operation
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, src.span()),
                         [&](const int &i) { dst.data()[i] = src.data()[i]; });
    return;
  }

  using src_data_block_t =
      Kokkos::Impl::BlockDataHandle<typename ViewTraits<ST, SP...>::value_type,
                                    ViewTraits<ST, SP...>>;
  using dst_data_block_t =
      Kokkos::Impl::BlockDataHandle<typename ViewTraits<DT, DP...>::value_type,
                                    ViewTraits<DT, DP...>>;

  using size_type = typename ViewTraits<DT, DP...>::size_type;

  auto league_size = team.league_size();
  auto team_ID     = team.league_rank();

  // Construct per-team range
  auto team_block     = (dst.extent(0)) / league_size;
  auto team_block_mod = (dst.extent(0)) % league_size;
  auto start_offset   = team_ID * team_block;
  team_block =
      team_ID == league_size - 1 ? team_block + team_block_mod : team_block;
  auto team_range = Kokkos::pair(size_type(start_offset),
                                 size_type(start_offset + team_block));

  // Construct per-team subviews
  auto src_subview = Kokkos::Impl::get_local_subview(src, team_range);
  auto dst_subview = Kokkos::Impl::get_local_subview(dst, team_range);

  // Construct subview offsets
  auto src_subview_ptr = Kokkos::Impl::get_view_adr(src_subview);
  auto dst_subview_ptr = Kokkos::Impl::get_view_adr(dst_subview);

  if (src_rank != my_rank) {
    team.team_barrier();
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
#ifdef KRS_ENABLE_MPISPACE
      src_data_block_t data_block = src_data_block_t(
          dst_subview_ptr, src_subview.impl_map().handle().loc.win,
          src_subview.impl_map().handle().loc.offset, src_subview.span(),
          src_rank);
#else
      src_data_block_t data_block =
          src_data_block_t(dst_subview_ptr, src_subview_ptr, src_subview.span(), src_rank);
#endif
      data_block.get();
#ifdef KRS_ENABLE_MPISPACE
      MPI_Win_flush_all(src.impl_map().m_handle.loc.win);
#endif
    });
  } else if (dst_rank != my_rank) {
    team.team_barrier();
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
#ifdef KRS_ENABLE_MPISPACE
      dst_data_block_t data_block = dst_data_block_t(
          src_subview_ptr, dst_subview.impl_map().handle().loc.win,
          dst_subview.impl_map().handle().loc.offset, dst_subview.span(),
          dst_rank);
#else
      src_data_block_t data_block =
          src_data_block_t(dst_subview_ptr, src_subview_ptr, src_subview.span(), dst_rank);
#endif
      data_block.put();
#ifdef KRS_ENABLE_MPISPACE
      MPI_Win_flush_all(src.impl_map().m_handle.loc.win);
#endif
    });
  } else {
    static_assert("Unable to determine view data location");
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
  int src_rank = src.impl_map().get_logical_PE();
  int dst_rank = dst.impl_map().get_logical_PE();
  int my_rank  = get_my_pe();

  if (src_rank != my_rank && dst_rank != my_rank) {
    // Both views are remote, copy through view accessor (TODO)
    static_assert("local_deep_copy for provided views is not supported");
  }

  if (dst_rank == my_rank && src_rank == my_rank) {
    // Both views are local, copy as array operation
    for (size_t i = 0; i < src.span(); ++i) dst.data()[i] = src.data()[i];
    return;
  }

  using src_data_block_t =
      Kokkos::Impl::BlockDataHandle<typename ViewTraits<ST, SP...>::value_type,
                                    ViewTraits<ST, SP...>>;
  using dst_data_block_t =
      Kokkos::Impl::BlockDataHandle<typename ViewTraits<DT, DP...>::value_type,
                                    ViewTraits<DT, DP...>>;

  // Construct subview offsets
  auto src_subview_ptr = Kokkos::Impl::get_view_adr(src);
  auto dst_subview_ptr = Kokkos::Impl::get_view_adr(dst);

  printf("LDC: %p, %p, %p %p\n", dst.data(), src.data(), dst_subview_ptr,
         src_subview_ptr);

  if (src_rank != my_rank) {
#ifdef KRS_ENABLE_MPISPACE
    src_data_block_t data_block = src_data_block_t(
        dst_subview_ptr, src.impl_map().handle().loc.win,
        src.impl_map().handle().loc.offset, src.span(), src_rank);
#else
    src_data_block_t data_block = src_data_block_t(
        dst_subview_ptr, src_subview_ptr, src.span(), src_rank);
#endif
    data_block.get();
#ifdef KRS_ENABLE_MPISPACE
    MPI_Win_flush_all(src.impl_map().m_handle.loc.win);
#endif
  } else if (dst_rank != my_rank) {
#ifdef KRS_ENABLE_MPISPACE
    dst_data_block_t data_block = dst_data_block_t(
        src_subview_ptr, dst.impl_map().handle().loc.win,
        dst.impl_map().handle().loc.offset, dst.span(), dst_rank);
#else
    src_data_block_t data_block = src_data_block_t(
        dst_subview_ptr, src_subview_ptr, src.span(), dst_rank);
#endif
    data_block.put();
#ifdef KRS_ENABLE_MPISPACE
    MPI_Win_flush_all(src.impl_map().m_handle.loc.win);
#endif
  } else {
    static_assert("Unable to determine view data location");
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
