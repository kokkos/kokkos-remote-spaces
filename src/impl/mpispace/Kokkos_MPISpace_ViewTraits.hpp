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

#ifndef KOKKOS_REMOTESPACES_MPI_VIEWTRAITS_HPP
#define KOKKOS_REMOTESPACES_MPI_VIEWTRAITS_HPP

namespace Kokkos {

/*
 * ViewTraits class evaluated during View specialization
 */

template <class... Properties>
struct ViewTraits<void, Kokkos::Experimental::MPISpace, Properties...> {
  static_assert(
      std::is_same<typename ViewTraits<void, Properties...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Properties...>::memory_space,
                       void>::value &&
          std::is_same<
              typename ViewTraits<void, Properties...>::HostMirrorSpace,
              void>::value &&
          std::is_same<typename ViewTraits<void, Properties...>::array_layout,
                       void>::value,
      "Only one View Execution or Memory Space template argument");

  // Specify layout, keep subsequent space and memory traits arguments
  using execution_space =
      typename Kokkos::Experimental::MPISpace::execution_space;
  using memory_space = typename Kokkos::Experimental::MPISpace::memory_space;
  using HostMirrorSpace =
      typename Kokkos::Impl::HostMirror<Kokkos::Experimental::MPISpace>::Space;
  using array_layout  = typename execution_space::array_layout;
  using specialize    = Kokkos::Experimental::RemoteSpaceSpecializeTag;
  using memory_traits = typename ViewTraits<void, Properties...>::memory_traits;
  using hooks_policy  = typename ViewTraits<void, Properties...>::hooks_policy;
};

}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_MPI_VIEWTRAITS_HPP