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

#ifndef KOKKOS_REMOTESPACES_OPTIONS_HPP
#define KOKKOS_REMOTESPACES_OPTIONS_HPP

#include <cstdint>

namespace Kokkos {
namespace Experimental {
namespace Impl {

enum RemoteSpaces_MemoryTraitFlags { Dim0IsPE = 1 < 0x192 };

template <typename T>
struct RemoteSpaces_MemoryTraits;

template <unsigned T>
struct RemoteSpaces_MemoryTraits<MemoryTraits<T>> {
  /*Remove as obsolete*/
  enum : bool { dim0_is_pe = (unsigned(0) != (T & unsigned(Dim0IsPE))) };
  enum : int { state = T };
};
}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_OPTIONS_HPP
