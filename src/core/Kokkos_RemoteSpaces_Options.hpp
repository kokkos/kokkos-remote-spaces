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

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace Kokkos {

enum RemoteSpaces_MemoryTraitsFlags { Dim0IsPE = 16384, Cached = 32768 };

template <typename T>
struct RemoteSpaces_MemoryTraits;

template <unsigned T>
struct RemoteSpaces_MemoryTraits<MemoryTraits<T>> {
  enum : bool { dim0_is_pe = (unsigned(0) != (T & unsigned(Dim0IsPE))) };
  enum : bool { is_cached = (unsigned(0) != (T & unsigned(Cached))) };
  enum : int { state = T };
};
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_OPTIONS_HPP
