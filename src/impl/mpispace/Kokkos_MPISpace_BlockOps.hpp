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

#ifndef KOKKOS_REMOTESPACES_shmem_BLOCK_OPS_HPP
#define KOKKOS_REMOTESPACES_shmem_BLOCK_OPS_HPP

#include <shmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

#define KOKKOS_REMOTESPACES_PUT(type, op)                  \
  static KOKKOS_INLINE_FUNCTION void shmem_block_type_put( \
      type *dst, const type *src, size_t nelems, int pe) { \
    op(dst, src, nelems, pe);                              \
  }

KOKKOS_REMOTESPACES_PUT(char, shmem_char_put)
KOKKOS_REMOTESPACES_PUT(unsigned char, shmem_uchar_put)
KOKKOS_REMOTESPACES_PUT(short, shmem_short_put)
KOKKOS_REMOTESPACES_PUT(unsigned short, shmem_ushort_put)
KOKKOS_REMOTESPACES_PUT(int, shmem_int_put)
KOKKOS_REMOTESPACES_PUT(unsigned int, shmem_uint_put)
KOKKOS_REMOTESPACES_PUT(long, shmem_long_put)
KOKKOS_REMOTESPACES_PUT(unsigned long, shmem_ulong_put)
KOKKOS_REMOTESPACES_PUT(long long, shmem_longlong_put)
KOKKOS_REMOTESPACES_PUT(unsigned long long, shmem_ulonglong_put)
KOKKOS_REMOTESPACES_PUT(float, shmem_float_put)
KOKKOS_REMOTESPACES_PUT(double, shmem_double_put)

#undef KOKKOS_REMOTESPACES_PUT

#define KOKKOS_REMOTESPACES_GET(type, op)                  \
  static KOKKOS_INLINE_FUNCTION void shmem_block_type_get( \
      type *dst, const type *src, size_t nelems, int pe) { \
    op(dst, src, nelems, pe);                              \
  }

KOKKOS_REMOTESPACES_GET(char, shmem_char_get)
KOKKOS_REMOTESPACES_GET(unsigned char, shmem_uchar_get)
KOKKOS_REMOTESPACES_GET(short, shmem_short_get)
KOKKOS_REMOTESPACES_GET(unsigned short, shmem_ushort_get)
KOKKOS_REMOTESPACES_GET(int, shmem_int_get)
KOKKOS_REMOTESPACES_GET(unsigned int, shmem_uint_get)
KOKKOS_REMOTESPACES_GET(long, shmem_long_get)
KOKKOS_REMOTESPACES_GET(unsigned long, shmem_ulong_get)
KOKKOS_REMOTESPACES_GET(long long, shmem_longlong_get)
KOKKOS_REMOTESPACES_GET(unsigned long long, shmem_ulonglong_get)
KOKKOS_REMOTESPACES_GET(float, shmem_float_get)
KOKKOS_REMOTESPACES_GET(double, shmem_double_get)

#undef KOKKOS_REMOTESPACES_GET

template <class T, class Traits, typename Enable = void>
struct MPIBlockDataElement {};

// Atomic Operators
template <class T, class Traits>
struct MPIBlockDataElement<T, Traits> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *src;
  T *dst;
  size_t nelems;
  int pe;

  KOKKOS_INLINE_FUNCTION
  MPIBlockDataElement(T *src_, T *dst_, size_t size_, int pe_)
      : src(src_), dst(dst_), nelems(size_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  void put() const { shmem_block_type_put(dst, src, nelems, pe); }

  KOKKOS_INLINE_FUNCTION
  void get() const { shmem_block_type_get(dst, src, nelems, pe); }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_shmem_BLOCK_OPS_HPP
