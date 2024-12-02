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

#ifndef KOKKOS_REMOTESPACES_NVSHMEM_BLOCK_OPS_HPP
#define KOKKOS_REMOTESPACES_NVSHMEM_BLOCK_OPS_HPP

#include <nvshmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

// #define KRS_USES_NBI

#define KOKKOS_REMOTESPACES_PUT(type, op)                                 \
  static __device__ void shmem_block_type_put(type *dst, const type *src, \
                                              size_t nelems, int pe) {    \
    op(dst, src, nelems, pe);                                             \
  }

#define KOKKOS_REMOTESPACES_GET(type, op)                                 \
  static __device__ void shmem_block_type_get(type *dst, const type *src, \
                                              size_t nelems, int pe) {    \
    op(dst, src, nelems, pe);                                             \
  }

#ifdef KRS_USES_NBI_BLOCK
#error Not supported
KOKKOS_REMOTESPACES_PUT(char, nvshmemx_char_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(unsigned char, nvshmemx_uchar_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(short, nvshmemx_short_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(unsigned short, nvshmemx_ushort_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(int, nvshmemx_int_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(unsigned int, nvshmemx_uint_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(long, nvshmemx_long_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(unsigned long, nvshmemx_ulong_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(long long, nvshmemx_longlong_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(unsigned long long, nvshmemx_ulonglong_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(float, nvshmemx_float_put_nbi_block)
KOKKOS_REMOTESPACES_PUT(double, nvshmemx_double_put_nbi_block)
KOKKOS_REMOTESPACES_GET(char, nvshmemx_char_get_nbi_block)
KOKKOS_REMOTESPACES_GET(unsigned char, nvshmemx_uchar_get_nbi_block)
KOKKOS_REMOTESPACES_GET(short, nvshmemx_short_get_nbi_block)
KOKKOS_REMOTESPACES_GET(unsigned short, nvshmemx_ushort_get_nbi_block)
KOKKOS_REMOTESPACES_GET(int, nvshmemx_int_get_nbi_block)
KOKKOS_REMOTESPACES_GET(unsigned int, nvshmemx_uint_get_nbi_block)
KOKKOS_REMOTESPACES_GET(long, nvshmemx_long_get_nbi_block)
KOKKOS_REMOTESPACES_GET(unsigned long, nvshmemx_ulong_get_nbi_block)
KOKKOS_REMOTESPACES_GET(long long, nvshmemx_longlong_get_nbi_block)
KOKKOS_REMOTESPACES_GET(unsigned long long, nvshmemx_ulonglong_get_nbi_block)
KOKKOS_REMOTESPACES_GET(float, nvshmemx_float_get_nbi_block)
KOKKOS_REMOTESPACES_GET(double, nvshmemx_double_get_nbi_block)
#elif defined(KRS_USES_NBI)
KOKKOS_REMOTESPACES_PUT(char, nvshmem_char_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned char, nvshmem_uchar_put_nbi)
KOKKOS_REMOTESPACES_PUT(short, nvshmem_short_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned short, nvshmem_ushort_put_nbi)
KOKKOS_REMOTESPACES_PUT(int, nvshmem_int_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned int, nvshmem_uint_put_nbi)
KOKKOS_REMOTESPACES_PUT(long, nvshmem_long_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned long, nvshmem_ulong_put_nbi)
KOKKOS_REMOTESPACES_PUT(long long, nvshmem_longlong_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned long long, nvshmem_ulonglong_put_nbi)
KOKKOS_REMOTESPACES_PUT(float, nvshmem_float_put_nbi)
KOKKOS_REMOTESPACES_PUT(double, nvshmem_double_put_nbi)
KOKKOS_REMOTESPACES_GET(char, nvshmem_char_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned char, nvshmem_uchar_get_nbi)
KOKKOS_REMOTESPACES_GET(short, nvshmem_short_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned short, nvshmem_ushort_get_nbi)
KOKKOS_REMOTESPACES_GET(int, nvshmem_int_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned int, nvshmem_uint_get_nbi)
KOKKOS_REMOTESPACES_GET(long, nvshmem_long_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned long, nvshmem_ulong_get_nbi)
KOKKOS_REMOTESPACES_GET(long long, nvshmem_longlong_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned long long, nvshmem_ulonglong_get_nbi)
KOKKOS_REMOTESPACES_GET(float, nvshmem_float_get_nbi)
KOKKOS_REMOTESPACES_GET(double, nvshmem_double_get_nbi)
#else
KOKKOS_REMOTESPACES_PUT(char, nvshmem_char_put)
KOKKOS_REMOTESPACES_PUT(unsigned char, nvshmem_uchar_put)
KOKKOS_REMOTESPACES_PUT(short, nvshmem_short_put)
KOKKOS_REMOTESPACES_PUT(unsigned short, nvshmem_ushort_put)
KOKKOS_REMOTESPACES_PUT(int, nvshmem_int_put)
KOKKOS_REMOTESPACES_PUT(unsigned int, nvshmem_uint_put)
KOKKOS_REMOTESPACES_PUT(long, nvshmem_long_put)
KOKKOS_REMOTESPACES_PUT(unsigned long, nvshmem_ulong_put)
KOKKOS_REMOTESPACES_PUT(long long, nvshmem_longlong_put)
KOKKOS_REMOTESPACES_PUT(unsigned long long, nvshmem_ulonglong_put)
KOKKOS_REMOTESPACES_PUT(float, nvshmem_float_put)
KOKKOS_REMOTESPACES_PUT(double, nvshmem_double_put)
KOKKOS_REMOTESPACES_GET(char, nvshmem_char_get)
KOKKOS_REMOTESPACES_GET(unsigned char, nvshmem_uchar_get)
KOKKOS_REMOTESPACES_GET(short, nvshmem_short_get)
KOKKOS_REMOTESPACES_GET(unsigned short, nvshmem_ushort_get)
KOKKOS_REMOTESPACES_GET(int, nvshmem_int_get)
KOKKOS_REMOTESPACES_GET(unsigned int, nvshmem_uint_get)
KOKKOS_REMOTESPACES_GET(long, nvshmem_long_get)
KOKKOS_REMOTESPACES_GET(unsigned long, nvshmem_ulong_get)
KOKKOS_REMOTESPACES_GET(long long, nvshmem_longlong_get)
KOKKOS_REMOTESPACES_GET(unsigned long long, nvshmem_ulonglong_get)
KOKKOS_REMOTESPACES_GET(float, nvshmem_float_get)
KOKKOS_REMOTESPACES_GET(double, nvshmem_double_get)
#endif

#undef KOKKOS_REMOTESPACES_PUT
#undef KOKKOS_REMOTESPACES_GET

template <class T, class Traits, typename Enable = void>
struct NVSHMEMBlockDataElement {};

template <class T, class Traits>
struct NVSHMEMBlockDataElement<T, Traits> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *src;
  T *dst;
  size_t nelems;
  int pe;

  KOKKOS_INLINE_FUNCTION
  NVSHMEMBlockDataElement(T *src_, T *dst_, size_t size_, int pe_)
      : src(src_), dst(dst_), nelems(size_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  void put() const { shmem_block_type_put(dst, src, nelems, pe); }

  KOKKOS_INLINE_FUNCTION
  void get() const { shmem_block_type_get(dst, src, nelems, pe); }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_NVSHMEM_BLOCK_OPS_HPP
