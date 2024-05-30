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

#ifndef KOKKOS_REMOTESPACES_SHMEM_OPS_HPP
#define KOKKOS_REMOTESPACES_SHMEM_OPS_HPP

#include <shmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

#define KOKKOS_REMOTESPACES_P(type, op)                                       \
  static KOKKOS_INLINE_FUNCTION void shmem_type_p(type *ptr, const type &val, \
                                                  int pe) {                   \
    op(ptr, val, pe);                                                         \
  }

KOKKOS_REMOTESPACES_P(char, shmem_char_p)
KOKKOS_REMOTESPACES_P(unsigned char, shmem_uchar_p)
KOKKOS_REMOTESPACES_P(short, shmem_short_p)
KOKKOS_REMOTESPACES_P(unsigned short, shmem_ushort_p)
KOKKOS_REMOTESPACES_P(int, shmem_int_p)
KOKKOS_REMOTESPACES_P(unsigned int, shmem_uint_p)
KOKKOS_REMOTESPACES_P(long, shmem_long_p)
KOKKOS_REMOTESPACES_P(unsigned long, shmem_ulong_p)
KOKKOS_REMOTESPACES_P(long long, shmem_longlong_p)
KOKKOS_REMOTESPACES_P(unsigned long long, shmem_ulonglong_p)
KOKKOS_REMOTESPACES_P(float, shmem_float_p)
KOKKOS_REMOTESPACES_P(double, shmem_double_p)

#undef KOKKOS_REMOTESPACES_P

#define KOKKOS_REMOTESPACES_G(type, op)                                \
  static KOKKOS_INLINE_FUNCTION type shmem_type_g(type *ptr, int pe) { \
    return op(ptr, pe);                                                \
  }

KOKKOS_REMOTESPACES_G(char, shmem_char_g)
KOKKOS_REMOTESPACES_G(unsigned char, shmem_uchar_g)
KOKKOS_REMOTESPACES_G(short, shmem_short_g)
KOKKOS_REMOTESPACES_G(unsigned short, shmem_ushort_g)
KOKKOS_REMOTESPACES_G(int, shmem_int_g)
KOKKOS_REMOTESPACES_G(unsigned int, shmem_uint_g)
KOKKOS_REMOTESPACES_G(long, shmem_long_g)
KOKKOS_REMOTESPACES_G(unsigned long, shmem_ulong_g)
KOKKOS_REMOTESPACES_G(long long, shmem_longlong_g)
KOKKOS_REMOTESPACES_G(unsigned long long, shmem_ulonglong_g)
KOKKOS_REMOTESPACES_G(float, shmem_float_g)
KOKKOS_REMOTESPACES_G(double, shmem_double_g)

#undef KOKKOS_REMOTESPACES_G

#define KOKKOS_REMOTESPACES_ATOMIC_SET(type, op)            \
  static KOKKOS_INLINE_FUNCTION void shmem_type_atomic_set( \
      type *ptr, type value, int pe) {                      \
    return op(ptr, value, pe);                              \
  }

KOKKOS_REMOTESPACES_ATOMIC_SET(int, shmem_int_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned int, shmem_uint_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(long, shmem_long_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long, shmem_ulong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(long long, shmem_longlong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long long, shmem_ulonglong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(float, shmem_float_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(double, shmem_double_atomic_set)

#undef KOKKOS_REMOTESPACES_ATOMIC_SET

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH(type, op)                      \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_fetch(type *ptr, \
                                                             int pe) {  \
    return op(ptr, pe);                                                 \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH(int, shmem_int_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned int, shmem_uint_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long, shmem_long_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long, shmem_ulong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long long, shmem_longlong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long long,
                                 shmem_ulonglong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(float, shmem_float_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(double, shmem_double_atomic_fetch)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH

#define KOKKOS_REMOTESPACES_ATOMIC_ADD(type, op)            \
  static KOKKOS_INLINE_FUNCTION void shmem_type_atomic_add( \
      type *ptr, type value, int pe) {                      \
    return op(ptr, value, pe);                              \
  }

KOKKOS_REMOTESPACES_ATOMIC_ADD(int, shmem_int_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned int, shmem_uint_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long, shmem_long_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long, shmem_ulong_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long long, shmem_longlong_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long long, shmem_ulonglong_atomic_add)

#undef KOKKOS_REMOTESPACES_ATOMIC_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(type, op)            \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_fetch_add( \
      type *ptr, type value, int pe) {                            \
    return op(ptr, value, pe);                                    \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(int, shmem_int_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned int, shmem_uint_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long, shmem_long_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long,
                                     shmem_ulong_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long long, shmem_longlong_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long long,
                                     shmem_ulonglong_atomic_fetch_add)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(type, op)            \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_compare_swap( \
      type *ptr, type cond, type value, int pe) {                    \
    return op(ptr, cond, value, pe);                                 \
  }
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(int, shmem_int_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned int,
                                        shmem_uint_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long, shmem_long_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long,
                                        shmem_ulong_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long long,
                                        shmem_longlong_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long long,
                                        shmem_ulonglong_atomic_compare_swap)

#undef KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP

#define KOKKOS_REMOTESPACES_ATOMIC_SWAP(type, op)            \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_swap( \
      type *ptr, type value, int pe) {                       \
    return op(ptr, value, pe);                               \
  }
KOKKOS_REMOTESPACES_ATOMIC_SWAP(int, shmem_int_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned int, shmem_uint_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long, shmem_long_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long, shmem_ulong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long long, shmem_longlong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long long, shmem_ulonglong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(float, shmem_float_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(double, shmem_double_atomic_swap)

#undef KOKKOS_REMOTESPACES_ATOMIC_SWAP

template <class T, class Traits, typename Enable = void>
struct SHMEMDataElement {};

// Atomic Operators
template <class T, class Traits>
struct SHMEMDataElement<
    T, Traits,
    typename std::enable_if<Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  SHMEMDataElement(T *ptr_, int pe_, int i_) : ptr(ptr_ + i_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    shmem_type_atomic_set(ptr, val, pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T tmp;
    tmp = 1;
    shmem_type_atomic_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T tmp;
    tmp = 0 - 1;
    shmem_type_atomic_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    T tmp;
    tmp = 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    T tmp;
    tmp = 0 - 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    T tmp;
    tmp = 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    T tmp;
    tmp = 0 - 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type &val) const {
    return shmem_type_atomic_fetch_add(ptr, val, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type &val) const {
    T tmp;
    tmp = 0 - val;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp * val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp / val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp % val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp & val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp ^ val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp | val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp << val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp >> val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp % val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp;
  }
};

// Default Operators
template <class T, class Traits>
struct SHMEMDataElement<
    T, Traits,
    typename std::enable_if<!Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  SHMEMDataElement(T *ptr_, int pe_, int i_) : ptr(ptr_ + i_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    shmem_type_p(ptr, val, pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp++;
    shmem_type_p(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T tmp;
    shmem_type_g(ptr, pe);
    tmp--;
    shmem_type_p(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    T tmp;
    shmem_type_g(ptr, pe);
    tmp++;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp--;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp++;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp--;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp += val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp -= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp *= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp /= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp %= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp &= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp ^= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp |= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp <<= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp >>= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp % val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_SHMEM_OPS_HPP
