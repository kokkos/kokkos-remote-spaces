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

#ifndef KOKKOS_REMOTESPACES_NVSHMEM_OPS_HPP
#define KOKKOS_REMOTESPACES_NVSHMEM_OPS_HPP

#include <nvshmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
#define KOKKOS_REMOTESPACES_P(type, op)                                        \
  static KOKKOS_INLINE_FUNCTION void shmem_type_p(type *ptr, const type &val,  \
                                                  int pe) {                    \
    op(ptr, val, pe);                                                          \
  }

KOKKOS_REMOTESPACES_P(char, nvshmem_char_p)
KOKKOS_REMOTESPACES_P(unsigned char, nvshmem_uchar_p)
KOKKOS_REMOTESPACES_P(short, nvshmem_short_p)
KOKKOS_REMOTESPACES_P(unsigned short, nvshmem_ushort_p)
KOKKOS_REMOTESPACES_P(int, nvshmem_int_p)
KOKKOS_REMOTESPACES_P(unsigned int, nvshmem_uint_p)
KOKKOS_REMOTESPACES_P(long, nvshmem_long_p)
KOKKOS_REMOTESPACES_P(unsigned long, nvshmem_ulong_p)
KOKKOS_REMOTESPACES_P(long long, nvshmem_longlong_p)
KOKKOS_REMOTESPACES_P(unsigned long long, nvshmem_ulonglong_p)
KOKKOS_REMOTESPACES_P(float, nvshmem_float_p)
KOKKOS_REMOTESPACES_P(double, nvshmem_double_p)

#undef KOKKOS_REMOTESPACES_P

#define KOKKOS_REMOTESPACES_G(type, op)                                        \
  static KOKKOS_INLINE_FUNCTION type shmem_type_g(type *ptr, int pe) {         \
    return op(ptr, pe);                                                        \
  }

KOKKOS_REMOTESPACES_G(char, nvshmem_char_g)
KOKKOS_REMOTESPACES_G(unsigned char, nvshmem_uchar_g)
KOKKOS_REMOTESPACES_G(short, nvshmem_short_g)
KOKKOS_REMOTESPACES_G(unsigned short, nvshmem_ushort_g)
KOKKOS_REMOTESPACES_G(int, nvshmem_int_g)
KOKKOS_REMOTESPACES_G(unsigned int, nvshmem_uint_g)
KOKKOS_REMOTESPACES_G(long, nvshmem_long_g)
KOKKOS_REMOTESPACES_G(unsigned long, nvshmem_ulong_g)
KOKKOS_REMOTESPACES_G(long long, nvshmem_longlong_g)
KOKKOS_REMOTESPACES_G(unsigned long long, nvshmem_ulonglong_g)
KOKKOS_REMOTESPACES_G(float, nvshmem_float_g)
KOKKOS_REMOTESPACES_G(double, nvshmem_double_g)

#undef KOKKOS_REMOTESPACES_G

#define KOKKOS_REMOTESPACES_ATOMIC_SET(type, op)                               \
  static KOKKOS_INLINE_FUNCTION void shmem_type_atomic_set(                    \
      type *ptr, type value, int pe) {                                         \
    return op(ptr, value, pe);                                                 \
  }

KOKKOS_REMOTESPACES_ATOMIC_SET(int, nvshmem_int_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned int, nvshmem_uint_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(long, nvshmem_long_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long, nvshmem_ulong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(long long, nvshmem_longlong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long long, nvshmem_ulonglong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(float, nvshmem_float_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(double, nvshmem_double_atomic_set)

#undef KOKKOS_REMOTESPACES_ATOMIC_SET

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH(type, op)                             \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_fetch(type *ptr,        \
                                                             int pe) {         \
    return op(ptr, pe);                                                        \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH(int, nvshmem_int_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned int, nvshmem_uint_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long, nvshmem_long_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long, nvshmem_ulong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long long, nvshmem_longlong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long long,
                                 nvshmem_ulonglong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(float, nvshmem_float_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(double, nvshmem_double_atomic_fetch)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH

#define KOKKOS_REMOTESPACES_ATOMIC_ADD(type, op)                               \
  static KOKKOS_INLINE_FUNCTION void shmem_type_atomic_add(                    \
      type *ptr, type value, int pe) {                                         \
    return op(ptr, value, pe);                                                 \
  }

KOKKOS_REMOTESPACES_ATOMIC_ADD(int, nvshmem_int_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned int, nvshmem_uint_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long, nvshmem_long_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long, nvshmem_ulong_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long long, nvshmem_longlong_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long long, nvshmem_ulonglong_atomic_add)

#undef KOKKOS_REMOTESPACES_ATOMIC_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(type, op)                         \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_fetch_add(              \
      type *ptr, type value, int pe) {                                         \
    return op(ptr, value, pe);                                                 \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(int, nvshmem_int_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned int,
                                     nvshmem_uint_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long, nvshmem_long_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long,
                                     nvshmem_ulong_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long long,
                                     nvshmem_longlong_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long long,
                                     nvshmem_ulonglong_atomic_fetch_add)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(type, op)                      \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_compare_swap(           \
      type *ptr, type cond, type value, int pe) {                              \
    return op(ptr, cond, value, pe);                                           \
  }
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(int, nvshmem_int_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned int,
                                        nvshmem_uint_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long, nvshmem_long_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long,
                                        nvshmem_ulong_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long long,
                                        nvshmem_longlong_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long long,
                                        nvshmem_ulonglong_atomic_compare_swap)

#undef KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP

#define KOKKOS_REMOTESPACES_ATOMIC_SWAP(type, op)                              \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_swap(                   \
      type *ptr, type value, int pe) {                                         \
    return op(ptr, value, pe);                                                 \
  }
KOKKOS_REMOTESPACES_ATOMIC_SWAP(int, nvshmem_int_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned int, nvshmem_uint_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long, nvshmem_long_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long, nvshmem_ulong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long long, nvshmem_longlong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long long,
                                nvshmem_ulonglong_atomic_swap)

#undef KOKKOS_REMOTESPACES_ATOMIC_SWAP

#else // KOKKOS_ENABLE_NVSHMEMSPACE

#define KOKKOS_REMOTESPACES_P(type, p[])                                       \
  static inline void shmem_type_p(type *ptr, const type &val, int pe) {        \
    *ptr = val;                                                                \
  }

#define KOKKOS_REMOTESPACES_G(type, func)                                      \
  static inline type shmem_type_g(type *ptr, int pe) { return *ptr; }          \

#define KOKKOS_REMOTESPACES_ATOMIC_SET(type, op)                               \
static inline void shmem_type_atomic_set(type *ptr, type value, int pe) {      \
  *ptr = value;                                                                \
}

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH(type, op)                             \
static inline type shmem_type_atomic_fetch(type *ptr, int pe) { return *ptr; } 

#define KOKKOS_REMOTESPACES_ATOMIC_ADD(type, op)                               \
static inline void shmem_type_atomic_add(type *ptr, type value, int pe) {      \
  *ptr += value;                                                               \
}

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(type, op)                         \
static inline type shmem_type_atomic_fetch_add(type *ptr, type value, int pe) {\
  T tmp = *ptr;                                                                \
  *ptr += value;                                                               \
  return tmp;                                                                  \
}

#define KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(type, op)                      \
static inline type shmem_type_atomic_compare_swap(type *ptr, type cond,        \
                                                  type value, int pe) {        \
  if (cond == *ptr) {                                                          \
    type tmp = *ptr;                                                           \
    *ptr = value;                                                              \
    return tmp;                                                                \
  }                                                                            \
  return *ptr;                                                                 \
}

#define KOKKOS_REMOTESPACES_ATOMIC_SWAP(type, op)                              \
static inline type shmem_type_atomic_swap(type *ptr, type value, int pe) {     \
  type tmp = *ptr;                                                             \
  *ptr = value;                                                                \
  return tmp;                                                                  \
}

#undef KOKKOS_REMOTESPACES_P
#undef KOKKOS_REMOTESPACES_G
#undef KOKKOS_REMOTESPACES_ATOMIC_SET
#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH
#undef KOKKOS_REMOTESPACES_ATOMIC_ADD
#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD
#undef KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP
#undef KOKKOS_REMOTESPACES_ATOMIC_SWAP

#endif
template <class T, class Traits, typename Enable = void>
struct NVSHMEMDataElement {};

// Atomic Operators
template <class T, class Traits>
struct NVSHMEMDataElement<
    T, Traits,
    typename std::enable_if<Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataElement(T *ptr_, int pe_, int i_) : ptr(ptr_ + i_), pe(pe_) {}

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
      tmp = oldval;
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
      tmp = oldval;
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
      tmp = oldval;
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
      tmp = oldval;
      newval = tmp & val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp = oldval;
      newval = tmp ^ val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp = oldval;
      newval = tmp | val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp = oldval;
      newval = tmp << val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp = oldval;
      newval = tmp >> val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
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
    tmp = shmem_type_g(ptr, pe);
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
struct NVSHMEMDataElement<
    T, Traits,
    typename std::enable_if<!Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  NVSHMEMDataElement(T *ptr_, int pe_, int i_) : ptr(ptr_ + i_), pe(pe_) {}

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

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_REMOTESPACES_NVSHMEM_OPS_HPP