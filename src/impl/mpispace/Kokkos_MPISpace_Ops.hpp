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

#ifndef KOKKOS_REMOTESPACES_MPI_OPS_HPP
#define KOKKOS_REMOTESPACES_MPI_OPS_HPP

#include <type_traits>

namespace Kokkos {
namespace Impl {

#define KOKKOS_REMOTESPACES_P(type, mpi_type)                                  \
  static KOKKOS_INLINE_FUNCTION void mpi_type_p(                               \
      const type val, const size_t offset, const int pe, const MPI_Win &win) { \
    assert(win != MPI_WIN_NULL);                                               \
    int _typesize;                                                             \
    MPI_Request request;                                                       \
    MPI_Type_size(mpi_type, &_typesize);                                       \
    MPI_Rput(&val, 1, mpi_type, pe,                                            \
             sizeof(SharedAllocationHeader) + offset * _typesize, 1, mpi_type,  \
             win, &request);                                                    \
    MPI_Wait(&request, MPI_STATUS_IGNORE);                                     \
  }

KOKKOS_REMOTESPACES_P(char, MPI_SIGNED_CHAR)
KOKKOS_REMOTESPACES_P(unsigned char, MPI_UNSIGNED_CHAR)
KOKKOS_REMOTESPACES_P(short, MPI_SHORT)
KOKKOS_REMOTESPACES_P(unsigned short, MPI_UNSIGNED_SHORT)
KOKKOS_REMOTESPACES_P(int, MPI_INT)
KOKKOS_REMOTESPACES_P(unsigned int, MPI_UNSIGNED)
KOKKOS_REMOTESPACES_P(long, MPI_INT64_T)
KOKKOS_REMOTESPACES_P(long long, MPI_LONG_LONG)
KOKKOS_REMOTESPACES_P(unsigned long long, MPI_UNSIGNED_LONG_LONG)
KOKKOS_REMOTESPACES_P(unsigned long, MPI_UNSIGNED_LONG)
KOKKOS_REMOTESPACES_P(float, MPI_FLOAT)
KOKKOS_REMOTESPACES_P(double, MPI_DOUBLE)

#undef KOKKOS_REMOTESPACES_P

#define KOKKOS_REMOTESPACES_G(type, mpi_type)                                 \
  static KOKKOS_INLINE_FUNCTION void mpi_type_g(                              \
      type &val, const size_t offset, const int pe, const MPI_Win &win) {     \
    assert(win != MPI_WIN_NULL);                                              \
    int _typesize;                                                            \
    MPI_Request request;                                                      \
    MPI_Type_size(mpi_type, &_typesize);                                      \
    MPI_Rget(&val, 1, mpi_type, pe,                                           \
             sizeof(SharedAllocationHeader) + offset * _typesize, 1, mpi_type, \
             win, &request);                                                   \
    MPI_Wait(&request, MPI_STATUS_IGNORE);                                    \
  }

KOKKOS_REMOTESPACES_G(char, MPI_SIGNED_CHAR)
KOKKOS_REMOTESPACES_G(unsigned char, MPI_UNSIGNED_CHAR)
KOKKOS_REMOTESPACES_G(short, MPI_SHORT)
KOKKOS_REMOTESPACES_G(unsigned short, MPI_UNSIGNED_SHORT)
KOKKOS_REMOTESPACES_G(int, MPI_INT)
KOKKOS_REMOTESPACES_G(unsigned int, MPI_UNSIGNED)
KOKKOS_REMOTESPACES_G(long, MPI_INT64_T)
KOKKOS_REMOTESPACES_G(long long, MPI_LONG_LONG)
KOKKOS_REMOTESPACES_G(unsigned long long, MPI_UNSIGNED_LONG_LONG)
KOKKOS_REMOTESPACES_G(unsigned long, MPI_UNSIGNED_LONG)
KOKKOS_REMOTESPACES_G(float, MPI_FLOAT)
KOKKOS_REMOTESPACES_G(double, MPI_DOUBLE)
#undef KOKKOS_REMOTESPACES_G

#define KOKKOS_REMOTESPACES_ATOMIC_SET(type)                                   \
  static KOKKOS_INLINE_FUNCTION void mpi_type_atomic_set(type *ptr,            \
                                                         type value, int pe) { \
    /* TBD */                                                                  \
  }

KOKKOS_REMOTESPACES_ATOMIC_SET(int)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned int)
KOKKOS_REMOTESPACES_ATOMIC_SET(long)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long)
KOKKOS_REMOTESPACES_ATOMIC_SET(long long)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long long)
KOKKOS_REMOTESPACES_ATOMIC_SET(float)
KOKKOS_REMOTESPACES_ATOMIC_SET(double)

#undef KOKKOS_REMOTESPACES_ATOMIC_SET

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH(type)                        \
  static KOKKOS_INLINE_FUNCTION type mpi_type_atomic_fetch(type *ptr, \
                                                           int pe) {  \
    /* TBD */                                                         \
    return 0;                                                         \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH(int)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned int)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(float)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(double)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH

#define KOKKOS_REMOTESPACES_ATOMIC_ADD(type)                                   \
  static KOKKOS_INLINE_FUNCTION void mpi_type_atomic_add(type *ptr,            \
                                                         type value, int pe) { \
    /* TBD */                                                                  \
  }

KOKKOS_REMOTESPACES_ATOMIC_ADD(int)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned int)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long long)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long long)

#undef KOKKOS_REMOTESPACES_ATOMIC_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(type)              \
  static KOKKOS_INLINE_FUNCTION type mpi_type_atomic_fetch_add( \
      type *ptr, type value, int pe) {                          \
    /* TBD */                                                   \
    return 0;                                                   \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(int)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned int)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long long)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long long)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(type)              \
  static KOKKOS_INLINE_FUNCTION type mpi_type_atomic_compare_swap( \
      type *ptr, type cond, type value, int pe) {                  \
    /* TBD */                                                      \
    return 0;                                                      \
  }
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(int)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned int)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long long)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long long)

#undef KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP

#define KOKKOS_REMOTESPACES_ATOMIC_SWAP(type)              \
  static KOKKOS_INLINE_FUNCTION type mpi_type_atomic_swap( \
      type *ptr, type value, int pe) {                     \
    /* TBD */                                              \
    return 0;                                              \
  }
KOKKOS_REMOTESPACES_ATOMIC_SWAP(int)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned int)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long long)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long long)

#undef KOKKOS_REMOTESPACES_ATOMIC_SWAP

template <class T, class Traits, typename Enable = void>
struct MPIDataElement {};

// Atomic Operators
template <class T, class Traits>
struct MPIDataElement<
    T, Traits,
    typename std::enable_if<!Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  const MPI_Win *win;
  int offset;
  int pe;

  KOKKOS_INLINE_FUNCTION
  MPIDataElement(MPI_Win *win_, int pe_, int i_)
      : win(win_), offset(i_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val++;
    mpi_type_p(val, offset, pe, *win);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val--;
    mpi_type_p(val, offset, pe, *win);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val++;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val--;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val++;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val--;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp += val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp -= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp *= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp /= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp %= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp &= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp ^= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp |= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp <<= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp >>= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp % val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp;
  }
};

// Default Operators
template <class T, class Traits>
struct MPIDataElement<
    T, Traits,
    typename std::enable_if<Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  const MPI_Win *win;
  int offset;
  int pe;

  KOKKOS_INLINE_FUNCTION
  MPIDataElement(MPI_Win *win_, int pe_, int i_)
      : win(win_), offset(i_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val++;
    mpi_type_p(val, offset, pe, *win);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val--;
    mpi_type_p(val, offset, pe, *win);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val++;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val--;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val++;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    T val = T();
    mpi_type_g(val, offset, pe, *win);
    val--;
    mpi_type_p(val, offset, pe, *win);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp += val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp -= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp *= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp /= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp %= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp &= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp ^= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp |= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp <<= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    tmp >>= val;
    mpi_type_p(tmp, offset, pe, *win);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp % val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    T tmp = T();
    mpi_type_g(tmp, offset, pe, *win);
    return tmp;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_MPI_OPS_HPP
