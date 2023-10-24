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

#ifndef KOKKOS_REMOTESPACES_MPISPACE_BLOCK_OPS_HPP
#define KOKKOS_REMOTESPACES_MPISPACE_BLOCK_OPS_HPP

#include <shmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

#define KOKKOS_REMOTESPACES_PUT(type, mpi_type)                                \
  static KOKKOS_INLINE_FUNCTION void mpi_block_type_put(                       \
      const type *ptr, const size_t offset, const size_t nelems, const int pe, \
      const MPI_Win &win) {                                                    \
    assert(win != MPI_WIN_NULL);                                               \
    int _typesize;                                                             \
    MPI_Request request;                                                       \
    MPI_Type_size(mpi_type, &_typesize);                                       \
    const void *src_adr = ptr;                                                 \
    size_t win_offset   = sizeof(SharedAllocationHeader) + offset * _typesize; \
    MPI_Rput(src_adr, nelems, mpi_type, pe, win_offset, nelems, mpi_type, win, \
             &request);                                                        \
    MPI_Wait(&request, MPI_STATUS_IGNORE);                                     \
  }

KOKKOS_REMOTESPACES_PUT(char, MPI_SIGNED_CHAR)
KOKKOS_REMOTESPACES_PUT(unsigned char, MPI_UNSIGNED_CHAR)
KOKKOS_REMOTESPACES_PUT(short, MPI_SHORT)
KOKKOS_REMOTESPACES_PUT(unsigned short, MPI_UNSIGNED_SHORT)
KOKKOS_REMOTESPACES_PUT(int, MPI_INT)
KOKKOS_REMOTESPACES_PUT(unsigned int, MPI_UNSIGNED)
KOKKOS_REMOTESPACES_PUT(long, MPI_INT64_T)
KOKKOS_REMOTESPACES_PUT(long long, MPI_LONG_LONG)
KOKKOS_REMOTESPACES_PUT(unsigned long long, MPI_UNSIGNED_LONG_LONG)
KOKKOS_REMOTESPACES_PUT(unsigned long, MPI_UNSIGNED_LONG)
KOKKOS_REMOTESPACES_PUT(float, MPI_FLOAT)
KOKKOS_REMOTESPACES_PUT(double, MPI_DOUBLE)

#undef KOKKOS_REMOTESPACES_PUT

#define KOKKOS_REMOTESPACES_GET(type, mpi_type)                                \
  static KOKKOS_INLINE_FUNCTION void mpi_block_type_get(                       \
      type *ptr, const size_t offset, const size_t nelems, const int pe,       \
      const MPI_Win &win) {                                                    \
    assert(win != MPI_WIN_NULL);                                               \
    int _typesize;                                                             \
    MPI_Request request;                                                       \
    MPI_Type_size(mpi_type, &_typesize);                                       \
    void *dst_adr     = ptr;                                                   \
    size_t win_offset = sizeof(SharedAllocationHeader) + offset * _typesize;   \
    MPI_Rget(dst_adr, nelems, mpi_type, pe, win_offset, nelems, mpi_type, win, \
             &request);                                                        \
    MPI_Wait(&request, MPI_STATUS_IGNORE);                                     \
  }

KOKKOS_REMOTESPACES_GET(char, MPI_SIGNED_CHAR)
KOKKOS_REMOTESPACES_GET(unsigned char, MPI_UNSIGNED_CHAR)
KOKKOS_REMOTESPACES_GET(short, MPI_SHORT)
KOKKOS_REMOTESPACES_GET(unsigned short, MPI_UNSIGNED_SHORT)
KOKKOS_REMOTESPACES_GET(int, MPI_INT)
KOKKOS_REMOTESPACES_GET(unsigned int, MPI_UNSIGNED)
KOKKOS_REMOTESPACES_GET(long, MPI_INT64_T)
KOKKOS_REMOTESPACES_GET(long long, MPI_LONG_LONG)
KOKKOS_REMOTESPACES_GET(unsigned long long, MPI_UNSIGNED_LONG_LONG)
KOKKOS_REMOTESPACES_GET(unsigned long, MPI_UNSIGNED_LONG)
KOKKOS_REMOTESPACES_GET(float, MPI_FLOAT)
KOKKOS_REMOTESPACES_GET(double, MPI_DOUBLE)

#undef KOKKOS_REMOTESPACES_GET

template <class T, class Traits, typename Enable = void>
struct MPIBlockDataElement {};

template <class T, class Traits>
struct MPIBlockDataElement<T, Traits> {
  const MPI_Win win;
  T *ptr;
  int offset;
  int pe;
  size_t nelems;
  typedef const T const_value_type;
  typedef T non_const_value_type;

  KOKKOS_INLINE_FUNCTION
  MPIBlockDataElement(T *ptr_, MPI_Win win_, int pe_, size_t i_, size_t size_)
      : win(win_), ptr(ptr_), offset(i_), pe(pe_), nelems(size_) {}

  KOKKOS_INLINE_FUNCTION
  void put() const { mpi_block_type_put(ptr, offset, nelems, pe, win); }

  KOKKOS_INLINE_FUNCTION
  void get() const { mpi_block_type_get(ptr, offset, nelems, pe, win); }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_MPISPACE_BLOCK_OPS_HPP
