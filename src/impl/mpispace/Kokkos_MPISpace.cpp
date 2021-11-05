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

#include <Kokkos_Core.hpp>
#include <Kokkos_MPISpace.hpp>
#include <csignal>
#include <mpi.h>

namespace Kokkos {
namespace Experimental {

MPI_Win MPISpace::current_win;
std::vector<MPI_Win> MPISpace::mpi_windows;

/* Default allocation mechanism */
MPISpace::MPISpace() : allocation_mode(Kokkos::Experimental::Symmetric) {}

void MPISpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void MPISpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *MPISpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  void *ptr = 0;
  if (arg_alloc_size) {
    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      current_win = MPI_WIN_NULL;
      MPI_Win_allocate(arg_alloc_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &ptr,
                       &current_win);
                             
      assert(current_win != MPI_WIN_NULL);
      
      int ret = MPI_Win_lock_all(MPI_MODE_NOCHECK, current_win);
      if (ret != MPI_SUCCESS) {
        Kokkos::abort("MPI window lock all failed.");
      }
      int i;
      for (i = 0; i < mpi_windows.size(); ++i) {
        if (mpi_windows[i] == MPI_WIN_NULL)
          break;
      }

      if (i == mpi_windows.size())
        mpi_windows.push_back(current_win);
      else
        mpi_windows[i] = current_win;
    } else {
      Kokkos::abort("MPISpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}

void MPISpace::deallocate(void *const, const size_t) const {
  int last_valid;
  for (last_valid = 0; last_valid < mpi_windows.size(); ++last_valid){
    if (mpi_windows[last_valid] == MPI_WIN_NULL)
      break;
  }

  last_valid--;
  for (int i = 0; i < mpi_windows.size(); ++i)
  {
    if (mpi_windows[i] == current_win) {
      mpi_windows[i] = mpi_windows[last_valid];
      mpi_windows[last_valid] = MPI_WIN_NULL;
      break;
    }
  }

  assert(current_win != MPI_WIN_NULL);
  MPI_Win_unlock_all(current_win);
  MPI_Win_free(&current_win);
  
  // We pass a mempory space instance do multiple Views thus 
  // setting "current_win = MPI_WIN_NULL;" will result in a wrong handle if
  // subsequent view runs out of scope
  // Fixme: The following only works when views are allocated sequentially
  // We need a thread-safe map to associate views and windows

  if(last_valid != 0)
    current_win = mpi_windows[last_valid - 1];
  else
    current_win = MPI_WIN_NULL;
}

void MPISpace::fence() {
  for (int i = 0; i < mpi_windows.size(); i++) {
    if (mpi_windows[i] != MPI_WIN_NULL) {
      MPI_Win_flush_all(mpi_windows[i]);
    } else {
      break;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

size_t get_num_pes() {
  int n_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  return n_ranks;
}

size_t get_my_pe() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

KOKKOS_FUNCTION
size_t get_indexing_block_size(size_t size) {
  size_t num_pes, block;
  num_pes = get_num_pes();
  block = (size + num_pes - 1) / num_pes;
  return block;
}

std::pair<size_t, size_t> getRange(size_t size, size_t pe)
{
    size_t start, end; 
    size_t block = get_indexing_block_size(size);
    start  = pe * block;
    end = (pe + 1) * block;

    size_t num_pes = get_num_pes();

    if(size<num_pes)
    {
      size_t diff = (num_pes * block) - size;
      if(pe > num_pes - 1 - diff)
        end --;
    } else
    {   
      if (pe == num_pes - 1){
        size_t diff = size - (num_pes - 1) * block;
        end = start + diff;
      }
      end--;
    }
    return std::make_pair(start, end);
}
} // namespace Experimental

namespace Impl {

Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::MPISpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace, HostSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy((char*)dst, (char*)src, n);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                       Kokkos::Experimental::MPISpace>::DeepCopy(void *dst,
                                                                 const void
                                                                     *src,
                                                                 size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                       Kokkos::Experimental::MPISpace,
                       ExecutionSpace>::DeepCopy(void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                       Kokkos::Experimental::MPISpace,
                       ExecutionSpace>::DeepCopy(const ExecutionSpace &exec,
                                                 void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_get(void *dst, const void *src, size_t pe, size_t n) {
  // TBD
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_put(void *dst, const void *src, size_t pe, size_t n) {
  // TBD
}

} // namespace Impl
} // namespace Kokkos
