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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <Kokkos_MPISpace.hpp>
#include <mpi.h>

namespace Kokkos {
namespace Experimental {

MPI_Win MPISpace::current_win;
std::vector<MPI_Win> MPISpace::mpi_windows;

/* Default allocation mechanism */
MPISpace::MPISpace() : allocation_mode(Symmetric) {}

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
    if (allocation_mode == Symmetric) {
      current_win = MPI_WIN_NULL;
      MPI_Win_allocate(arg_alloc_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &ptr,
                       &current_win);
      int i = -1;
      for (i = 0; i < mpi_windows.size(); i++)
        if (mpi_windows[i] == MPI_WIN_NULL)
          break;
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
  int last_valid = -1;
  for (last_valid = 0; last_valid < mpi_windows.size(); last_valid++)
    if (mpi_windows[last_valid] == MPI_WIN_NULL)
      break;
  last_valid--;
  for (int i = 0; i < mpi_windows.size(); i++)
    if (mpi_windows[i] == current_win) {
      mpi_windows[i] = mpi_windows[last_valid];
      mpi_windows[last_valid] = MPI_WIN_NULL;
      break;
    }

  MPI_Win_free(&current_win);
  current_win = MPI_WIN_NULL;
}

void MPISpace::fence() {

  for (int i = 0; i < mpi_windows.size(); i++)
    if (mpi_windows[i] != MPI_WIN_NULL)
      MPI_Win_fence(MPI_MODE_NOPRECEDE, mpi_windows[i]);
    else
      break;
}
} // namespace Experimental

namespace Impl
{
  Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::MPISpace, Kokkos::Experimental::RemoteSpaceSpecializeTag>
  ::DeepCopy(void *dst, const void *src, size_t n) {
      memcpy(dst, src, n);
  }

  Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace, HostSpace, Kokkos::Experimental::RemoteSpaceSpecializeTag>
  ::DeepCopy(void *dst, const void *src, size_t n) {
      memcpy(dst, src, n);
  }
}

} // namespace Kokkos


namespace Kokkos {

namespace Impl {

SharedAllocationRecord<void, void>
    SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::s_root_record;

void SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::deallocate(
    SharedAllocationRecord<void, void> *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

SharedAllocationRecord<Kokkos::Experimental::MPISpace,
                       void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
    /*Kokkos::Impl::DeepCopy<CudaSpace, HostSpace>(
    &header, RecordBase::m_alloc_ptr, sizeof(SharedAllocationHeader));*/

    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::MPISpace::name()),
        header.m_label, data(), size());
  }
#endif
  m_space.current_win = win;
  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::
    SharedAllocationRecord(
        const Kokkos::Experimental::MPISpace &arg_space,
        const std::string &arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
          &SharedAllocationRecord<Kokkos::Experimental::MPISpace,
                                  void>::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader *>(arg_space.allocate(
              sizeof(SharedAllocationHeader) + arg_alloc_size)),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record =
      static_cast<SharedAllocationRecord<void, void> *>(this);
  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  win = m_space.current_win;
}

//----------------------------------------------------------------------------

void *
SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::allocate_tracked(
    const Kokkos::Experimental::MPISpace &arg_space,
    const std::string &arg_alloc_label, const size_t arg_alloc_size) {
  if (!arg_alloc_size)
    return (void *)0;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);
  return r->data();
}

void SharedAllocationRecord<Kokkos::Experimental::MPISpace,
                            void>::deallocate_tracked(void *const
                                                          arg_alloc_ptr) {
  if (arg_alloc_ptr != 0) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *
SharedAllocationRecord<Kokkos::Experimental::MPISpace,
                       void>::reallocate_tracked(void *const arg_alloc_ptr,
                                                 const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                         Kokkos::Experimental::MPISpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

SharedAllocationRecord<Kokkos::Experimental::MPISpace, void> *
SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::get_record(
    void *alloc_ptr) {
  typedef SharedAllocationHeader Header;
  typedef SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>
      RecordHost;

  SharedAllocationHeader const *const head =
      alloc_ptr ? Header::get_header(alloc_ptr) : (SharedAllocationHeader *)0;
  RecordHost *const record =
      head ? static_cast<RecordHost *>(head->m_record) : (RecordHost *)0;

  if (!alloc_ptr || record->m_alloc_ptr != head) {
    Kokkos::Impl::throw_runtime_exception(std::string(
        "Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::MPISpace "
        ", void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<Kokkos::Experimental::MPISpace, void>::
    print_records(std::ostream &s, const Kokkos::Experimental::MPISpace &,
                  bool detail) {
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "MPISpace", &s_root_record, detail);
}

} // namespace Impl
} // namespace Kokkos
