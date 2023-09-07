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

#ifndef KOKKOS_REMOTESPACES_ERROR_HPP
#define KOKKOS_REMOTESPACES_ERROR_HPP

namespace Kokkos {
namespace Impl {
namespace Experimental {

class RemoteSpacesMemoryAllocationFailure : public std::bad_alloc {
 public:
  enum class FailureMode {
    OutOfMemoryError,
    AllocationNotAligned,
    InvalidAllocationSize,
    Unknown
  };
  enum class AllocationMechanism {
    SHMEMMALLOCDEFAULT,
    SHMEMMALLOC,
    NVSHMEMMALLOC,
    ROCSHMEMMALLOC,
    MPIWINALLOC
  };

 private:
  size_t m_attempted_size;
  size_t m_attempted_alignment;
  FailureMode m_failure_mode;
  AllocationMechanism m_mechanism;

 public:
  RemoteSpacesMemoryAllocationFailure(
      size_t arg_attempted_size, size_t arg_attempted_alignment,
      FailureMode arg_failure_mode = FailureMode::OutOfMemoryError,
      AllocationMechanism arg_mechanism =
          AllocationMechanism::SHMEMMALLOCDEFAULT) noexcept
      : m_attempted_size(arg_attempted_size),
        m_attempted_alignment(arg_attempted_alignment),
        m_failure_mode(arg_failure_mode),
        m_mechanism(arg_mechanism) {}

  RemoteSpacesMemoryAllocationFailure() noexcept = delete;

  RemoteSpacesMemoryAllocationFailure(
      RemoteSpacesMemoryAllocationFailure const &) noexcept = default;
  RemoteSpacesMemoryAllocationFailure(
      RemoteSpacesMemoryAllocationFailure &&) noexcept = default;

  RemoteSpacesMemoryAllocationFailure &operator             =(
      RemoteSpacesMemoryAllocationFailure const &) noexcept = default;
  RemoteSpacesMemoryAllocationFailure &operator             =(
      RemoteSpacesMemoryAllocationFailure &&) noexcept = default;

  ~RemoteSpacesMemoryAllocationFailure() noexcept override = default;

  [[nodiscard]] const char *what() const noexcept override {
    if (m_failure_mode == FailureMode::OutOfMemoryError) {
      return "Memory allocation error: out of memory";
    } else if (m_failure_mode == FailureMode::AllocationNotAligned) {
      return "Memory allocation error: allocation result was under-aligned";
    }

    return nullptr;  // unreachable
  }

  [[nodiscard]] size_t attempted_size() const noexcept {
    return m_attempted_size;
  }

  [[nodiscard]] size_t attempted_alignment() const noexcept {
    return m_attempted_alignment;
  }

  [[nodiscard]] AllocationMechanism allocation_mechanism() const noexcept {
    return m_mechanism;
  }

  [[nodiscard]] FailureMode failure_mode() const noexcept {
    return m_failure_mode;
  }

  void print_error_message(std::ostream &o) const {
    o << "Allocation of size " << Impl::human_memory_size(m_attempted_size);
    o << " failed";
    switch (m_failure_mode) {
      case FailureMode::OutOfMemoryError:
        o << ", likely due to insufficient memory.";
        break;
      case FailureMode::AllocationNotAligned:
        o << " because the allocation was improperly aligned.";
        break;
      case FailureMode::InvalidAllocationSize:
        o << " because the requested allocation size is not a valid size for "
             "the"
             " requested allocation mechanism (it's probably too large).";
        break;
      case FailureMode::Unknown: o << " because of an unknown error.";
    }
    o << "  (The allocation mechanism was ";
    switch (m_mechanism) {
      case AllocationMechanism::SHMEMMALLOC: o << "shmem_malloc()."; break;
      case AllocationMechanism::NVSHMEMMALLOC: o << "nvshmem_malloc()."; break;
      case AllocationMechanism::ROCSHMEMMALLOC:
        o << "rocshmem_malloc().";
        break;
      case AllocationMechanism::MPIWINALLOC: o << "MPI_Win_allocate().";
    }
    append_additional_error_information(o);
    o << ")" << std::endl;
  }

  [[nodiscard]] std::string get_error_message() const {
    std::ostringstream out;
    print_error_message(out);
    return out.str();
  }

  virtual void append_additional_error_information(std::ostream &) const {}
};

}  // namespace Experimental
}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_ERROR_HPP