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

#ifndef KOKKOS_REMOTESPACES_NVSHMEM_VIEWTRAITS_HPP
#define KOKKOS_REMOTESPACES_NVSHMEM_VIEWTRAITS_HPP

namespace Kokkos {
/*
 * ViewTraits class evaluated during View specialization
 */

template <class... Properties>
struct ViewTraits<void, Kokkos::Experimental::NVSHMEMSpace, Properties...> {
  static_assert(
      std::is_same<typename ViewTraits<void, Properties...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Properties...>::memory_space,
                       void>::value &&
          std::is_same<
              typename ViewTraits<void, Properties...>::HostMirrorSpace,
              void>::value &&
          std::is_same<typename ViewTraits<void, Properties...>::array_layout,
                       void>::value,
      "Only one View Execution or Memory Space template argument");

  // Specify layout, keep subsequent space and memory traits arguments
  using execution_space =
      typename Kokkos::Experimental::NVSHMEMSpace::execution_space;
  using memory_space =
      typename Kokkos::Experimental::NVSHMEMSpace::memory_space;
  using HostMirrorSpace = typename Kokkos::Impl::HostMirror<
      Kokkos::Experimental::NVSHMEMSpace>::Space;
  using array_layout  = typename execution_space::array_layout;
  using specialize    = Kokkos::Experimental::RemoteSpaceSpecializeTag;
  using memory_traits = typename ViewTraits<void, Properties...>::memory_traits;
  using hooks_policy  = typename ViewTraits<void, Properties...>::hooks_policy;
};

}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_NVSHMEM_VIEWTRAITS_HPP
