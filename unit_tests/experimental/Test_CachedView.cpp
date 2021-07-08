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

#ifndef TEST_CACHED_VIEW_HPP
#define TEST_CACHED_VIEW_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using DeviceSpace_t = Kokkos::CudaSpace;
using HostSpace_t = Kokkos::HostSpace;

using RemoteTraits = Kokkos::RemoteSpaces_MemoryTraitsFlags;

template <class Data_t> void test_cached_view1D(int dim0) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int num_teams = 3;
  int team_size = 1;
  int thread_vector_length = 1;

  using ViewHost_1D_t =
      Kokkos::View<Data_t *, Kokkos::LayoutRight, HostSpace_t>;
  using ViewDevice_1D_t =
      Kokkos::View<Data_t *, Kokkos::LayoutRight, DeviceSpace_t>;                   
  using ViewRemote_1D_t =
      Kokkos::View<Data_t *, Kokkos::GlobalLayoutRight, RemoteSpace_t,
                   Kokkos::MemoryTraits<RemoteTraits::Cached>>;

  ViewRemote_1D_t v_r = ViewRemote_1D_t("RemoteView", dim0);
  ViewDevice_1D_t v_d = ViewDevice_1D_t(v_r.data(),v_r.extent(0));
  ViewDevice_1D_t v_d_out = ViewDevice_1D_t("DataView", v_r.extent(0));
  ViewHost_1D_t v_h   = ViewHost_1D_t("HostView", v_r.extent(0));

  printf("v_r extent: %i, %i, %p\n", v_r.extent(0), dim0, v_r.data());

  // Init
  // for (int i = 0; i < v_h.extent(0); ++i)
  //  v_h(i) = my_rank * v_h.extent(0) + i;
  //Kokkos::Experimental::deep_copy(v_r, v_h);

  int next_rank = (my_rank + 1) % num_ranks;

  auto policy = Kokkos::TeamPolicy<>
       (num_teams, team_size, thread_vector_length);
  using team_t = Kokkos::TeamPolicy<>::member_type;

 Kokkos::parallel_for("Init",  v_r.extent(0), KOKKOS_LAMBDA(const int i){
   v_d(i) = my_rank * v_r.extent(0) + i;
  });

  RemoteSpace_t().fence();

  Kokkos::Experimental::remote_parallel_for(
    "Increment", policy, KOKKOS_LAMBDA(const team_t& team) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,v_r.extent(0)),
        [&] (const int i) {
        int index = next_rank * v_r.extent(0) + i;
        v_d_out(i) = v_r(index);    
      });
    }, v_r);

  RemoteSpace_t().fence();
  Kokkos::deep_copy(v_h, v_d_out);

  for (int i = 0; i < dim0 / num_ranks; ++i)
    ASSERT_EQ(v_h(i), next_rank * v_r.extent(0) + i);
}

TEST(TEST_CATEGORY, test_cached_view) {
   // 1D
  test_cached_view1D<int>(1024);
}

#endif /* TEST_ATOMIC_GLOBALVIEW_HPP */
