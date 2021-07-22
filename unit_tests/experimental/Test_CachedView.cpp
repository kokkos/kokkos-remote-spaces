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

// This unit test covers our use-case in CGSOLVE
// We do not test for puts as we do not have that capability in RACERlib yet.
template <class Data_t> void test_cached_view1D(int dim0) {
  int myRank;
  int numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  MPI_Barrier(MPI_COMM_WORLD);

  using ViewHost_1D_t =
      Kokkos::View<Data_t *, Kokkos::LayoutRight, HostSpace_t>;
  using ViewDevice_1D_t =
      Kokkos::View<Data_t *, Kokkos::LayoutRight, DeviceSpace_t>;                   
  using ViewRemote_1D_t =
      Kokkos::View<Data_t *, Kokkos::GlobalLayoutRight, RemoteSpace_t,
                   Kokkos::MemoryTraits<RemoteTraits::Cached>>;

  ViewRemote_1D_t v_r = ViewRemote_1D_t("RemoteView", dim0);
  ViewDevice_1D_t v_d = ViewDevice_1D_t(v_r.data(),v_r.extent(0));
  ViewDevice_1D_t v_d_out_1 = ViewDevice_1D_t("DataView", v_r.extent(0));
  ViewDevice_1D_t v_d_out_2 = ViewDevice_1D_t("DataView", v_r.extent(0));
  ViewDevice_1D_t v_d_out_3 = ViewDevice_1D_t("DataView", v_r.extent(0));
  ViewHost_1D_t v_h   = ViewHost_1D_t("HostView", v_r.extent(0));

  int next_rank = (myRank + 1) % numRanks;

  int num_teams = 3;
  int num_teams_adjusted = num_teams - 2;
  int elements_per_team = v_r.extent(0) / num_teams_adjusted;
  int elements_per_team_mod = v_r.extent(0) % num_teams_adjusted;
  int team_size = 1;
  int thread_vector_length = 1;
  int data_block = dim0 / numRanks;

  auto policy = Kokkos::TeamPolicy<>
       (num_teams, team_size, thread_vector_length);
  using team_t = Kokkos::TeamPolicy<>::member_type;

 Kokkos::parallel_for("Init", v_r.extent(0), KOKKOS_LAMBDA(const int i){
   v_d(i) = myRank * data_block + i;
  });

  Kokkos::fence(); 
  RemoteSpace_t().fence();

  Kokkos::Experimental::remote_parallel_for(
    "Increment", policy, KOKKOS_LAMBDA(const team_t& team) {
    int start = team.league_rank() * elements_per_team;
    int team_block = team.league_rank() == team.league_size()-1 ?  elements_per_team + elements_per_team_mod : elements_per_team;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,team_block),
        [&] (const int i) {
        int index = next_rank * data_block + start + i;        
        v_d_out_1(start+i) = v_r(index);    
        v_d_out_2(start+i) = v_r(index);    
        v_d_out_3(start+i) = v_r(index);    
      });
    }, v_r);

  Kokkos::fence(); 
  RemoteSpace_t().fence();

  Kokkos::deep_copy(v_h, v_d_out_1);
  for (int i = 0; i < data_block; ++i)
    ASSERT_EQ(v_h(i), next_rank * data_block + i);

  Kokkos::deep_copy(v_h, v_d_out_2);
  for (int i = 0; i <data_block; ++i)
    ASSERT_EQ(v_h(i), next_rank * data_block + i);

  Kokkos::deep_copy(v_h, v_d_out_3);
  for (int i = 0; i < data_block; ++i)
    ASSERT_EQ(v_h(i), next_rank * data_block + i);
}

TEST(TEST_CATEGORY, test_cached_view) {
  // 1D
  test_cached_view1D<double>(123456);
  test_cached_view1D<double>(654321); 
  test_cached_view1D<double>(12321);
}

#endif /* TEST_CACHED_VIEW_HPP */
