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

#ifndef TEST_LOCAL_DEEP_COPY_HPP_
#define TEST_LOCAL_DEEP_COPY_HPP_

#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t, class Space_A, class Space_B >
void test_localdeepcopy(
  typename std::enable_if<(
    std::is_same<Space_A, Kokkos::HostSpace>::value &&
    std::is_same<Space_B, RemoteSpace_t>::value)>
    ::type* = nullptr )
{
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_t = Kokkos::View<Data_t**, Space_A>;
  using ViewRemote_t = Kokkos::View<Data_t**, Space_B>;
  using TeamPolicy_t =  Kokkos::TeamPolicy<>;
  
  
  ViewHost_t v_H ("HostView",1,1);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, 1);
  ViewRemote_t v_R_cpy = ViewRemote_t("RemoteView", num_ranks, 1);

  Kokkos::parallel_for(
    "Team", TeamPolicy_t(1,Kokkos::AUTO), KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
      Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, 1),
              [&](const int i) {
                v_R(my_rank,0) = 0x123;
              });
      
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        Kokkos::Experimental::local_deep_copy(team, v_R_cpy, v_R);
       });
  });
  
  Kokkos::Experimental::deep_copy(v_H, v_R_cpy); 
  ASSERT_EQ(0x123, v_H(0,0));
}

template <class Data_t, class Space_A, class Space_B >
void test_localdeepcopy(int i1,
  typename std::enable_if<(
    std::is_same<Space_A, Kokkos::HostSpace>::value &&
    std::is_same<Space_B, RemoteSpace_t>::value)>
    ::type* = nullptr )
{
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_t = Kokkos::View<Data_t**, Space_A>;
  using ViewRemote_t = Kokkos::View<Data_t**, Space_B>;
  using TeamPolicy_t =  Kokkos::TeamPolicy<>;
  
  
  ViewHost_t v_H ("HostView",1,i1);

  ViewRemote_t v_R = ViewRemote_t("RemoteView", num_ranks, i1);
  ViewRemote_t v_R_cpy = ViewRemote_t("RemoteView", num_ranks, i1);


  Kokkos::parallel_for(
    "Team", TeamPolicy_t(1,Kokkos::AUTO), KOKKOS_LAMBDA(typename TeamPolicy_t::member_type team) {
      Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, 1),
              [&](const int i) {
                for(int  j = 0; j<i1; ++j)
                v_R(my_rank,j) = 0x123;
              });
      
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        Kokkos::Experimental::local_deep_copy(team, v_R_cpy, v_R);
       });
  });
  
  Kokkos::Experimental::deep_copy(v_H, v_R_cpy); 
  for(int  j = 0; j<i1; ++j)
    ASSERT_EQ(0x123, v_H(0,j));
}


TEST(TEST_CATEGORY, test_localdeepcopy) {
  //Scalar
  test_localdeepcopy<int, Kokkos::HostSpace, RemoteSpace_t>();
 /* test_localdeepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>();
  test_localdeepcopy<double, Kokkos::HostSpace, RemoteSpace_t>();*/

  // 1D
  test_localdeepcopy<int, Kokkos::HostSpace, RemoteSpace_t>(50);
 /* test_localdeepcopy<int64_t, Kokkos::HostSpace, RemoteSpace_t>(150);
  test_localdeepcopy<double, Kokkos::HostSpace, RemoteSpace_t>(1500);*/
}

#endif /* TEST_LOCAL_DEEP_COPY_HPP_ */