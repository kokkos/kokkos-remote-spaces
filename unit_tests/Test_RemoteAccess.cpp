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

#ifndef TEST_REMOTE_ACCESS_HPP_
#define TEST_REMOTE_ACCESS_HPP_

#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>

using RemoteSpace = Kokkos::Experimental::DefaultRemoteMemorySpace;


template <class Data_t, class Space_t >
void test_remote_accesses(int size)
{
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using TeamPolicy = Kokkos::TeamPolicy<>;
  TeamPolicy policy = TeamPolicy(1, Kokkos::AUTO);

  using RemoteView_t = Kokkos::View<Data_t**, Space_t>;
  using HostSpace_t = Kokkos::View<Data_t**, Kokkos::HostSpace> ;

  // Allocate remote view
  RemoteView_t v_R = RemoteView_t("RemoteView", num_ranks, size);

  Kokkos::parallel_for(
  policy, KOKKOS_LAMBDA(const TeamPolicy::member_type &team) {
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, size),
      [&](const int i) {
        v_R(num_ranks-my_rank-1, i) = (Data_t) my_rank * size + i;
      });
  });

  RemoteSpace().fence();

  HostSpace_t v_H ("HostView",1,size);
  Kokkos::Experimental::deep_copy(v_H, v_R); 

  Data_t check (0), ref(0);
  for (int i=0; i<size; i++) {
    check += v_H(0,i);
    ref += (num_ranks - my_rank - 1) * size + i;
  }
  ASSERT_EQ(check, ref);
}

TEST(TEST_CATEGORY, test_remote_accesses) {
  test_remote_accesses<int, RemoteSpace>(12345);
  test_remote_accesses<int64_t, RemoteSpace>(4567);
  test_remote_accesses<double, RemoteSpace>(89);
}

#endif /* TEST_REMOTE_ACCESS_HPP_ */