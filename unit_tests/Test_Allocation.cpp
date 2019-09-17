/*
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef TEST_ALLOCATION_HPP_
#define TEST_ALLOCATION_HPP_

#include<gtest/gtest.h>
#include<mpi.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_RemoteSpaces.hpp>

template<class ViewType>
void check_extents(ViewType view, int r) {
  int rank = view.rank;
  ASSERT_EQ(r,rank);
}

template<class ViewType, class ... Args>
void check_extents(ViewType view, int r, int N, Args...args) {
  if(r!=0)
  ASSERT_EQ(view.extent(r),N);
  check_extents(view,r+1,args...);
}

template<class DataType, class RemoteSpace, class ... Args>
void test_allocate_symmetric_remote_view_by_rank (Args...args) {

  int myRank,numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  MPI_Comm_size(MPI_COMM_WORLD,&numRanks);

  int* rank_list = new int[numRanks];
  for(int r=0; r<0; r++)
    rank_list[r] = r;
  typedef Kokkos::View<DataType, RemoteSpace> remote_view_type;
  remote_view_type view =
    Kokkos::allocate_symmetric_remote_view<remote_view_type>("MyView",numRanks,rank_list,args...);
  check_extents(view,0,numRanks,args...);
}

TEST( allocation, symmetric_view_by_rank ) {
  test_allocate_symmetric_remote_view_by_rank<double*,Kokkos::DefaultRemoteMemorySpace> ();
  test_allocate_symmetric_remote_view_by_rank<double**,Kokkos::DefaultRemoteMemorySpace> (113);
  test_allocate_symmetric_remote_view_by_rank<double***,Kokkos::DefaultRemoteMemorySpace> (7,5);
  test_allocate_symmetric_remote_view_by_rank<double****,Kokkos::DefaultRemoteMemorySpace> (9,10,7);
}



#endif /* TEST_ALLOCATION_HPP_ */
