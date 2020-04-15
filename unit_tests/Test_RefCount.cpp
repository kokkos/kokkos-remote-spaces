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

#ifndef TEST_REFCOUNTING_HPP_
#define TEST_REFCOUNTING_HPP_

#include<gtest/gtest.h>
#include<mpi.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_RemoteSpaces.hpp>

template<class DataType, class RemoteSpace, class ... Args>
void test_reference_counting (Args...args) {
  
  int my_rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);

  Kokkos::View<DataType*, RemoteSpace> outer("outer", num_ranks * 10 * sizeof(DataType)); 
  {
    Kokkos::View<DataType*, RemoteSpace> inner = outer;
    ASSERT_EQ(inner.use_count(),2);
  }
  ASSERT_EQ(outer.use_count(),1);
  
  for(int i = my_rank * 10; i < (my_rank+1) * 10; i++)
    outer(i,0) = (my_rank + 1) * 100 + i%10;

  RemoteSpace().fence();

  for(int i=0; i<num_ranks * 10; i++) {
     ASSERT_EQ(outer(i,0),(i/10 + 1)*100 + i%10);
  }
  
  Kokkos::finalize();
  MPI_Finalize();
}

TEST( TEST_CATEGORY, test_reference_counting ) {
  test_reference_counting<int*,Kokkos::DefaultRemoteMemorySpace> ();
  test_reference_counting<double*,Kokkos::DefaultRemoteMemorySpace> ();
}

#endif /* TEST_REFCOUNTING_HPP_ */
