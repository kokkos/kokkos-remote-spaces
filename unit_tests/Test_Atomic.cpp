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

#ifndef TEST_ATOMIC_GLOBALVIEW_HPP_
#define TEST_ATOMIC_GLOBALVIEW_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

template <class Data_t> void test_atomic_globalview1D(int dim0) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_1D_t =
      Kokkos::View<Data_t *, Kokkos::HostSpace>;
  using ViewRemote_1D_t =
      Kokkos::View<Data_t *, RemoteSpace_t,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_1D_t v = ViewRemote_1D_t("RemoteView", dim0);
  ViewHost_1D_t v_h("HostView", v.extent(0));

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    v_h(i) = 0;

  Kokkos::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", dim0, KOKKOS_LAMBDA(const int i) { v(i)++; });

  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i){
    ASSERT_EQ(v_h(i), num_ranks);
  }
}

template <class Data_t> void test_atomic_globalview2D(int dim0, int dim1) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_2D_t =
      Kokkos::View<Data_t **, Kokkos::HostSpace>;
  using ViewRemote_2D_t =
      Kokkos::View<Data_t **, RemoteSpace_t,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_2D_t v = ViewRemote_2D_t("RemoteView", dim0, dim1);
  ViewHost_2D_t v_h("HostView", v.extent(0), v.extent(1));

  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      v_h(i, j) = 0;

  Kokkos::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", dim0, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j)
          v(i, j)++;
      });

  Kokkos::deep_copy(v_h, v);
  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      ASSERT_EQ(v_h(i, j), num_ranks);
}

template <class Data_t>
void test_atomic_globalview3D(int dim0, int dim1, int dim2) {
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  using ViewHost_3D_t =
      Kokkos::View<Data_t ***,Kokkos::HostSpace>;
  using ViewRemote_3D_t =
      Kokkos::View<Data_t ***, RemoteSpace_t,
                   Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using TeamPolicy_t = Kokkos::TeamPolicy<>;

  ViewRemote_3D_t v = ViewRemote_3D_t("RemoteView", dim0, dim1, dim2);
  ViewHost_3D_t v_h("HostView", v.extent(0), v.extent(1), v.extent(2));
  // Init
  for (int i = 0; i < v_h.extent(0); ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int l = 0; l < v_h.extent(2); ++l)
        v_h(i, j, l) = 0;

  Kokkos::deep_copy(v, v_h);

  Kokkos::parallel_for(
      "Increment", dim0, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim1; ++j)
          for (int k = 0; k < dim2; ++k)
            v(i, j, k)++;
      });

  Kokkos::deep_copy(v_h, v);

  auto local_range = Kokkos::Experimental::get_local_range(dim0);

  for (int i = 0; i < local_range.second - local_range.first; ++i)
    for (int j = 0; j < v_h.extent(1); ++j)
      for (int l = 0; l < v_h.extent(2); ++l)
        ASSERT_EQ(v_h(i, j, l), num_ranks);
}

TEST(TEST_CATEGORY, test_atomic_globalview) {

  #ifdef KOKKOS_ENABLE_MPISPACE
  // TODO: Requires implementing atomic RMA support in MPISpace
  #else
  // 1D
  test_atomic_globalview1D<int>(0);
  test_atomic_globalview1D<int>(1);
  test_atomic_globalview1D<int>(31);

  // 2D
  test_atomic_globalview2D<int>(128, 312);
  test_atomic_globalview2D<int>(256, 237);
  test_atomic_globalview2D<int>(1, 1);

  // 3D
  test_atomic_globalview3D<int>(1, 1, 1);
  test_atomic_globalview3D<int>(255, 1024, 3);
  test_atomic_globalview3D<int>(3, 33, 1024);
  #endif
  

}

#endif /* TEST_ATOMIC_GLOBALVIEW_HPP_ */