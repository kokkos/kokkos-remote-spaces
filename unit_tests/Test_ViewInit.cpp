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

#include <Kokkos_RemoteSpaces.hpp>
#include <gtest/gtest.h>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;

#define is_Layout_PL(Layout) \
  std::enable_if_t<Kokkos::Experimental::Is_Partitioned_Layout<Layout>::value>
#define is_Layout_GL(Layout) \
  std::enable_if_t<!Kokkos::Experimental::Is_Partitioned_Layout<Layout>::value>
#define ZERO DataType(0)
#define ONE DataType(1)

template <class DataType, class Layout, class RemoteSpace, class... Args>
is_Layout_PL(Layout) test_viewinit(Args... args) {
  int numRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  int err_out;

  using RemoteView_t = Kokkos::View<DataType, Layout, RemoteSpace>;
  RemoteView_t view("MyRemoteView", numRanks, args...);
  RemoteSpace_t().fence();

  Kokkos::parallel_reduce("Check zero",view.span(),KOKKOS_LAMBDA(int i, int & err){
    err += view.data()[i] == ZERO ? ZERO : ONE;
  }, err_out);

  ASSERT_EQ(err_out,ZERO);
}

template <class DataType, class Layout, class RemoteSpace, class... Args>
is_Layout_GL(Layout) test_viewinit(Args... args) {
  int err_out;
  using RemoteView_t = Kokkos::View<DataType, Layout, RemoteSpace>;
  RemoteView_t view("MyRemoteView", args...);
  RemoteSpace_t().fence();

  Kokkos::parallel_reduce("Check zero",view.span(),KOKKOS_LAMBDA(int i, int & err){
    err += view.data()[i] == ZERO ? ZERO : ONE;
  }, err_out);

  ASSERT_EQ(err_out,ZERO);
}

#define GENBLOCK(TYPE, LAYOUT, SPACE)      \
test_viewinit<TYPE, LAYOUT, SPACE>(1);     \
test_viewinit<TYPE, LAYOUT, SPACE>(4567);  \
test_viewinit<TYPE, LAYOUT, SPACE>(45617); \
test_viewinit<TYPE, LAYOUT, SPACE>(1,3);   \
test_viewinit<TYPE, LAYOUT, SPACE>(23,12); \
test_viewinit<TYPE, LAYOUT, SPACE>(1,5617);

TEST(TEST_CATEGORY, test_viewinit) {
  using PLL_t = Kokkos::PartitionedLayoutLeft;
  using PLR_t = Kokkos::PartitionedLayoutRight;
  using LL_t  = Kokkos::LayoutLeft;
  using LR_t  = Kokkos::LayoutRight;

  GENBLOCK(int, PLL_t, RemoteSpace_t)
  GENBLOCK(int, PLR_t, RemoteSpace_t)
  GENBLOCK(double, PLL_t, RemoteSpace_t)
  GENBLOCK(double, PLR_t, RemoteSpace_t)
  GENBLOCK(int, LL_t, RemoteSpace_t)
  GENBLOCK(int, LR_t, RemoteSpace_t)
  GENBLOCK(double, LL_t, RemoteSpace_t)
  GENBLOCK(double, LR_t, RemoteSpace_t)

  RemoteSpace_t::fence(); 
}
