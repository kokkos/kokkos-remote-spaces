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

#define value(j,k,l,rank,range) (l)+(k)*(size_l)+(j)*(size_l*size_k)+\
        rank*((range.second - range.first)*size_k*size_l)

#define SUM(start, end, red) \
  for(int i=start;i<=end;i++) red+=i; 

template <class DataType, class Layout, class RemoteSpace>
is_Layout_GL(Layout) test_mdrangepolicy(int x, int y, int z) { 
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  int res = 0, res_ref = 0;
  int next_rank = (my_rank + 1) % num_ranks;

  int size_j, size_k, size_l;

  size_j = x;
  size_k = y;
  size_l = z;

  using MyRangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
  using ViewRemote_3D_t = Kokkos::View<int ***, RemoteSpace_t>;
  using Local3D_t = typename ViewRemote_3D_t::HostMirror;

  ViewRemote_3D_t view = ViewRemote_3D_t("RemoteView", size_j, size_k, size_l);
  auto local_range = Kokkos::Experimental::get_local_range(size_j);
  Local3D_t v_H("HostView", view.extent(0), size_k, size_l);

  MyRangePolicy local_md_range({0, 0, 0}, {local_range.second - local_range.first, size_k, size_l});
  
  Kokkos::parallel_for("Init", local_md_range , KOKKOS_LAMBDA(const int j, const int k, const int l){
    v_H(j, k, l) = value(j,k,l, my_rank,local_range);
  });
  Kokkos::fence();
  Kokkos::deep_copy(view,v_H);
  RemoteSpace_t::fence();

  auto remote_range =
      Kokkos::Experimental::get_range(size_j, (my_rank + 1) % num_ranks);
  MyRangePolicy remote_md_range({remote_range.first, 0, 0}, {remote_range.second, size_k, size_l});

  Kokkos::parallel_reduce("Remote Access via View", remote_md_range , KOKKOS_LAMBDA(const int j, const int k, const int l, int & tmp ){
    tmp += view(j, k, l);
    int val = view(j, k, l);
  }, res);
  Kokkos::fence();
  RemoteSpace_t::fence();

  int start = value(0,0,0,next_rank,remote_range);
  int end   = value((remote_range.second - remote_range.first)-1,(size_k-1), (size_l-1), next_rank,remote_range);

  SUM(start, end, res_ref);  
  ASSERT_EQ(res,res_ref);
  Kokkos::fence();
}

template <class DataType, class Layout, class RemoteSpace>
is_Layout_PL(Layout) test_mdrangepolicy(int x, int y, int z) {  
  // tbd
}

#define GEN_BLOCK(Type, Layout, Space)                  \
  test_mdrangepolicy<Type ***, Layout, Space>(4, 5, 6); \
  test_mdrangepolicy<Type ***, Layout, Space>(1, 2, 10);\
  test_mdrangepolicy<Type ***, Layout, Space>(5, 2, 2);

TEST(TEST_CATEGORY, test_mdrangepolicy) { 
  using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
  using PLL_t         = Kokkos::PartitionedLayoutLeft;
  using PLR_t         = Kokkos::PartitionedLayoutRight;
  using LL_t          = Kokkos::LayoutLeft;
  using LR_t          = Kokkos::LayoutRight;

  GEN_BLOCK(int, PLL_t, RemoteSpace_t)
  GEN_BLOCK(int, PLR_t, RemoteSpace_t)
  GEN_BLOCK(double, PLL_t, RemoteSpace_t)
  GEN_BLOCK(double, PLR_t, RemoteSpace_t)

  GEN_BLOCK(int, LL_t, RemoteSpace_t)
  GEN_BLOCK(int, LR_t, RemoteSpace_t)
  GEN_BLOCK(double, LL_t, RemoteSpace_t)
  GEN_BLOCK(double, LL_t, RemoteSpace_t)

  RemoteSpace_t::fence();
  
}
