#ifndef KOKKOS_REMOTESPACES_HPP_
#define KOKKOS_REMOTESPACES_HPP_

#ifdef KOKKOS_ENABLE_QUOSPACE
#include<Kokkos_QUOSpace.hpp>
#define SPACE Kokkos::QUOSpace
#endif

#ifdef KOKKOS_ENABLE_SHMEMSPACE
#include<Kokkos_SHMEMSpace.hpp>
#define SPACE Kokkos::SHMEMSpace
#endif

namespace Kokkos {
  enum { Monolithic, Symmetric, Asymmetric };


  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  struct SHMEMSpace;
  #endif

  template<typename ViewType>
  ViewType allocate_symmetric_remote_view(const char* const label, int num_ranks, int* rank_list, int N0 = 0) {
    typedef typename ViewType::memory_space t_mem_space;
    typedef typename ViewType::array_layout t_layout;

    t_mem_space space;
    space.impl_set_allocation_mode(Kokkos::Symmetric);
    space.impl_set_rank_list(rank_list);
    space.impl_set_extent(N0*sizeof(typename ViewType::value_type));  
    
    t_layout layout(num_ranks,N0);
    return ViewType(Kokkos::view_alloc(std::string(label),space),num_ranks,N0);
  }
}

#endif
