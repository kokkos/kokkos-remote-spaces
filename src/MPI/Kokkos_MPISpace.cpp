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

#include <Kokkos_Core.hpp>
#include <Kokkos_MPISpace.hpp>
#include <mpi.h>


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

MPI_Win MPISpace::current_win;

/* Default allocation mechanism */
MPISpace::MPISpace():rank_list(NULL),allocation_mode(Symmetric)
{}

void MPISpace::impl_set_rank_list(int* const rank_list_) {
  rank_list = rank_list_;
}

void MPISpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void MPISpace::impl_set_extent(const int64_t extent_) {
  extent = extent_;
}

void * MPISpace::allocate( const size_t arg_alloc_size ) const 
{
  static_assert( sizeof(void*) == sizeof(uintptr_t)
               , "Error sizeof(void*) != sizeof(uintptr_t)" );

  static_assert( Kokkos::Impl::is_integral_power_of_two( Kokkos::Impl::MEMORY_ALIGNMENT )
               , "Memory alignment must be power of two" );

  constexpr uintptr_t alignment = Kokkos::Impl::MEMORY_ALIGNMENT ;
  constexpr uintptr_t alignment_mask = alignment - 1 ;

  void * ptr = 0 ;
  if (arg_alloc_size) {

    if( allocation_mode == Kokkos::Symmetric ) {
      //int num_pes = shmem_n_pes();
      //int my_id = shmem_my_pe();
      current_win = MPI_WIN_NULL;
      printf("Allocate Window A %p %li\n",&current_win,*((long*)&current_win));
      MPI_Win_allocate((arg_alloc_size+sizeof(int)-1)/sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &ptr, &current_win);
      //ptr = shmalloc(arg_alloc_size);
      printf("Allocate Window B %p %li\n",&current_win,*((long*)&current_win));
    } else {
      Kokkos::abort("MPISpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}


void MPISpace::deallocate( void * const
    , const size_t
    ) const
{
      printf("Free Window %p %li\n",&current_win,*((long*)&current_win));
  int assert = 0;
  MPI_Win win = current_win;
  current_win = MPI_WIN_NULL;
  MPI_Win_fence(assert,win);
  MPI_Win_free(&win);
}

void MPISpace::fence() {
  int assert = 0;
  MPI_Win_fence(assert,current_win);
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::MPISpace , void >::s_root_record ;

void
SharedAllocationRecord< Kokkos::MPISpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::MPISpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::MPISpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif
  printf("~SharedAllocationRecord %p %li\n",this,*((long*)&win));
  m_space.current_win = win;
  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::MPISpace , void >::
SharedAllocationRecord( const Kokkos::MPISpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG
          & SharedAllocationRecord< Kokkos::MPISpace , void >::s_root_record,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
   }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
  win = m_space.current_win;
  printf("SharedAllocationRecord(...) %p %li\n",this,*((long*)&win));
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::MPISpace , void >::
allocate_tracked( const Kokkos::MPISpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );
printf("A\n");
  return r->data();
}

void SharedAllocationRecord< Kokkos::MPISpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
printf("B\n");
}

void * SharedAllocationRecord< Kokkos::MPISpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<MPISpace,MPISpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< Kokkos::MPISpace , void > *
SharedAllocationRecord< Kokkos::MPISpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::MPISpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::MPISpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< Kokkos::MPISpace , void >::
print_records( std::ostream & s , const Kokkos::MPISpace & , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "MPISpace" , & s_root_record , detail );
}

} // namespace Impl
} // namespace Kokkos

