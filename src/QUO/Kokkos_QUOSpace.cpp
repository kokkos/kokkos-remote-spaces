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
#include <Kokkos_QUOSpace.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {
  
  QUO_context default_quo_context() {
    static QUO_context quo_context = NULL;
    if( quo_context == NULL ) {
      int return_val = QUO_create(&quo_context,MPI_COMM_WORLD);
      if(return_val != QUO_SUCCESS)
        Kokkos::abort("QUO Remote Memory Space Context Creation Failed!");
    }
    return quo_context;
  }
  
}

/* Default allocation mechanism */
QUOSpace::QUOSpace():quo_context(Impl::default_quo_context()),rank_list(NULL),allocation_mode(Monolithic)
{}

void QUOSpace::impl_set_rank_list(int* const rank_list_) {
  rank_list = rank_list_;
}

void QUOSpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void QUOSpace::impl_set_extent(const int64_t extent_) {
  extent = extent_;
}

void * QUOSpace::allocate( const size_t arg_alloc_size ) const
{
  static_assert( sizeof(void*) == sizeof(uintptr_t)
               , "Error sizeof(void*) != sizeof(uintptr_t)" );

  static_assert( Kokkos::Impl::is_integral_power_of_two( Kokkos::Impl::MEMORY_ALIGNMENT )
               , "Memory alignment must be power of two" );

  constexpr uintptr_t alignment = Kokkos::Impl::MEMORY_ALIGNMENT ;
  constexpr uintptr_t alignment_mask = alignment - 1 ;

  void * ptr = 0 ;

  if (allocation_mode == Kokkos::Monolithic) {
    if (arg_alloc_size) {
	int qid = -1;
	int num_qids = -1;
	QUO_id (quo_context, &qid);
	QUO_nqids (quo_context, &num_qids);
	// TODO make sure everybody requested same size. 

	QUO_xpm_context xpm;
	size_t alloc_size_with_header =
	  arg_alloc_size + num_qids * (sizeof (QUO_context) +
				       sizeof (QUO_xpm_context)) +
	  sizeof (uint64_t);
	QUO_xpm_allocate (quo_context, qid == 0 ? arg_alloc_size : 0, &xpm);
	QUO_xpm_view_t r_view;
	QUO_xpm_view_by_qid (xpm, 0, &r_view);

	char *base_ptr = (char*)r_view.base;

	// Store my QUO context handle
	{
	  QUO_context *my_context_ptr =
	    (QUO_context *) (base_ptr + qid * sizeof (QUO_context));
	  *my_context_ptr = quo_context;
	}

	// Store my QUO xpm context handle
	{
	  QUO_xpm_context *my_context_ptr =
	    (QUO_xpm_context *) (base_ptr + num_qids * sizeof (QUO_context) +
				 qid * sizeof (QUO_xpm_context));
	  *my_context_ptr = xpm;
	}

	// Store number of thingies
	{
	  uint64_t *count_ptr =
	    (uint64_t *) (base_ptr +
			  num_qids * (sizeof (QUO_context) +
				      sizeof (QUO_xpm_context)));
	  *count_ptr = num_qids;
	}

	ptr =
	  base_ptr + num_qids * (sizeof (QUO_context) +
				 sizeof (QUO_xpm_context)) +
	  sizeof (uint64_t);
    }
  }
  if( allocation_mode == Kokkos::Symmetric ) {
	int qid = -1;
	int num_qids = -1;
	QUO_id (quo_context, &qid);
	QUO_nqids (quo_context, &num_qids);
	// TODO make sure everybody requested same size. 

	QUO_xpm_context xpm;
	size_t header_size = num_qids * (sizeof (QUO_context) +
				         sizeof (QUO_xpm_context)) +
	                     sizeof (uint64_t);
	QUO_xpm_allocate (quo_context, qid == 0 ? extent+header_size : extent, &xpm);
	QUO_xpm_view_t r_view;
	QUO_xpm_view_by_qid (xpm, 0, &r_view);

	char *base_ptr = (char*)r_view.base;

	// Store my QUO context handle
	{
	  QUO_context *my_context_ptr =
	    (QUO_context *) (base_ptr + qid * sizeof (QUO_context));
	  *my_context_ptr = quo_context;
	}

	// Store my QUO xpm context handle
	{
	  QUO_xpm_context *my_context_ptr =
	    (QUO_xpm_context *) (base_ptr + num_qids * sizeof (QUO_context) +
				 qid * sizeof (QUO_xpm_context));
	  *my_context_ptr = xpm;
	}

	// Store number of thingies
	{
	  uint64_t *count_ptr =
	    (uint64_t *) (base_ptr +
			  num_qids * (sizeof (QUO_context) +
				      sizeof (QUO_xpm_context)));
	  *count_ptr = num_qids;
	}

	ptr =
	  base_ptr + num_qids * (sizeof (QUO_context) +
				 sizeof (QUO_xpm_context)) +
	  sizeof (uint64_t);
  } 
  return ptr;
}


void QUOSpace::deallocate( void * const arg_alloc_ptr
    , const size_t
    ) const
{
  uint64_t* count_ptr = (uint64_t*) (arg_alloc_ptr - sizeof(uint64_t));
  int num_qids = *count_ptr;
  int qid; 
  QUO_id(quo_context, &qid); 
  QUO_xpm_context* xpm_ptr = (QUO_xpm_context*) (arg_alloc_ptr - sizeof(uint64_t) - (num_qids-qid) * sizeof(QUO_xpm_context));
  
  QUO_xpm_context xpm = *xpm_ptr;

  QUO_xpm_view_t r_view;
  QUO_xpm_view_by_qid(xpm, 0, &r_view);

  void* base_ptr = r_view.base;
  QUO_xpm_free(xpm);
}

void QUOSpace::fence() {
  QUO_barrier(quo_context);
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::QUOSpace , void >::s_root_record ;

void
SharedAllocationRecord< Kokkos::QUOSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::QUOSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::QUOSpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::QUOSpace , void >::
SharedAllocationRecord( const Kokkos::QUOSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG 
       & SharedAllocationRecord< Kokkos::QUOSpace , void >::s_root_record ,
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
  RecordBase::m_alloc_ptr->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;

}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::QUOSpace , void >::
allocate_tracked( const Kokkos::QUOSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::QUOSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::QUOSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<QUOSpace,QUOSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< Kokkos::QUOSpace , void > *
SharedAllocationRecord< Kokkos::QUOSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::QUOSpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::QUOSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< Kokkos::QUOSpace , void >::
print_records( std::ostream & s , const Kokkos::QUOSpace & , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "QUOSpace" , & s_root_record , detail );
}

} // namespace Impl
} // namespace Kokkos

