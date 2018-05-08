#include<shmem.h>
#include<type_traits>
//----------------------------------------------------------------------------
/** \brief  View mapping for non-specialized data type and standard layout */

namespace Kokkos {
namespace Impl {

KOKKOS_INLINE_FUNCTION
void shmem_type_p(int* ptr, const int& val, const int pe) {
  #ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
  shmem_int_p(ptr,val,pe);
  #endif
}

KOKKOS_INLINE_FUNCTION
int shmem_type_g(int* ptr, const int pe) {
  #ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
  return shmem_int_g(ptr,pe);
  #else
  return 0;
  #endif
}

struct SHMEMSpaceSpecializeTag {};

template<class T>
struct SHMEMDataElement {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T* ptr;
  int pe;
  SHMEMDataElement(int* ptr_, int pe_, int i_ ):ptr(ptr_+i_),pe(pe_) {}
  KOKKOS_INLINE_FUNCTION
  const_value_type operator = (const_value_type& val) const {
    shmem_type_p(ptr,val,pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T val = shmem_type_g(ptr,pe);
    val++;
    shmem_type_p(ptr,val,pe);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T val = shmem_type_g(ptr,pe);
    val--;
    shmem_type_p(ptr,val,pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator ++ () const {
    T val = shmem_type_g(ptr,pe);
    val++;
    shmem_type_p(ptr,val,pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator -- () const {
    T val = shmem_type_g(ptr,pe);
    val--;
    shmem_type_p(ptr,val,pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator ++ (int) const {
    T val = shmem_type_g(ptr,pe);
    val++;
    shmem_type_p(ptr,val,pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator -- (int) const {
    T val = shmem_type_g(ptr,pe);
    val--;
    shmem_type_p(ptr,val,pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator += (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp+=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator -= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp-=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator *= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp*=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator /= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp/=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator %= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp%=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator &= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp&=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator ^= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp^=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator |= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp|=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator <<= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp<<=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator >>= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    tmp>>=val;
    shmem_type_p(ptr,tmp,pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator + (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp+val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator - (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp-val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator * (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp*val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator / (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp/val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator % (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp%val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator ! () const {
    T tmp = shmem_type_g(ptr,pe);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator && (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp&&val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator || (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp||val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator & (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp&val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator | (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp|val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator ^ (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp^val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator ~ () const {
    T tmp = shmem_type_g(ptr,pe);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator << (const unsigned int& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp<<val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator >> (const unsigned int& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp>>val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator == (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp==val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator != (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp!=val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator >= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp>=val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator <= (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp<=val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator < (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp<val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator > (const_value_type& val) const {
    T tmp = shmem_type_g(ptr,pe);
    return tmp>val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type () const {
    return shmem_type_g(ptr,pe);
  }
};

template<class T>
struct SHMEMDataHandle;

template<>
struct SHMEMDataHandle<int> {
  int* ptr;
  KOKKOS_INLINE_FUNCTION
  SHMEMDataHandle():ptr(NULL){}
  KOKKOS_INLINE_FUNCTION
  SHMEMDataHandle(int* ptr_):ptr(ptr_){}
  template<typename iType>
  KOKKOS_INLINE_FUNCTION
  SHMEMDataElement<int> operator() (const int& pe, const iType& i) const {
    SHMEMDataElement<int> element(ptr,pe,i);
    return element;
  }
};

template< class Traits >
struct ViewDataHandle<Traits,typename std::enable_if<std::is_same<typename Traits::specialize,SHMEMSpaceSpecializeTag>::value>::type> {

  typedef typename Traits::value_type   value_type  ;
  typedef SHMEMDataHandle<value_type> handle_type ;
  typedef SHMEMDataElement<value_type> return_type ;
  typedef Kokkos::Impl::SharedAllocationTracker  track_type  ;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( value_type * arg_data_ptr
                           , track_type const & /*arg_tracker*/ )
  {
    return handle_type( arg_data_ptr );
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( handle_type const arg_data_ptr
                           , size_t offset )
  {
    return handle_type( arg_data_ptr + offset );
  }
};

}


template< class ... Prop >
struct ViewTraits< void, SHMEMSpace , Prop ... >
{
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert( std::is_same< typename ViewTraits<void,Prop...>::execution_space , void >::value &&
                 std::is_same< typename ViewTraits<void,Prop...>::memory_space    , void >::value &&
                 std::is_same< typename ViewTraits<void,Prop...>::HostMirrorSpace , void >::value &&
                 std::is_same< typename ViewTraits<void,Prop...>::array_layout    , void >::value
               , "Only one View Execution or Memory Space template argument" );

  typedef typename SHMEMSpace::execution_space                   execution_space ;
  typedef typename SHMEMSpace::memory_space                      memory_space ;
  typedef typename Kokkos::Impl::HostMirror< SHMEMSpace >::Space HostMirrorSpace ;
  typedef typename execution_space::array_layout            array_layout ;
  typedef typename ViewTraits<void,Prop...>::memory_traits  memory_traits ;
  typedef typename Impl::SHMEMSpaceSpecializeTag       specialize ;
};

namespace Impl {

template< class Traits >
class ViewMapping< Traits , SHMEMSpaceSpecializeTag >
{
private:

  template< class , class ... > friend class ViewMapping ;
  template< class , class ... > friend class Kokkos::View ;

  typedef ViewOffset< typename Traits::dimension
                    , typename Traits::array_layout
                    , void
                    >  offset_type ;

  typedef typename ViewDataHandle< Traits >::handle_type  handle_type ;

  handle_type  m_handle ;
  offset_type  m_offset ;
  int m_num_pes;

  KOKKOS_INLINE_FUNCTION
  ViewMapping( const handle_type & arg_handle , const offset_type & arg_offset )
    : m_handle( arg_handle )
    , m_offset( arg_offset )
    {}

public:

  typedef void printable_label_typedef;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

  template< typename iType >
  KOKKOS_INLINE_FUNCTION constexpr size_t extent( const iType & r ) const
    { return m_offset.m_dim.extent(r); }

  KOKKOS_INLINE_FUNCTION constexpr
  typename Traits::array_layout layout() const
    { return m_offset.layout(); }

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0() const { return m_num_pes; }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() const { return m_offset.dimension_1(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() const { return m_offset.dimension_2(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() const { return m_offset.dimension_3(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() const { return m_offset.dimension_4(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() const { return m_offset.dimension_5(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() const { return m_offset.dimension_6(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() const { return m_offset.dimension_7(); }

  // Is a regular layout with uniform striding for each index.
  using is_regular = typename offset_type::is_regular ;

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const { return m_offset.stride_0(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const { return m_offset.stride_1(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const { return m_offset.stride_2(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const { return m_offset.stride_3(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const { return m_offset.stride_4(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const { return m_offset.stride_5(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const { return m_offset.stride_6(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const { return m_offset.stride_7(); }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION void stride( iType * const s ) const { m_offset.stride(s); }

  //----------------------------------------
  // Range span

  /** \brief  Span of the mapped range */
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const { return m_offset.span(); }

  /** \brief  Is the mapped range span contiguous */
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return m_offset.span_is_contiguous(); }

  typedef typename ViewDataHandle< Traits >::return_type  reference_type ;
  typedef typename Traits::value_type *                   pointer_type ;

  /** \brief  Query raw pointer to memory */
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const
    {
      return m_handle;
    }

  //----------------------------------------
  // The View class performs all rank and bounds checking before
  // calling these element reference methods.

  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference() const { return m_handle[0]; }

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename
    std::enable_if< std::is_integral<I0>::value &&
                    ! std::is_same< typename Traits::array_layout , Kokkos::LayoutStride >::value
                  , reference_type >::type
  reference( const I0 & i0 ) const { return m_handle(i0,0); }

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename
    std::enable_if< std::is_integral<I0>::value &&
                    std::is_same< typename Traits::array_layout , Kokkos::LayoutStride >::value
                  , reference_type >::type
  reference( const I0 & i0 ) const { return m_handle(i0, 0); }

  template< typename I0 , typename I1 >
  KOKKOS_FORCEINLINE_FUNCTION
  const reference_type reference( const I0 & i0 , const I1 & i1 ) const
    { const reference_type element = m_handle( i0, m_offset(0,i1) );
       return element; }

  template< typename I0 , typename I1 , typename I2 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 ) const
    { return m_handle( i0, m_offset(0,i1,i2) ); }

  template< typename I0 , typename I1 , typename I2 , typename I3 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3 ) const
    { return m_handle( i0, m_offset(0,i1,i2,i3) ); }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 ) const
    { return m_handle( i0, m_offset(0,i1,i2,i3,i4) ); }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 , const I5 & i5 ) const
    { return m_handle( i0, m_offset(0,i1,i2,i3,i4,i5) ); }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 , const I5 & i5 , const I6 & i6 ) const
    { return m_handle( i0, m_offset(0,i1,i2,i3,i4,i5,i6) ); }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 , typename I7 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 , const I5 & i5 , const I6 & i6 , const I7 & i7 ) const
    { return m_handle( i0, m_offset(0,i1,i2,i3,i4,i5,i6,i7) ); }

  //----------------------------------------

private:

  enum { MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
  enum { MemorySpanSize = sizeof(typename Traits::value_type) };

public:

  /** \brief  Span, in bytes, of the referenced memory */
  KOKKOS_INLINE_FUNCTION constexpr size_t memory_span() const
    {
      return ( m_offset.span() * sizeof(typename Traits::value_type) + MemorySpanMask ) & ~size_t(MemorySpanMask);
    }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION ~ViewMapping() {}
  KOKKOS_INLINE_FUNCTION ViewMapping() : m_handle(), m_offset(), m_num_pes(0) {}
  KOKKOS_INLINE_FUNCTION ViewMapping( const ViewMapping & rhs )
    : m_handle( rhs.m_handle ), m_offset( rhs.m_offset ), m_num_pes(rhs.m_num_pes) {}
  KOKKOS_INLINE_FUNCTION ViewMapping & operator = ( const ViewMapping & rhs )
    { m_handle = rhs.m_handle ; m_offset = rhs.m_offset ; m_num_pes = rhs.m_num_pes; return *this ; }

  KOKKOS_INLINE_FUNCTION ViewMapping( ViewMapping && rhs )
    : m_handle( rhs.m_handle ), m_offset( rhs.m_offset ), m_num_pes(rhs.m_num_pes) {}
  KOKKOS_INLINE_FUNCTION ViewMapping & operator = ( ViewMapping && rhs )
    { m_handle = rhs.m_handle ; m_offset = rhs.m_offset ; m_num_pes = rhs.m_num_pes; return *this ; }

  //----------------------------------------

  /**\brief  Span, in bytes, of the required memory */
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t memory_span( typename Traits::array_layout const & arg_layout )
    {
      typedef std::integral_constant< unsigned , 0 >  padding ;
      return ( offset_type( padding(), arg_layout ).span() * MemorySpanSize + MemorySpanMask ) & ~size_t(MemorySpanMask);
    }

  /**\brief  Wrap a span of memory */
  template< class ... P >
  KOKKOS_INLINE_FUNCTION
  ViewMapping( Kokkos::Impl::ViewCtorProp< P ... > const & arg_prop
             , typename Traits::array_layout const & arg_layout
             )
    : m_handle( ( (Kokkos::Impl::ViewCtorProp<void,pointer_type> const &) arg_prop ).value )
    {
      typedef typename Traits::value_type           value_type ;
      typedef std::integral_constant
        < unsigned
        ,  Kokkos::Impl::ViewCtorProp< P ... >::allow_padding ? sizeof(value_type) : 0
        > padding ;

      typename Traits::array_layout layout;
      for(int i=0; i<Traits::rank; i++)
        layout.dimension[i] = arg_layout.dimension[i];
      layout.dimension[0] = 1;
      m_offset = offset_type( padding(), layout );
      m_num_pes = shmem_n_pes();
    }

  /**\brief  Assign data */
  KOKKOS_INLINE_FUNCTION
  void assign_data( pointer_type arg_ptr )
    { m_handle = handle_type( arg_ptr ); }

  //----------------------------------------
  /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */
  template< class ... P >
  Kokkos::Impl::SharedAllocationRecord<> *
  allocate_shared( Kokkos::Impl::ViewCtorProp< P... > const & arg_prop
                 , typename Traits::array_layout const & arg_layout )
  {
    typedef Kokkos::Impl::ViewCtorProp< P... > alloc_prop ;

    typedef typename alloc_prop::execution_space  execution_space ;
    typedef typename Traits::memory_space         memory_space ;
    typedef typename Traits::value_type           value_type ;
    typedef ViewValueFunctor< execution_space , value_type > functor_type ;
    typedef Kokkos::Impl::SharedAllocationRecord< memory_space , functor_type > record_type ;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    typedef std::integral_constant
      < unsigned
      , alloc_prop::allow_padding ? sizeof(value_type) : 0
      > padding ;


    typename Traits::array_layout layout;
    for(int i=0; i<Traits::rank; i++)
      layout.dimension[i] = arg_layout.dimension[i];
    layout.dimension[0] = 1;
    m_offset = offset_type( padding(), layout );
    m_num_pes = shmem_n_pes();

    const size_t alloc_size =
      ( m_offset.span() * MemorySpanSize + MemorySpanMask ) & ~size_t(MemorySpanMask);

    // Create shared memory tracking record with allocate memory from the memory space
    record_type * const record =
      record_type::allocate( ( (Kokkos::Impl::ViewCtorProp<void,memory_space> const &) arg_prop ).value
                           , ( (Kokkos::Impl::ViewCtorProp<void,std::string>  const &) arg_prop ).value
                           , alloc_size );

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    if ( alloc_size ) {
#endif
    m_handle = handle_type( reinterpret_cast< pointer_type >( record->data() ) );
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    }
#endif

    //  Only initialize if the allocation is non-zero.
    //  May be zero if one of the dimensions is zero.
    if ( alloc_size && alloc_prop::initialize ) {
      // Assume destruction is only required when construction is requested.
      // The ViewValueFunctor has both value construction and destruction operators.
      /*record->m_destroy = functor_type( ( (Kokkos::Impl::ViewCtorProp<void,execution_space> const &) arg_prop).value
                                      , (value_type *) m_handle
                                      , m_offset.span()
                                      );*/

      // Construct values
      record->m_destroy.construct_shared_allocation();
    }

    return record ;
  }
};

}
}
