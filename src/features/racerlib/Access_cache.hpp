#ifndef RACERLIB_ACCESSCACHE_HPP
#define RACERLIB_ACCESSCACHE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>
#include <Config.hpp>

using namespace Kokkos;

namespace RACERlib {
namespace Features {
namespace Cache {

#ifdef __CUDA_ARCH__
#define subset_sync(mask, x) __ballot_sync(mask, x)
#define active_subset_sync(x) __ballot_sync(__activemask(), x)
#else
#define subset_sync(mask, x) mask
#define active_subset_sync(x) 0xFFFFFFFF
#endif

#define cache_debug(...)
//#define cache_debug(...) printf(__VA_ARGS__)

#ifdef __CUDA_ARCH__
static __device__ __inline__ uint32_t warp_id(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}
#else
static inline uint32_t warp_id(){
  return 0;
}
#endif

struct RemoteCache {

  RemoteCache() :
    flags(nullptr),
    values(nullptr),
    waiting(nullptr),
    num_pes(0),
    pe_num_entries(0),
    modulo_mask(0)
  {}

  void init(int npes, int pe_ne, size_t elem_sz){
    num_pes = npes;
    pe_num_entries = pe_ne;
    modulo_mask = pe_num_entries - 1;
    elem_size = elem_sz;

    bool not_power_of_two = pe_ne & (pe_ne - 1);
    if (not_power_of_two){
      Kokkos::abort("number of entries in cache must be a power of 2");
    }

    cache_size = pe_num_entries*num_pes * (2*sizeof(unsigned int) + elem_size);
  }

  KOKKOS_INLINE_FUNCTION
  int global_cache_slot(int pe, uint32_t slot){
    return pe*pe_num_entries + slot;
  }

  template <class T, class Fallback>
  KOKKOS_FUNCTION
  T get(int pe, uint32_t offset, Fallback* fb){
    int slot = claim_slot<T>(pe, offset, fb);
    auto ready = ready_flag(offset);
    if (slot < 0){
      slot = -slot;
      while (volatile_load(&flags[slot]) != ready);
      T* values_T = (T*) values;
      cache_debug("Returning newly claimed value on slot %d at offset %" PRIu32 " = %12.8f\n",
                  slot, offset, values_T[slot]);
      return volatile_load(&values_T[slot]);
    } else if (slot == std::numeric_limits<int>::max()){
      //no slots left, fall back
      return fb->template get<T>(pe, offset);
    } else {
      //data already there
      T* values_T = (T*) values;
      cache_debug("Returning existing value on slot %d at offset %" PRIu32 " = %12.8f\n",
                  slot, offset, values_T[slot]);
      return volatile_load(&values_T[slot]);
    }
  }

  KOKKOS_INLINE_FUNCTION uint32_t ready_flag(uint32_t offset){
    return offset + 1;
  }

  KOKKOS_INLINE_FUNCTION uint32_t claim_flag(uint32_t offset){
    uint32_t claim_bit = (1u)<<31;
    return offset | claim_bit;
  }

  template <class T, class Fallback>
  KOKKOS_FUNCTION
  int claim_slot(int pe, uint32_t offset, Fallback* fb){
    auto pe_cache_slot = offset & modulo_mask;
    auto glbl_cache_slot = global_cache_slot(pe, pe_cache_slot);

    T* values_T = reinterpret_cast<T*>(values);
    cache_debug("Pe %d, offset %" PRIu32 " maps to global slot %d - %x\n",
           pe, offset, glbl_cache_slot, modulo_mask);

    //try to claim the slot
    unsigned int empty_sentinel = 0;
    //the very last bit is used to indicate that the cache slot is pending
    //attempt to claim the cache slot
    auto ready = ready_flag(offset);
    auto new_claim = claim_flag(offset);
    auto slot_value = atomic_compare_exchange(&flags[glbl_cache_slot], empty_sentinel, new_claim);
    if (slot_value == ready){
      cache_debug("Returning existing value at offset %" PRIu32 "\n", offset);
      return glbl_cache_slot;
    } else if (slot_value == new_claim) {
      //we have claimed the cache line, but have not yet filled the value
      auto ticket_number = atomic_fetch_add(&waiting[glbl_cache_slot], 1);
      cache_debug("Got ticket %u at offset %" PRIu32 " for slot %d from claim %u\n",
                  ticket_number, offset, glbl_cache_slot, new_claim);
      if (ticket_number == 0){
        //I am responsible for initiating the RDMA get
        T ret = fb->template get<T>(pe, offset);
        //store the value in the cache first
        volatile_store(&values_T[glbl_cache_slot], ret);
        //make sure value is written before ready flag
        KOKKOS_REMOTE_THREADFENCE();
        //now let all other threads know the cache entry is ready to read
        volatile_store(&flags[glbl_cache_slot], ready);
        cache_debug("Filled cache slot at offset %" PRIu32 " on warp %" PRIu32 " on slot %d = %12.8f\n",
                    offset, warp_id(), glbl_cache_slot, ret);
        return glbl_cache_slot;
      } else {
        return -glbl_cache_slot;
      }
    } else {
      // no open slots could be claimed, send back sentinel value
      return std::numeric_limits<int>::max();
    }
  }

  void invalidate();

  KOKKOS_INLINE_FUNCTION uint32_t hash32shift(uint32_t key){
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key;
  }

  unsigned int* flags;
  unsigned int* waiting;
  //assume 64-bit values for now, we will cast as appropriate
  void* values;

  int num_pes;
  int modulo_mask;
  int pe_num_entries;
  size_t cache_size;
  size_t elem_size;


};

} // namespace Cache
} // namespace Features
} // namespace RACERlib

#endif // RACERLIB_ACCESSCACHE_HPP
