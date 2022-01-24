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

#ifndef RACERLIB_ACCESSCACHE_HPP
#define RACERLIB_ACCESSCACHE_HPP

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Core.hpp>
#include <RACERlib_Config.hpp>

using namespace Kokkos;

namespace Kokkos {
namespace Experimental {
namespace RACERlib {
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
static __device__ __inline__ uint32_t warp_id() {
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}
#else
static inline uint32_t warp_id() { return 0; }
#endif

struct RemoteCache {

  RemoteCache()
      : flags(nullptr), values(nullptr), waiting(nullptr), num_ranks(0),
        rank_num_entries(0), modulo_mask(0) {}

  void init(int nranks, int rank_ne, size_t elem_sz) {
    num_ranks = nranks;
    rank_num_entries = rank_ne;
    modulo_mask = rank_num_entries - 1;
    elem_size = elem_sz;

    bool not_power_of_two = rank_num_entries & (rank_num_entries - 1);
    if (not_power_of_two) {
      Kokkos::abort("Number of entries in cache must be a power of 2");
    }

    cache_size =
        rank_num_entries * num_ranks * (2 * sizeof(unsigned int) + elem_size);
  }

  KOKKOS_INLINE_FUNCTION
  int global_cache_slot(int rank, uint32_t slot) {
    return rank * rank_num_entries + slot;
  }

  template <class T, class Fallback>
  KOKKOS_FUNCTION T get(int pe, uint32_t offset, Fallback *fb) {
    int slot = claim_slot<T>(pe, offset, fb);
    auto ready = ready_flag(offset);
    if (slot < 0) {
      slot = -slot;
      while (volatile_load(&flags[slot]) != ready)
        ;
      T *values_T = (T *)values;
      cache_debug("Returning newly claimed value on slot %d at offset %" PRIu32
                  " = %12.8f\n",
                  slot, offset, values_T[slot]);
      return volatile_load(&values_T[slot]);
      //Warning: max is a host func
    } else if (slot == std::numeric_limits<int>::max()) {
      // no slots left, fall back
      return fb->get(pe, offset);
    } else {
      // data already there
      T *values_T = (T *)values;
      cache_debug("Returning existing value on slot %d at offset %" PRIu32
                  " = %12.8f\n",
                  slot, offset, values_T[slot]);
      return volatile_load(&values_T[slot]);
    }
  }

  KOKKOS_INLINE_FUNCTION uint32_t ready_flag(uint32_t offset) {
    return offset + 1;
  }

  KOKKOS_INLINE_FUNCTION uint32_t claim_flag(uint32_t offset) {
    uint32_t claim_bit = (1u) << 31;
    return offset | claim_bit;
  }

  template <class T, class Fallback>
  KOKKOS_FUNCTION int claim_slot(int pe, uint32_t offset, Fallback *fb) {
    auto pe_cache_slot = offset & modulo_mask;
    auto glbl_cache_slot = global_cache_slot(pe, pe_cache_slot);

    T *values_T = reinterpret_cast<T *>(values);
    cache_debug("Rank %d, offset %" PRIu32 " maps to global slot %d - %x\n", pe,
                offset, glbl_cache_slot, modulo_mask);

    // Try to claim the slot
    // The very last bit is used to indicate that the cache slot is pending
    // attempt to claim the cache slot
    unsigned int empty_sentinel = 0;   
    auto ready = ready_flag(offset);
    auto new_claim = claim_flag(offset);
    auto slot_value = atomic_compare_exchange(&flags[glbl_cache_slot],
                                              empty_sentinel, new_claim);
    if (slot_value == ready) {

      cache_debug("Returning existing value at offset %" PRIu32 "\n", offset);
      return glbl_cache_slot;

    } else if (slot_value == new_claim) {

      // We have claimed the cache line, but have not yet filled the value
      auto ticket_number = atomic_fetch_add(&waiting[glbl_cache_slot], 1);
      cache_debug("Got ticket %u at offset %" PRIu32
                  " for slot %d from claim %u\n",
                  ticket_number, offset, glbl_cache_slot, new_claim);

      if (ticket_number == 0) {
        // I am responsible for initiating the RDMA get
        T ret = fb->get(pe, offset);

        // Store the value in the cache first
        volatile_store(&values_T[glbl_cache_slot], ret);

        // Make sure value is written before ready flag
        KOKKOS_REMOTE_THREADFENCE();

        // Now let all other threads know the cache entry is ready to read
        volatile_store(&flags[glbl_cache_slot], ready);
        cache_debug("Filled cache slot at offset %" PRIu32 " on warp %" PRIu32
                    " on slot %d = %12.8f\n",
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
  
  KOKKOS_INLINE_FUNCTION uint32_t hash32shift(uint32_t key) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key;
  }

  unsigned int *flags;
  unsigned int *waiting;

  // Assume 64-bit values for now, we will cast as appropriate
  void *values;

  int num_ranks;
  int modulo_mask;
  int rank_num_entries;
  size_t cache_size;
  size_t elem_size;
};

} // namespace Cache
} // namespace RACERlib
} // namespace Experimental
} // namespace Kokkos

#endif // RACERLIB_ACCESSCACHE_HPP
