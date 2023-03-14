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

#include <RACERlib_DeviceOps.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

// ETI this
template class RdmaScatterGatherWorker<int>;
template class RdmaScatterGatherWorker<double>;
template class RdmaScatterGatherWorker<size_t>;
// etc.

template <class T>
KOKKOS_FUNCTION T RdmaScatterGatherWorker<T>::get(int pe, uint32_t offset) {
  uint64_t *tail_ctr = &tx_element_request_ctrs[pe];
  uint64_t idx = Kokkos::atomic_fetch_add((unsigned long long *)tail_ctr, 1);
  // printf("idx: %i\n", int(idx));
  memory_fence();
  uint32_t trip_number     = idx / queue_size;
  uint32_t buf_slot        = idx % queue_size;
  uint64_t global_buf_slot = pe * queue_size + buf_slot;

  assert(trip_number < 1000);

  // If we have wrapped around the queue, wait for it to be free
  // this is a huge amount of extra storage, but it's the only way
  // to do it. I can't just use the ack counter to know when a slot
  // is free because I could overwrite the value before thread reads it
  while (atomic_load(
             (unsigned int *)&tx_element_request_trip_counts[global_buf_slot],
             Kokkos::Impl::memory_order_seq_cst_t()) != trip_number)
    ;

  // Enough previous requests are cleared that we can join the queue
  uint32_t *req_ptr = &tx_element_request_queue[global_buf_slot];

  // The queue begins as all zeroes
  // on even passes we set a flag bit of 1 to indicate this is a new request
  // on odd passes we set a flag bit of 0 to indicate this is a new request
  // the requested offset is a combination of the actual offset
  // and flag indicate that this is a new request
  uint32_t offset_request = MAKE_ELEMENT_REQUEST(offset, trip_number);
  atomic_store(req_ptr, offset_request, Kokkos::Impl::memory_order_seq_cst_t());
  memory_fence();

  // We now have to spin waiting for the request to be satisfied
  // we wil get the signal that this is ready when the ack count
  // exceeds our request idx

  uint64_t *ack_ptr = &ack_ctrs_d[pe];
  uint64_t ack = atomic_load(ack_ptr, Kokkos::Impl::memory_order_seq_cst_t());
  while (ack <= idx) {
    ack = atomic_load(ack_ptr, Kokkos::Impl::memory_order_seq_cst_t());
  }

  // at this point, our data is now available in the reply buffer
  T *reply_buffer_T = (T *)rx_element_reply_queue;
  T ret             = atomic_load(&reply_buffer_T[global_buf_slot],
                      Kokkos::Impl::memory_order_seq_cst_t());
  /*memory_fence();
  // update the trip count to signal any waiting threads they can go
  atomic_fetch_add((unsigned int
  *)&tx_element_request_trip_counts[global_buf_slot],
             1u); /*IS THIS NECCASSARY? FIXME*/
  return ret;
}

#define KOKKOS_DISABLE_CACHE

template <class T>
KOKKOS_FUNCTION T RdmaScatterGatherWorker<T>::request(int pe, uint32_t offset) {
#ifdef KOKKOS_DISABLE_CACHE
  return get(pe, offset);
#else
  return cache.get<T>(pe, offset, this);
#endif
}

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos