#include <RDMA_Worker.hpp>

namespace RACERlib {

template <class T>
KOKKOS_FUNCTION T RdmaScatterGatherWorker::get(int pe, uint32_t offset) {
  uint64_t *tail_ctr = &tx_element_request_ctrs[pe];
  uint64_t idx = Kokkos::atomic_fetch_add((unsigned long long *)tail_ctr, 1);
  uint32_t trip_number = idx / queue_size;
  uint32_t buf_slot = idx % queue_size;
  uint64_t global_buf_slot = pe * queue_size + buf_slot;
  // if we have wrapped around the queue, wait for it to be free
  // this is a huge amount of extra storage, but it's the only way
  // to do it. I can't just use the ack counter to know when a slot
  // is free because I could overwrite the value before thread reads it
  while (volatile_load((
             unsigned int *)&tx_element_request_trip_counts[global_buf_slot]) !=
         trip_number)
    ;

  // enough previous requests are cleared that we can join the queue
  uint32_t *req_ptr = &tx_element_request_queue[global_buf_slot];
  // the queue begins as all zeroes
  // on even passes we set a flag bit of 1 to indicate this is a new request
  // on odd passes we set a flag bit of 0 to indicate this is a new request
  // the requested offset is a combination of the actual offset
  // and flag indicate that this is a new request
  uint32_t offset_request = MAKE_ELEMENT_REQUEST(offset, trip_number);
  volatile_store(req_ptr, offset_request);
  // we now have to spin waiting for the request to be satisfied
  // we wil get the signal that this is ready when the ack count
  // exceeds our request idx

  uint64_t *ack_ptr = &ack_ctrs_d[pe];
  uint64_t ack = volatile_load(ack_ptr);
  while (ack <= idx) {
    ack = volatile_load(ack_ptr);
  }

  // at this point, our data is now available in the reply buffer
  T *reply_buffer_T = (T *)rx_element_reply_queue;
  T ret = volatile_load(&reply_buffer_T[global_buf_slot]);
  // update the trip count to signal any waiting threads they can go
  atomic_add((unsigned int *)&tx_element_request_trip_counts[global_buf_slot],
             1u);
  return ret;
}

template <class T>
KOKKOS_FUNCTION T RdmaScatterGatherWorker::request(int pe, uint32_t offset) {
#ifdef KOKKOS_DISABLE_CACHE
  return get<T>(pe, offset);
#else
  return cache.get<T>(pe, offset, this);
#endif
}

} // namespace RACERlib