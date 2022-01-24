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

#ifndef RACERLIB_RDMA_HELPERS
#define RACERLIB_RDMA_HELPERS

#define ELEMENT_OFFSET_BIT_SHIFT 1
#define ELEMENT_OFFSET_MASK 0xFFFFFFFE
#define BLOCK_SIZE_BIT_SHIFT 40
#define BLOCK_SIZE_MASK 0xFFFFFF0000000000
#define BLOCK_WINDOW_BIT_SHIFT 20
#define BLOCK_WINDOW_MASK 0xFFFFF00000
#define BLOCK_PE_BIT_SHIFT 1
#define BLOCK_PE_MASK 0xFFFFE

#define QUEUE_SIZE 1 << 20

#define NUM_SEND_WRS 16
#define NUM_RECV_WRS 16
#define TOTAL_NUM_WRS                                                          \
  (NUM_SEND_WRS + NUM_RECV_WRS + NUM_SEND_WRS + NUM_RECV_WRS)

#define START_SEND_REQUEST_WRS 0
#define START_RECV_REQUEST_WRS (START_SEND_REQUEST_WRS + NUM_SEND_WRS)
#define START_SEND_RESPONSE_WRS (START_RECV_REQUEST_WRS + NUM_RECV_WRS)
#define START_RECV_RESPONSE_WRS (START_SEND_RESPONSE_WRS + NUM_SEND_WRS)

#define MAKE_READY_FLAG(trip_number) (((trip_number) + 1) % 2)

#define MAKE_BLOCK_GET_REQUEST(size, pe, trip_number)                          \
  ((uint64_t(size) << BLOCK_SIZE_BIT_SHIFT) |                                  \
   (uint64_t(pe) << BLOCK_PE_BIT_SHIFT) | MAKE_READY_FLAG(trip_number))

#define MAKE_BLOCK_PACK_ACK(size, pe, trip_number)                             \
  MAKE_BLOCK_GET_REQUEST(size, pe, trip_number)

#define MAKE_BLOCK_PACK_REQUEST(size, pe, trip_number, window)                 \
  ((uint64_t(size) << BLOCK_SIZE_BIT_SHIFT) |                                  \
   (uint64_t(pe) << BLOCK_PE_BIT_SHIFT) | MAKE_READY_FLAG(trip_number) |       \
   (uint64_t(window) << BLOCK_WINDOW_BIT_SHIFT))

#define GET_ELEMENT_FLAG(request) (request & 1)

#define GET_BLOCK_FLAG(request) (request & 1)

#define GET_BLOCK_PE(request) (((request)&BLOCK_PE_MASK) >> BLOCK_PE_BIT_SHIFT)

#define GET_BLOCK_SIZE(request) ((request) >> BLOCK_SIZE_BIT_SHIFT)

#define GET_BLOCK_WINDOW(request)                                              \
  (((request)&BLOCK_WINDOW_MASK) >> BLOCK_WINDOW_BIT_SHIFT)

#define MAKE_ELEMENT_REQUEST(offset, trip_number)                              \
  (((offset) << ELEMENT_OFFSET_BIT_SHIFT) | MAKE_READY_FLAG(trip_number))

#define GET_ELEMENT_OFFSET(request) ((request >> ELEMENT_OFFSET_BIT_SHIFT))

#define GET_ELEMENT_FLAG(request) (request & 1)

#define cuda_safe(...)                                                         \
  do {                                                                         \
    CUresult result = (__VA_ARGS__);                                           \
    if (CUDA_SUCCESS != result) {                                              \
      std::cout << result << std::endl;                                        \
      fprintf(stderr, "%s:%d: CUDA failed\n", __FILE__, __LINE__);             \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define time_safe(...)                                                         \
  do {                                                                         \
    auto start = rdtsc();                                                      \
    __VA_ARGS__;                                                               \
    auto stop = rdtsc();                                                       \
    if ((stop - start) > 10000000000ULL) {                                     \
      fprintf(stderr, "call took way too long (%llu-%llu) at %s:%d\n", start,  \
              stop, __FILE__, __LINE__);                                       \
      ::abort();                                                               \
    }                                                                          \
  } while (0)

#define debugf(str, ...)                                                       \
  assert(request_tport != NULL);                                               \
  printf("PE %d: " str "\n", request_tport->my_rank, __VA_ARGS__);             \
  fflush(stdout)

#define debugf_2(str)                                                          \
 printf(str "\n");                                                             \


#ifdef KOKKOS_IBV_DEBUG
#define debug(...) 
#else
#define debug(...)
#endif

#ifdef KOKKOS_IBV_DEBUG
#define debug_2(...) debugf_2(__VA_ARGS__)
#else
#define debug_2(...)
#endif

#define assert_ibv(elem)                                                       \
  if (!elem) {                                                                 \
    fprintf(stderr, "call failed at %s:%d\n", __FILE__, __LINE__);             \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  } else {}

#define ibv_safe(...)                                                          \
  do {                                                                         \
    auto start = rdtsc();                                                      \
    int rc = __VA_ARGS__;                                                      \
    if (rc != 0) {                                                             \
      fprintf(stderr, "call failed with rc=%d at %s:%d\n", rc, __FILE__,       \
              __LINE__);                                                       \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
    auto stop = rdtsc();                                                       \
    if ((stop - start) > 100000000000ULL) {                                    \
      fprintf(stderr, "call took way too long (%llu-%llu) at %s:%d\n", start,  \
              stop, __FILE__, __LINE__);                                       \
      ::abort();                                                               \
    }                                                                          \
  } while (0)

#define ARCH_PPC64

#ifdef ARCH_PPC64
static inline unsigned long long rdtsc() {
  unsigned long long int result = 0;
  unsigned long int upper, lower, tmp;
  __asm__ volatile("0:                  \n"
                   "\tmftbu   %0           \n"
                   "\tmftb    %1           \n"
                   "\tmftbu   %2           \n"
                   "\tcmpw    %2,%0        \n"
                   "\tbne     0b         \n"
                   : "=r"(upper), "=r"(lower), "=r"(tmp));
  result = tmp; // get rid of compiler warnings
  result = upper;
  result = result << 32;
  result = result | lower;

  return (result);
}
#else
static inline uint64_t rdtsc(void) {
  uint32_t eax, edx;
  asm volatile("rdtsc" : "=a"(eax), "=d"(edx));
  return (uint64_t)eax | (uint64_t)edx << 32;
}
#endif

#endif // RACERLIB_RDMA_HELPERS