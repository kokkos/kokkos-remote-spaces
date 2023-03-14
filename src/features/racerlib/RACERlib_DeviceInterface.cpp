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

#include <RACERlib_DeviceInterface.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

#ifdef RAW_CUDA

// ETI this
template __device__ void
pack_response_kernel<double, Kokkos::Impl::CudaTeamMember const &>(
    double *local_values, RdmaScatterGatherWorker<double> *sgw,
    unsigned *completion_flag, Kokkos::Impl::CudaTeamMember const &team,
    bool final);

template __device__ void
aggregate_requests_kernel<double, Kokkos::Impl::CudaTeamMember const &>(
    RdmaScatterGatherWorker<double> *sgw,
    Kokkos::Impl::CudaTeamMember const &team, unsigned num_worker_teams);

template <typename T, class Team>
__device__ void pack_response_kernel(T *local_values,
                                     RdmaScatterGatherWorker<T> *sgw,
                                     unsigned *completion_flag, Team &&team,
                                     bool final) {
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t request;
  int my_thread       = threadIdx.x * blockDim.y + threadIdx.y;
  int total_threads   = blockDim.x * blockDim.y;
  uint32_t queue_size = RdmaScatterGatherEngine::queue_size;

  completion = 0;
  request    = 0;

  while (completion < 1) {
    uint64_t idx         = sgw->rx_block_request_ctr % queue_size;
    uint64_t trip_number = sgw->rx_block_request_ctr / queue_size;

    if (my_thread == 0) {
      request = atomic_load(&sgw->rx_block_request_cmd_queue[idx],
                            Kokkos::Impl::memory_order_seq_cst_t());
    }

    __syncthreads();

    if (GET_BLOCK_FLAG(request) == MAKE_READY_FLAG(trip_number)) {
      uint32_t num_requests = GET_BLOCK_SIZE(request);
      uint32_t pe           = GET_BLOCK_PE(request);
      uint32_t window       = GET_BLOCK_WINDOW(request);
      uint32_t reply_offset =
          pe * queue_size + sgw->tx_element_reply_ctrs[pe] % queue_size;
      uint32_t *offsets = sgw->rx_element_request_queue + window * queue_size;
      T *reply_tx_buffer_T = ((T *)sgw->tx_element_reply_queue) + reply_offset;

      uint32_t num_packed = 0;

      while (num_packed < num_requests) {
        __threadfence_system();
        uint32_t my_index = num_packed + my_thread;
        if (my_index < num_requests) {
          // This needs to be volatile to force visibility from the IB send
          uint32_t offset             = GET_ELEMENT_OFFSET(atomic_load(
              &offsets[my_index], Kokkos::Impl::memory_order_seq_cst_t()));
          reply_tx_buffer_T[my_index] = local_values[offset];
        }
        num_packed += total_threads;
      }
      if (my_thread == 0) {
        //++sgw->rx_block_request_ctr;
        atomic_fetch_inc(&sgw->rx_block_request_ctr);
        // atomic_fetch_(&sgw->rx_block_request_ctr);
        atomic_fetch_add(&sgw->tx_element_reply_ctrs[pe], num_requests);
        // sgw->tx_element_reply_ctrs[pe] += num_requests;
        memory_fence();
      }

      // Force visibility
      __threadfence_system();

      if (my_thread == 0) {
        atomic_store(&sgw->tx_block_reply_cmd_queue[idx], request, Kokkos::Impl::memory_order_seq_cst_t());
        memory_fence();
      }
    }

    if (my_thread == 0) {
      completion =
          atomic_load(completion_flag, Kokkos::Impl::memory_order_seq_cst_t());
    }
    __threadfence_system();
    __syncthreads();
  }  // While loop

  __syncthreads();

  if (my_thread == 0) {
    atomic_store(completion_flag, 0u, Kokkos::Impl::memory_order_seq_cst_t());
    memory_fence();
  }
}

template <typename T, class Team>
__device__ void aggregate_requests_kernel(RdmaScatterGatherWorker<T> *sgw,
                                          Team &&team,
                                          unsigned num_worker_teams) {
  int my_thread                 = threadIdx.x * blockDim.y + threadIdx.y;
  int total_threads             = blockDim.x * blockDim.y;
  uint32_t queue_size           = RdmaScatterGatherEngine::queue_size;
  static constexpr uint32_t mtu = 16384;  // try to at least send 16K elements
  static constexpr uint32_t max_mtu_stalls = 4;
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t total_requests;
  KOKKOS_REMOTE_SHARED int misses[32];  // TODO, make this an array

  for (int i = 0; i < 32; ++i) misses[i] = 0;

  completion     = 0;
  total_requests = 0;

  __syncthreads();

  while (completion < num_worker_teams) {
    for (int pe = 0; pe < sgw->num_ranks; ++pe) {
      uint64_t head = sgw->tx_element_aggregate_ctrs[pe];
      
      if(pe == sgw->my_rank) continue;

      if (my_thread == 0) {
        uint64_t last_cleared_on_device = sgw->ack_ctrs_d[pe];
        if (head > last_cleared_on_device) {
          uint64_t last_cleared_on_host = atomic_load(
              &sgw->ack_ctrs_h[pe], Kokkos::Impl::memory_order_seq_cst_t());
          if (last_cleared_on_device < last_cleared_on_host) {
            atomic_store(&sgw->ack_ctrs_d[pe], last_cleared_on_host,
                         Kokkos::Impl::memory_order_seq_cst_t());
            memory_fence();
          }
        }
        uint64_t max_index =
            Kokkos::atomic_fetch_add(&sgw->tx_element_request_ctrs[pe], 0u);
        memory_fence();
        total_requests = max_index - head;

        //printf("Total Req:%i\n", int(total_requests));
        if (total_requests < mtu && misses[pe] < max_mtu_stalls) {
          total_requests = 0;
          ++misses[pe];
        } else {
          misses[pe] = 0;
        }
     }

      __threadfence_system();
      __syncthreads();

      if (total_requests > 0) {
        unsigned requests_done = 0;
        while (requests_done < total_requests) {
          uint64_t my_offset = head + requests_done + my_thread;
          if (my_offset < head + total_requests) {
            uint64_t my_idx         = my_offset % queue_size;
            uint64_t my_trip_number = my_offset / queue_size;
            uint32_t ready_flag     = MAKE_READY_FLAG(my_trip_number);
            uint32_t req_slot       = my_idx + pe * queue_size;
            uint32_t next_request =
                atomic_load(&sgw->tx_element_request_queue[req_slot],
                            Kokkos::Impl::memory_order_seq_cst_t());
            while (GET_BLOCK_FLAG(next_request) != ready_flag) {
              next_request =
                  atomic_load(&sgw->tx_element_request_queue[req_slot],
                              Kokkos::Impl::memory_order_seq_cst_t());
            }
           // printf("pe %i, tid %i, offset %i, trip %i, reqDone %i, totReq %i\n", pe , 
            //int(my_idx), next_request, int(my_trip_number), int(requests_done), int(total_requests) );
            // This looks stupid, but is necessary to make visible to peer
            // devices
            // sgw->tx_element_request_queue[req_slot] = next_request;
            atomic_store(&sgw->tx_element_request_queue[req_slot], next_request,
                         Kokkos::Impl::memory_order_seq_cst_t());
            memory_fence();
          }
          requests_done += total_threads;
        }

        // We have written the requests, now make them peer visible
        __threadfence_system();

        if (my_thread == 0) {
          uint64_t tail_idx = sgw->tx_block_request_ctr++;
          sgw->tx_element_aggregate_ctrs[pe] += total_requests;
          uint64_t queue_idx   = tail_idx % queue_size;
          uint64_t trip_number = tail_idx / queue_size;
          uint64_t request =
              MAKE_BLOCK_GET_REQUEST(total_requests, pe, trip_number);
          atomic_store(&sgw->tx_block_request_cmd_queue[queue_idx], request,
                       Kokkos::Impl::memory_order_seq_cst_t());
          memory_fence();
        }

      }
        __threadfence_system();
        __syncthreads();
    }
    if (my_thread == 0) {
      completion = atomic_load(sgw->request_done_flag,
                               Kokkos::Impl::memory_order_seq_cst_t());
    }
    __syncthreads();
  }  // While loop

  __syncthreads();
  debug_2("Exiting kernel 0\n");

  if (my_thread == 0) {
    atomic_store(sgw->request_done_flag, 0u,
                 Kokkos::Impl::memory_order_seq_cst_t());
    atomic_store(sgw->response_done_flag, 1u,
                 Kokkos::Impl::memory_order_seq_cst_t());
    memory_fence();
  }
}

#else

// ETI this
template KOKKOS_FUNCTION void
pack_response_kernel<double, Kokkos::Impl::CudaTeamMember const &>(
    double *local_values, RdmaScatterGatherWorker<double> *sgw,
    unsigned *completion_flag, Kokkos::Impl::CudaTeamMember const &team,
    bool final);

template KOKKOS_FUNCTION void
aggregate_requests_kernel<double, Kokkos::Impl::CudaTeamMember const &>(
    RdmaScatterGatherWorker<double> *sgw,
    Kokkos::Impl::CudaTeamMember const &team, unsigned num_worker_teams);

template <class T, class Team>
KOKKOS_FUNCTION void aggregate_requests_kernel(RdmaScatterGatherWorker<T> *sgw,
                                               Team &&team,
                                               unsigned num_worker_teams) {
  uint32_t queue_size           = RdmaScatterGatherEngine::queue_size;
  static constexpr uint32_t mtu = 16384;  // try to at least send 16K elements
  static constexpr uint32_t max_mtu_stalls = 4;
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t total_requests;
  KOKKOS_REMOTE_SHARED int
      misses[32];  // TODO, make this an array, I'm too lazy right now
  for (int i = 0; i < 32; ++i) misses[i] = 0;

  completion = 0;

  team.team_barrier();
  while (completion < num_worker_teams) {
    for (int pe = 0; pe < sgw->num_ranks; ++pe) {
      uint64_t head = sgw->tx_element_aggregate_ctrs[pe];
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        total_requests                  = 0;
        uint64_t last_cleared_on_device = sgw->ack_ctrs_d[pe];
        if (head > last_cleared_on_device) {
          uint64_t last_cleared_on_host = atomic_load(
              &sgw->ack_ctrs_h[pe], Kokkos::Impl::memory_order_seq_cst_t());
          if (last_cleared_on_device < last_cleared_on_host) {
            atomic_store(&sgw->ack_ctrs_d[pe], last_cleared_on_host,
                         Kokkos::Impl::memory_order_seq_cst_t());
          }
        }
        uint64_t max_index =
            Kokkos::atomic_fetch_add(&sgw->tx_element_request_ctrs[pe], 0u);
        total_requests = max_index - head;
        if (total_requests < mtu && misses[pe] < max_mtu_stalls) {
          total_requests = 0;
          ++misses[pe];
        } else {
          misses[pe] = 0;
        }
      });
      team.team_barrier();
      if (total_requests > 0) {
        auto vec_length     = team.vector_length();
        uint64_t num_passes = total_requests / vec_length;
        if (total_requests % vec_length) num_passes++;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, 0, num_passes),
            [&](const int64_t pass) {
              uint64_t start = pass * vec_length;
              uint64_t stop  = start + vec_length;
              if (stop > total_requests) stop = total_requests;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team, start, stop),
                  [=](uint64_t offset) {
                    uint64_t my_offset      = head + offset;
                    uint64_t my_trip_number = my_offset / queue_size;
                    uint64_t my_idx         = my_offset % queue_size;
                    uint64_t ready_flag     = MAKE_READY_FLAG(my_trip_number);
                    uint64_t req_slot       = my_idx + pe * queue_size;
                    uint32_t next_request =
                        atomic_load(&sgw->tx_element_request_queue[req_slot],
                                    Kokkos::Impl::memory_order_seq_cst_t());
                    while (GET_BLOCK_FLAG(next_request) != ready_flag) {
                      next_request =
                          atomic_load(&sgw->tx_element_request_queue[req_slot],
                                      Kokkos::Impl::memory_order_seq_cst_t());
                    }
                    // This looks stupid, but is necessary to make visible to
                    // peer devices
                    sgw->tx_element_request_queue[req_slot] = next_request;
                  });
            });
        // We have written the requests, now make them peer visible
        KOKKOS_REMOTE_THREADFENCE_SYSTEM();

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          uint64_t tail_idx = sgw->tx_block_request_ctr++;
          sgw->tx_element_aggregate_ctrs[pe] += total_requests;
          uint64_t queue_idx   = tail_idx % queue_size;
          uint64_t trip_number = tail_idx / queue_size;
          uint64_t request =
              MAKE_BLOCK_GET_REQUEST(total_requests, pe, trip_number);
          atomic_store(&sgw->tx_block_request_cmd_queue[queue_idx], request,
                       Kokkos::Impl::memory_order_seq_cst_t());
        });
        team.team_barrier();
      }
    }

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      completion = atomic_load(sgw->request_done_flag,
                               Kokkos::Impl::memory_order_seq_cst_t());
    });
    team.team_barrier();
  }
  team.team_barrier();

  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    atomic_store(sgw->request_done_flag, 0u,
                 Kokkos::Impl::memory_order_seq_cst_t());
    atomic_store(sgw->response_done_flag, 1u,
                 Kokkos::Impl::memory_order_seq_cst_t());
  });
}

template <typename T, class Team>
KOKKOS_FUNCTION void pack_response_kernel(T *local_values,
                                          RdmaScatterGatherWorker<T> *sgw,
                                          unsigned *completion_flag,
                                          Team &&team, bool final) {
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t request;

  completion = 0;

  uint32_t queue_size = RdmaScatterGatherEngine::queue_size;
  while (completion == 0) {
    uint64_t idx         = sgw->rx_block_request_ctr % queue_size;
    uint64_t trip_number = sgw->rx_block_request_ctr / queue_size;
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      request = atomic_load(&sgw->rx_block_request_cmd_queue[idx],
                            Kokkos::Impl::memory_order_seq_cst_t());
    });
    team.team_barrier();
    if (GET_BLOCK_FLAG(request) == MAKE_READY_FLAG(trip_number)) {
      uint32_t num_requests = GET_BLOCK_SIZE(request);
      uint32_t pe           = GET_BLOCK_PE(request);
      uint32_t window       = GET_BLOCK_WINDOW(request);
      uint32_t reply_offset =
          pe * queue_size + sgw->tx_element_reply_ctrs[pe] % queue_size;
      uint32_t *offsets = sgw->rx_element_request_queue + window * queue_size;
      T *reply_tx_buffer_T = ((T *)sgw->tx_element_reply_queue) + reply_offset;

      auto vec_length     = 1;  // team.vector_length();
      uint64_t num_passes = num_requests / vec_length;
      if (num_requests % vec_length) num_passes++;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, 0, num_passes),
          [&](const int64_t pass) {
            uint64_t start = pass * vec_length;
            uint64_t stop  = start + vec_length;
            if (stop > num_requests) stop = num_requests;
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, start, stop),
                [=](uint64_t my_index) {
                  // This needs to be volatile to force
                  // visibility from the IB send
                  uint32_t offset = GET_ELEMENT_OFFSET(
                      atomic_load(&offsets[my_index],
                                  Kokkos::Impl::memory_order_seq_cst_t()));
                  reply_tx_buffer_T[my_index] = local_values[offset];
                });
          });
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        ++sgw->rx_block_request_ctr;
        sgw->tx_element_reply_ctrs[pe] += num_requests;
      });

      KOKKOS_REMOTE_THREADFENCE_SYSTEM();
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        atomic_store(&sgw->tx_block_reply_cmd_queue[idx], request,
                     Kokkos::Impl::memory_order_seq_cst_t());
      });
    }
    Kokkos::single(Kokkos::PerTeam(team),
                   [&]() { completion = atomic_load(completion_flag); });
    team.team_barrier();
  }
  team.team_barrier();

  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    atomic_store(completion_flag, 0u, Kokkos::Impl::memory_order_seq_cst_t());
  });
}

#endif  // RAW_CUDA

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos