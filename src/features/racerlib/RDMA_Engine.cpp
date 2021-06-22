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

#include <RDMA_Engine.hpp>
#include <RDMA_Helpers.hpp>

#include <infiniband/verbs.h>

#include <execinfo.h>
#include <map>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

using namespace Kokkos;

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

Transport *request_tport = nullptr;
Transport *response_tport = nullptr;

static std::vector<RdmaScatterGatherEngine *> heaps;

static SPSC_LockFree_Pool<uint32_t> available_reply_keys;
static std::vector<PendingRdmaRequest> pending_sg_requests;

static size_t aligned_size(size_t alignment, size_t size) {
  size_t npages = size / alignment;
  if (size % alignment)
    npages++;
  return npages * alignment;
}

static void *
allocate_host_pinned(size_t size,
                     size_t &real_size) //, bool host_write_optimized = false)
{
  size_t pagesize = (size_t)sysconf(_SC_PAGE_SIZE);
  real_size = aligned_size(pagesize, size);
  void *addr;
#ifdef KOKKOS_ENABLE_CUDA
  cuda_safe(cuMemAllocHost(&addr, real_size));
#else
  posix_memalign(&addr, pagesize, real_size);
#endif
  return addr;
}

static void free_host_pinned(void *buf, size_t size) {
#ifdef KOKKOS_ENABLE_CUDA
  // TODO - why is this segfaulting?
  // cudaFree(buf);
#else
  free(buf);
#endif
}

static void *allocate_device(size_t size, size_t &real_size) {
#ifdef KOKKOS_ENABLE_CUDA
  size_t pagesize = (size_t)sysconf(_SC_PAGE_SIZE);
  real_size = aligned_size(pagesize, size);
  void *addr;
  cuda_safe(cuMemAlloc((CUdeviceptr *)&addr, real_size));
  return addr;
#else
  // if there is no CUDA, just send back host pinned
  return allocate_host_pinned(size, real_size);
#endif
}

static void free_device(void *buf, size_t size) {
  // cuda_safe(cuMemFree((CUdeviceptr)buf));
}

static ibv_mr *pin_rdma_memory(Transport *tport, size_t size, void *buf) {
  int mr_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  debug("registering rdma memory on tport=%p, pd=%p, size=%d, buf=%p", tport,
        (tport ? tport->pd : nullptr), int(size), buf);
  ibv_mr *mr = ibv_reg_mr(tport->pd, buf, size, mr_flags);
  return mr;
}

static ibv_mr *allocate_host_rdma_memory(Transport *tport, size_t size) {
  size_t real_size;
  void *buf = allocate_host_pinned(size, real_size);
  assert_ibv(buf);
  return pin_rdma_memory(tport, real_size, buf);
}

static void free_host_rdma_memory(ibv_mr *mr) {
  void *buf = mr->addr;
  size_t size = mr->length;
  ibv_safe(ibv_dereg_mr(mr));
  free_host_pinned(buf, size);
}

static ibv_mr *allocate_device_rdma_memory(Transport *tport, size_t size) {
  size_t real_size;
  void *buf = allocate_device(size, real_size);
#ifdef KOKKOS_ENABLE_CUDA
  unsigned int set = 1;
  cuda_safe(cuPointerSetAttribute(&set, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                  (CUdeviceptr)buf));
#endif
  assert_ibv(buf);
  return pin_rdma_memory(tport, real_size, buf);
}

static void memcpy_to_device(void *dst, void *src, size_t size) {
#ifdef KOKKOS_ENABLE_CUDA
  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
#else
  memcpy(dst, src, size);
#endif
}

static void memset_device(void *buf, int value, size_t size) {
#ifdef KOKKOS_ENABLE_CUDA
  cudaMemsetAsync(buf, value, size);
#else
  ::memset(buf, value, size);
#endif
}

static void free_device_rdma_memory(ibv_mr *mr) {
  void *buf = mr->addr;
  size_t size = mr->length;
  ibv_dereg_mr(mr);
  free_device(buf, size);
}

void RdmaScatterGatherEngine::request_received(RdmaWorkRequest *req) {
  RemoteWindow *win = (RemoteWindow *)req->buf;
  // at this point, we don't need any checks on the epoch
  // we could receive a request with a mismatch epoch number
  // but we guarantee via barriers that messages are matched
  remote_window_start_reply(win);
}

void RdmaScatterGatherEngine::remote_window_start_reply(RemoteWindow *win) {
  uint32_t window_offset =
      (uintptr_t(win) - uintptr_t(rx_remote_windows_mr->addr)) /
      sizeof(RemoteWindow);
  uint64_t idx = rx_block_request_ctr % queue_size;
  uint64_t trip_number = rx_block_request_ctr / queue_size;
  uint64_t request = MAKE_BLOCK_PACK_REQUEST(win->num_entries, win->requester,
                                             trip_number, window_offset);
  volatile_store(&rx_block_request_cmd_queue[idx], request);

  debug("starting reply %" PRIu64 " back to %d on token %" PRIu32
        " on index %" PRIu64 " on window %u from %p to %p",
        request, win->requester, win->reply_token, idx, window_offset,
        win->cfg.reply_tx_buf, win->cfg.reply_rx_buf);
  pending_replies.push(win);
  ++rx_block_request_ctr;
}

void RdmaScatterGatherEngine::remote_window_finish_reply() {
  RdmaWorkRequest *rsp_wr = available_send_response_wrs.pop();
  if (pending_replies.empty()) {
    Kokkos::abort("empty replies");
  }
  RemoteWindow *win = pending_replies.front();
  pending_replies.pop();
  struct ibv_sge *sge = rsp_wr->sge;
  struct ibv_send_wr **bad_sr = &rsp_wr->bad_req.sr;
  struct ibv_send_wr *sr = &rsp_wr->req.sr;

  rsp_wr->type = RdmaWorkRequest::SEND_SG_RESPONSE;

  sge->length = win->num_entries * win->elem_size;
  sge->addr = (uint64_t)win->cfg.reply_tx_buf + win->offset * win->elem_size;
  sge->lkey = win->cfg.reply_tx_key;

  sr->wr.rdma.remote_addr =
      (uint64_t)win->cfg.reply_rx_buf + win->offset * win->elem_size;
  sr->wr.rdma.rkey = win->cfg.reply_rx_key;

  debug("finishing reply back to %d on token %" PRIu32
        " from %p to %p of size %d",
        win->requester, win->reply_token, sge->addr, sr->wr.rdma.remote_addr,
        int(sge->length));

  sr->next = nullptr;
  sr->wr_id = (uint64_t)rsp_wr;
  sr->num_sge = 1;
  sr->sg_list = sge;
  sr->send_flags = IBV_SEND_SIGNALED;
  sr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  sr->imm_data = win->reply_token;

  ibv_safe(ibv_post_send(response_tport->qps[win->requester], sr, bad_sr));
}

void RdmaScatterGatherEngine::response_received(RdmaWorkRequest *req,
                                                uint32_t token) {
  debug("got response token=%" PRIu32, token);
  PendingRdmaRequest &pending_req = pending_sg_requests[token];
  ack_response(pending_req);
}

void RdmaScatterGatherEngine::send_remote_window(Transport *tport, int pe,
                                                 uint32_t num_entries) {
  RdmaWorkRequest *req = available_send_request_wrs.pop();
  struct ibv_sge *sge = req->sge;
  struct ibv_send_wr **bad_sr = &req->bad_req.sr;
  struct ibv_send_wr *sr = &req->req.sr;

  uint32_t offset = tx_element_request_sent_ctrs[pe] % queue_size;
  uint32_t end = offset + num_entries;
  if (end > queue_size) {
    // first send up until wrap around
    send_remote_window(tport, pe, queue_size - offset);
    // then send the remaining
    num_entries = end - queue_size;
    offset = 0;
  }

  RemoteWindow *win = available_tx_windows.pop();
  win->num_entries = num_entries;
  win->offset = offset;
  req->buf = win;

  sge[0].addr = (uint64_t)win;
  sge[0].lkey = win->local_key;
  sge[0].length = sizeof(RemoteWindow);
  sge[1].addr = (uint64_t)tx_element_request_queue_mr->addr +
                (pe * queue_size + offset) * sizeof(uint32_t);
  sge[1].lkey = tx_element_request_queue_mr->lkey;
  sge[1].length = num_entries * sizeof(uint32_t);

  win->cfg = tx_remote_window_configs[pe];
  win->reply_token = available_reply_keys.pop();
  debug("request %p with %" PRIu32 " entries at offset=%" PRIu32
        " on token=%" PRIu32 " from pe=%d",
        win, win->num_entries, win->offset, win->reply_token, pe);

  PendingRdmaRequest &rdma_req = pending_sg_requests[win->reply_token];
  rdma_req.sge = this;
  rdma_req.num_entries = win->num_entries;
  rdma_req.start_idx = tx_element_request_sent_ctrs[pe];
  rdma_req.token = win->reply_token;
  rdma_req.pe = pe;
  tx_element_request_sent_ctrs[pe] += num_entries;

  sr->next = NULL;
  sr->wr_id = (uint64_t)req;
  sr->num_sge = 2;
  sr->sg_list = sge;
  sr->send_flags = IBV_SEND_SIGNALED;
  sr->opcode = IBV_WR_SEND;
  ibv_safe(ibv_post_send(tport->qps[pe], sr, bad_sr));
}

void RdmaScatterGatherEngine::poll(Transport *tport) {
  ibv_wc wc;
  // uint64_t start = rdtsc();
  int num_entries = ibv_poll_cq(tport->cq, 1, &wc);
  // uint64_t stop = rdtsc();
  if (num_entries < 0) {
    Kokkos::abort("cq is in error state");
  }
  if (num_entries >= 1) {
    if (wc.status != IBV_WC_SUCCESS) {
      Kokkos::abort("bad status in poll cq");
    }
    RdmaWorkRequest *req = (RdmaWorkRequest *)wc.wr_id;
    switch (req->type) {
    case RdmaWorkRequest::SEND_SG_REQUEST: {
      // nothing to do, just make request available again
      debug("cleared send request %p on cq", req->buf, tport->cq);
      time_safe(available_tx_windows.append((RemoteWindow *)req->buf));
      time_safe(available_send_request_wrs.append(req));
      break;
    }
    case RdmaWorkRequest::RECV_SG_REQUEST: {
      time_safe(request_received(req));
      // go ahead and put the request back into the queue
      ibv_safe(ibv_post_srq_recv(req->srq, &req->req.rr, &req->bad_req.rr));
      // ibv_safe(ibv_post_recv(req->qp, &req->req.rr, &req->bad_req.rr));
      break;
    }
    case RdmaWorkRequest::RECV_SG_RESPONSE: {
      time_safe(response_received(req, wc.imm_data));
      // go ahead and put the request back into the queue
      time_safe(ibv_safe(
          ibv_post_srq_recv(req->srq, &req->req.rr, &req->bad_req.rr)));
      // ibv_safe(ibv_post_recv(req->qp, &req->req.rr, &req->bad_req.rr));
      break;
    }
    case RdmaWorkRequest::SEND_SG_RESPONSE: {
      // nothing to do, just make request available again
      debug("cleared send response on cq", tport->cq);
      time_safe(available_send_response_wrs.append(req));
      break;
    }
    default:
      Kokkos::abort("got bad work request type");
      break;
    }
  }
}

static void run_on_core(int id) {
#if 0
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(id, &cpuset);
 
  pthread_t current_thread = pthread_self();
  pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
#endif
}

static void *run_response_thread(void *args) {
  run_on_core(4);
  RdmaScatterGatherEngine *engine = (RdmaScatterGatherEngine *)args;
  engine->poll_requests();
  return nullptr;
}

static void *run_ack_thread(void *args) {
  run_on_core(2);
  RdmaScatterGatherEngine *engine = (RdmaScatterGatherEngine *)args;
  engine->poll_responses();
  return nullptr;
}

static void *run_request_thread(void *args) {
  run_on_core(6);
  RdmaScatterGatherEngine *engine = (RdmaScatterGatherEngine *)args;
  engine->generate_requests();
  return nullptr;
}

void RdmaScatterGatherEngine::poll_requests() {
  while (is_running()) {
    time_safe(poll(request_tport));
    uint64_t idx = tx_block_reply_ctr % queue_size;
    uint64_t trip_number = tx_block_reply_ctr / queue_size;
    uint64_t request = volatile_load(&tx_block_reply_cmd_queue[idx]);
    if (GET_BLOCK_FLAG(request) == MAKE_READY_FLAG(trip_number)) {
      remote_window_finish_reply();
      ++tx_block_reply_ctr;
    }
  }
}

void RdmaScatterGatherEngine::poll_responses() {
  while (is_running()) {
    time_safe(poll(response_tport));
  }
}

void RdmaScatterGatherEngine::generate_requests() {
  while (is_running()) {
    check_for_new_block_requests();
  }
}

void RdmaScatterGatherEngine::check_for_new_block_requests() {
  uint64_t trip_number = tx_block_request_ctr / queue_size;
  uint64_t queue_idx = tx_block_request_ctr % queue_size;
  uint32_t ready_flag = MAKE_READY_FLAG(trip_number);
  uint64_t next_request = volatile_load(&tx_block_request_cmd_queue[queue_idx]);
  while (GET_BLOCK_FLAG(next_request) == ready_flag) {
    uint32_t pe = GET_BLOCK_PE(next_request);
    uint32_t num_entries = GET_BLOCK_SIZE(next_request);

    debug("got block request of size %" PRIu32 " for pe %" PRIu32
          " on idx %" PRIu64 " at offset %" PRIu64,
          num_entries, pe, tx_block_request_ctr,
          tx_element_request_sent_ctrs[pe]);
    send_remote_window(request_tport, pe, num_entries);

    ++tx_block_request_ctr;
    uint64_t trip_number = tx_block_request_ctr / queue_size;
    uint64_t queue_idx = tx_block_request_ctr % queue_size;
    ready_flag = MAKE_READY_FLAG(trip_number);
    next_request = volatile_load(&tx_block_request_cmd_queue[queue_idx]);
  }
}

void RdmaScatterGatherEngine::ack_response(PendingRdmaRequest &req) {
  uint64_t cleared_index = tx_element_request_acked_ctrs[req.pe];

  // we have to ack things in order
  if (cleared_index == req.start_idx) {
    cleared_index += req.num_entries;
  } else {
    debug("misordered ack %" PRIu32 ": expected %" PRIu32 ", got %" PRIu32,
          req.token, cleared_index, req.start_idx);
    misordered_acks.insert(&req);
    return;
  }

  // this is an ordered set of requests
  // see if any misordered requests can now be acked
  for (auto iter = misordered_acks.begin(); iter != misordered_acks.end();
       ++iter) {
    auto tmp = iter++;
    PendingRdmaRequest *req = *tmp;
    if (cleared_index == req->start_idx) {
      cleared_index += req->num_entries;
      available_reply_keys.append(req->token);
    } else {
      break;
    }
  }

  debug("acking index on pe=%d up to index=%" PRIu64 " on token=%" PRIu32,
        req.pe, cleared_index, req.token);

  tx_element_request_acked_ctrs[req.pe] = cleared_index;
  volatile_store(&ack_ctrs_h[req.pe], cleared_index);

  available_reply_keys.append(req.token);
}

void Cache::RemoteCache::invalidate() {
  memset_device(flags, 0, 2 * sizeof(unsigned int) * num_pes * pe_num_entries);
}

void RdmaScatterGatherEngine::fence() {
  cache.invalidate();
  MPI_Barrier(comm);
  epoch++;
}

RdmaScatterGatherEngine::~RdmaScatterGatherEngine() {
  // make sure everyone is done with their work
  MPI_Barrier(comm);

  void *ignore;

  stop_running();
  pthread_join(request_thread, &ignore);
  pthread_join(ack_thread, &ignore);
  pthread_join(response_thread, &ignore);

  free_device_rdma_memory(rx_element_reply_queue_mr);
  free_device_rdma_memory(tx_element_reply_queue_mr);
  free_host_rdma_memory(rx_remote_windows_mr);
  free_host_rdma_memory(tx_remote_windows_mr);
  free_host_rdma_memory(all_request_mr);

  free_host_pinned(ack_ctrs_h, num_pes * sizeof(uint64_t));
  free_device_rdma_memory(tx_element_request_queue_mr);
  free_device(tx_element_request_sent_ctrs, num_pes * sizeof(uint64_t));
  free_device(tx_element_aggregate_ctrs, num_pes * sizeof(uint64_t));

  free_device(ack_ctrs_d, num_pes * sizeof(uint64_t));
  free_device(cache.flags, cache.cache_size);

  delete[] tx_element_request_acked_ctrs;
}

RdmaScatterGatherEngine::RdmaScatterGatherEngine(MPI_Comm c, 
                                                 size_t elem_size)                                                
    : comm(c), tx_block_request_ctr(0), rx_block_request_ctr(0),
      tx_block_reply_ctr(0), epoch(0), terminate_signal(0) {
  if (available_reply_keys.size() == 0) {
    available_reply_keys.fill_iota(1000);
    pending_sg_requests.resize(1000);
  }

  if (!request_tport) {
    Kokkos::abort("request transport not initialized");
  }
  if (!response_tport) {
    Kokkos::abort("response transport not initialized");
  }

  MPI_Comm_size(comm, &num_pes);
  MPI_Comm_rank(comm, &rank);
  size_t ignore_actual_size;

  tx_element_request_ctrs = (uint64_t *)allocate_device(
      num_pes * sizeof(uint64_t), ignore_actual_size);
  memset_device(tx_element_request_ctrs, 0, num_pes * sizeof(uint64_t));

  ack_ctrs_d = (uint64_t *)allocate_device(num_pes * sizeof(uint64_t),
                                           ignore_actual_size);
  memset_device(ack_ctrs_d, 0, num_pes * sizeof(uint64_t));

  tx_element_reply_ctrs = (uint64_t *)allocate_device(
      num_pes * sizeof(uint64_t), ignore_actual_size);
  memset_device(tx_element_reply_ctrs, 0, num_pes * sizeof(uint64_t));

  tx_element_request_trip_counts = (uint32_t *)allocate_device(
      queue_size * num_pes * sizeof(uint32_t), ignore_actual_size);
  memset_device(tx_element_request_trip_counts, 0,
                num_pes * queue_size * sizeof(uint32_t));

  tx_element_request_queue_mr = allocate_device_rdma_memory(
      response_tport, queue_size * num_pes * sizeof(uint32_t));
  memset_device(tx_element_request_queue_mr->addr, 0,
                num_pes * queue_size * sizeof(uint32_t));

  tx_element_aggregate_ctrs = (uint64_t *)allocate_device(
      num_pes * sizeof(uint64_t), ignore_actual_size);
  memset_device(tx_element_aggregate_ctrs, 0, num_pes * sizeof(uint64_t));

  ack_ctrs_h = (uint64_t *)allocate_host_pinned(num_pes * sizeof(uint64_t),
                                                ignore_actual_size);
  tx_element_request_acked_ctrs = new uint64_t[num_pes];
  tx_element_request_sent_ctrs = new uint64_t[num_pes];
  for (int pe = 0; pe < num_pes; ++pe) {
    ack_ctrs_h[pe] = 0;
    tx_element_request_acked_ctrs[pe] = 0;
    tx_element_request_sent_ctrs[pe] = 0;
  }

  // these will both get pinned so need to be page-aligned
  size_t reply_size = elem_size * num_pes * queue_size;

  rx_element_reply_queue_mr =
      allocate_device_rdma_memory(response_tport, reply_size);
  tx_element_reply_queue_mr =
      allocate_device_rdma_memory(response_tport, reply_size);

  rx_element_request_queue_mr = allocate_device_rdma_memory(
      request_tport, NUM_RECV_WRS * queue_size * sizeof(uint32_t));
  rx_remote_windows_mr = allocate_host_rdma_memory(
      request_tport, NUM_RECV_WRS * sizeof(RemoteWindow));
  tx_remote_windows_mr = allocate_host_rdma_memory(
      request_tport, NUM_SEND_WRS * sizeof(RemoteWindow));
  available_tx_windows.fill_empty(NUM_SEND_WRS);
  for (int w = 0; w < NUM_SEND_WRS; ++w) {
    RemoteWindow *win = (RemoteWindow *)((char *)tx_remote_windows_mr->addr +
                                         w * sizeof(RemoteWindow));
    win->elem_size = elem_size;
    win->local_key = tx_remote_windows_mr->lkey;
    win->requester = rank;
    available_tx_windows.fill_append(win);
  }

  if (request_tport->pd != response_tport->pd) {
    Kokkos::abort("does not support request/response in different IBV "
                  "protection domains");
  }

  // TODO
  // we are for now assuming that request/response tports are on the same pd
  all_request_mr = allocate_host_rdma_memory(
      request_tport, sizeof(RdmaWorkRequest) * TOTAL_NUM_WRS);
  assert_ibv(all_request_mr->addr);
  ::memset(all_request_mr->addr, 0, sizeof(RdmaWorkRequest) * TOTAL_NUM_WRS);
  RdmaWorkRequest *all_requests = (RdmaWorkRequest *)all_request_mr->addr;

  available_send_request_wrs.fill_from_storage(NUM_SEND_WRS, all_requests);
  available_recv_request_wrs.fill_from_storage(
      NUM_RECV_WRS, all_requests + START_RECV_REQUEST_WRS);
  available_send_response_wrs.fill_from_storage(
      NUM_SEND_WRS, all_requests + START_SEND_RESPONSE_WRS);
  available_recv_response_wrs.fill_from_storage(
      NUM_RECV_WRS, all_requests + START_RECV_RESPONSE_WRS);

  int pe_num_entries = 1 << 18;

  cache.init(num_pes, pe_num_entries, elem_size);
  void *cache_arr = allocate_device(cache.cache_size, ignore_actual_size);
  cache.flags = (unsigned int *)cache_arr;
  cache.waiting = ((unsigned int *)cache_arr) + num_pes * pe_num_entries;
  cache.values = ((unsigned int *)cache_arr) + 2 * num_pes * pe_num_entries;

  tx_block_request_cmd_queue = (uint64_t *)allocate_host_pinned(
      queue_size * sizeof(uint64_t), ignore_actual_size);
  rx_block_request_cmd_queue = (uint64_t *)allocate_host_pinned(
      queue_size * sizeof(uint64_t), ignore_actual_size);
  tx_block_reply_cmd_queue = (uint64_t *)allocate_host_pinned(
      queue_size * sizeof(uint64_t), ignore_actual_size);
  memset(tx_block_request_cmd_queue, 0, queue_size * sizeof(uint64_t));
  memset(rx_block_request_cmd_queue, 0, queue_size * sizeof(uint64_t));
  memset(tx_block_reply_cmd_queue, 0, queue_size * sizeof(uint64_t));

  std::vector<RdmaScatterGatherBuffer> remote_bufs(num_pes);
  RdmaScatterGatherBuffer data;
#ifdef KOKKOS_ENABLE_CUDA
  ///*FIXME*/
  //cuda_safe(cuIpcGetMemHandle(&ipc_handle, (CUdeviceptr)buf));
  data.handle = ipc_handle;
#endif
  data.reply_tx_buffer = tx_element_reply_queue_mr->addr;
  data.reply_tx_key =
      tx_element_reply_queue_mr->lkey; // I want my local key sent back to me
  gethostname(data.hostname, 64);
  // we need to share what our actual remote buffer
  MPI_Allgather(&data, sizeof(RdmaScatterGatherBuffer), MPI_BYTE,
                remote_bufs.data(), sizeof(RdmaScatterGatherBuffer), MPI_BYTE,
                comm);

  direct_ptrs_h = new void *[num_pes];
  for (int pe = 0; pe < num_pes; ++pe) {
    if (::strcmp(data.hostname, remote_bufs[pe].hostname) == 0) {
#ifdef KOKKOS_ENABLE_CUDA
      if (pe != rank) {
        char *peer_buf;
        cuda_safe(cuIpcOpenMemHandle((CUdeviceptr *)&peer_buf,
                                     remote_bufs[pe].handle,
                                     CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
        direct_ptrs_h[pe] = nullptr; // peer_buf + header_size;
      } else {
        direct_ptrs_h[pe] = nullptr;
      }
#else
      direct_ptrs_h[pe] = nullptr;
#endif
    } else {
      direct_ptrs_h[pe] = nullptr;
    }
  }
  direct_ptrs_d =
      (void **)allocate_device(num_pes * sizeof(void *), ignore_actual_size);

  memcpy_to_device(direct_ptrs_d, direct_ptrs_h, num_pes * sizeof(void *));

  size_t reply_buffer_stride = elem_size * queue_size;
  tx_remote_window_configs.resize(num_pes);
  for (int pe = 0; pe < num_pes; ++pe) {
    if (pe == rank)
      continue;

    RemoteWindowConfig &cfg = tx_remote_window_configs[pe];
    cfg.reply_rx_buf =
        ((char *)rx_element_reply_queue_mr->addr + reply_buffer_stride * pe);
    cfg.reply_rx_key =
        rx_element_reply_queue_mr
            ->rkey; // remote processes will put here, send remote key
    cfg.reply_tx_buf =
        ((char *)remote_bufs[pe].reply_tx_buffer + reply_buffer_stride * rank);
    cfg.reply_tx_key = remote_bufs[pe].reply_tx_key;
    debug("pe=%d tx_reply=%p:%" PRIu32 " rx_reply=%p:%" PRIu32, pe,
          cfg.reply_tx_buf, cfg.reply_tx_key, cfg.reply_rx_buf,
          cfg.reply_rx_key);
  }

  uint32_t *request_buf = (uint32_t *)rx_element_request_queue_mr->addr;
  for (int r = 0; r < NUM_RECV_WRS; ++r) {
    RdmaWorkRequest *req = available_recv_request_wrs.pop();
    struct ibv_sge *sge = req->sge;
    struct ibv_recv_wr **bad_rr = &req->bad_req.rr;
    struct ibv_recv_wr *rr = &req->req.rr;

    RemoteWindow *win = get_rx_remote_window(r);
    sge[0].length = sizeof(RemoteWindow);
    sge[0].addr = (uint64_t)win;
    sge[0].lkey = rx_remote_windows_mr->lkey;
    sge[1].length = queue_size * sizeof(uint32_t);
    sge[1].addr = (uint64_t)(request_buf + r * queue_size);
    sge[1].lkey = rx_element_request_queue_mr->lkey;
    req->type = RdmaWorkRequest::RECV_SG_REQUEST;
    req->srq = request_tport->srq;
    req->buf = win;

    rr->next = NULL;
    rr->wr_id = (uint64_t)req;
    rr->num_sge = 2;
    rr->sg_list = sge;

    debug("post rr=%p of size=%d, key=%" PRIu32 ", addr=%p", req,
          int(sge->length), sge->lkey, sge->addr);

    ibv_safe(ibv_post_srq_recv(req->srq, rr, bad_rr));
    debug("posting receive on window %p", win);
  }

  for (int r = 0; r < NUM_RECV_WRS; ++r) {
    RdmaWorkRequest *req = available_recv_response_wrs.pop();
    struct ibv_sge *sge = req->sge;
    struct ibv_recv_wr **bad_rr = &req->bad_req.rr;
    struct ibv_recv_wr *rr = &req->req.rr;

    sge->length = sizeof(RdmaWorkRequest);
    sge->addr = (uint64_t)req;
    sge->lkey = all_request_mr->lkey;
    req->type = RdmaWorkRequest::RECV_SG_RESPONSE;
    req->srq = response_tport->srq;

    rr->next = nullptr;
    rr->wr_id = (uint64_t)req;
    rr->num_sge = 1;
    rr->sg_list = sge;
    ibv_safe(ibv_post_srq_recv(req->srq, rr, bad_rr));
  }

  pthread_create(&ack_thread, nullptr, run_ack_thread, this);
  pthread_create(&response_thread, nullptr, run_response_thread, this);
  pthread_create(&request_thread, nullptr, run_request_thread, this);

  request_done_flag =
      (unsigned *)allocate_device(sizeof(unsigned) * 2, ignore_actual_size);
  response_done_flag = request_done_flag + 1;
  fence_done_flag =
      (unsigned *)allocate_host_pinned(sizeof(unsigned), ignore_actual_size);
  volatile_store(fence_done_flag, 0u);
  memset_device(request_done_flag, 0, 2 * sizeof(unsigned));

  run_on_core(0);
}

RdmaScatterGatherEngine *allocate_rdma_scatter_gather_engine(
                                                             size_t elem_size,
                                                             MPI_Comm comm) {
  RdmaScatterGatherEngine *sge =
      new RdmaScatterGatherEngine(comm, elem_size);
  return sge;
}

void free_rdma_scatter_gather_engine(RdmaScatterGatherEngine *sge) {
  delete sge;
}

void rdma_ibv_init() {
  if (!request_tport) {
    request_tport = new Transport(MPI_COMM_WORLD);
  }
  if (!response_tport) {
    response_tport = new Transport(MPI_COMM_WORLD);
  }
}

void rdma_ibv_finalize() {
  if (request_tport) {
    delete request_tport;
    request_tport = nullptr;
  }

  if (response_tport) {
    delete response_tport;
    response_tport = nullptr;
  }
}

} // namespace RACERlib
} // namespace Experimental
} // namespace Kokkos
