#ifndef RACERLIB_RDMA_ENGINE_H
#define RACERLIB_RDMA_ENGINE_H

#define RAW_CUDA

#include <Kokkos_Atomic.hpp>
#include <Kokkos_View.hpp>
#include <Access_cache.hpp>
#include <mpi.h>
#include <pthread.h>
#include <infiniband/verbs.h>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <iostream>
#include <Config.hpp>

#define ELEMENT_OFFSET_BIT_SHIFT 1
#define ELEMENT_OFFSET_MASK      0xFFFFFFFE
#define BLOCK_SIZE_BIT_SHIFT   40
#define BLOCK_SIZE_MASK        0xFFFFFF0000000000
#define BLOCK_WINDOW_BIT_SHIFT 20
#define BLOCK_WINDOW_MASK      0xFFFFF00000
#define BLOCK_PE_BIT_SHIFT     1
#define BLOCK_PE_MASK          0xFFFFE

#define NUM_SEND_WRS 16
#define NUM_RECV_WRS 16
#define TOTAL_NUM_WRS (NUM_SEND_WRS + NUM_RECV_WRS + NUM_SEND_WRS + NUM_RECV_WRS)

#define START_SEND_REQUEST_WRS 0
#define START_RECV_REQUEST_WRS (START_SEND_REQUEST_WRS + NUM_SEND_WRS)
#define START_SEND_RESPONSE_WRS (START_RECV_REQUEST_WRS + NUM_RECV_WRS)
#define START_RECV_RESPONSE_WRS (START_SEND_RESPONSE_WRS + NUM_SEND_WRS)

#define MAKE_READY_FLAG(trip_number) \
  (((trip_number)+1)%2)

#define MAKE_BLOCK_GET_REQUEST(size, pe, trip_number) \
    ( (uint64_t(size)<< BLOCK_SIZE_BIT_SHIFT) \
    | (uint64_t(pe)<<BLOCK_PE_BIT_SHIFT) \
    | MAKE_READY_FLAG(trip_number) \
    )

#define MAKE_BLOCK_PACK_ACK(size, pe, trip_number) \
    MAKE_BLOCK_GET_REQUEST(size,pe,trip_number)

#define MAKE_BLOCK_PACK_REQUEST(size, pe, trip_number, window) \
    ( (uint64_t(size)<< BLOCK_SIZE_BIT_SHIFT) \
    | (uint64_t(pe)<<BLOCK_PE_BIT_SHIFT) \
    | MAKE_READY_FLAG(trip_number) \
    | (uint64_t(window)<<BLOCK_WINDOW_BIT_SHIFT) \
    )

#define GET_ELEMENT_FLAG(request) \
   (request & 1)

#define GET_BLOCK_FLAG(request) \
   (request & 1)

#define GET_BLOCK_PE(request) \
   (((request) & BLOCK_PE_MASK) >> BLOCK_PE_BIT_SHIFT)

#define GET_BLOCK_SIZE(request) \
   ((request)>>BLOCK_SIZE_BIT_SHIFT)

#define GET_BLOCK_WINDOW(request) \
   (((request) & BLOCK_WINDOW_MASK) >> BLOCK_WINDOW_BIT_SHIFT)

#define MAKE_ELEMENT_REQUEST(offset, trip_number) \
    (((offset) << ELEMENT_OFFSET_BIT_SHIFT) | MAKE_READY_FLAG(trip_number))

#define GET_ELEMENT_OFFSET(request) \
    ((request >> ELEMENT_OFFSET_BIT_SHIFT))

#define GET_ELEMENT_FLAG(request) \
    (request & 1)

#define ARCH_PPC64

#ifdef ARCH_PPC64
static inline unsigned long long rdtsc(){
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ volatile(
                "0:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     0b         \n"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = tmp; //get rid of compiler warnings
  result = upper;
  result = result<<32;
  result = result|lower;

  return(result);
}
#else
static inline uint64_t rdtsc(void) {
  uint32_t eax, edx;
  asm volatile("rdtsc" : "=a" (eax), "=d" (edx));
  return (uint64_t)eax | (uint64_t)edx << 32;
}
#endif

#define cuda_safe(...)                                                          \
  do {                                                                          \
    CUresult result = (__VA_ARGS__); \
    if (CUDA_SUCCESS != result) {                                              \
        std::cout << result << std::endl;   \
        fprintf(stderr, "%s:%d: CUDA failed\n", __FILE__, __LINE__); \
        exit(-1);                                                             \
    }                                                                         \
  } while (0)

#define time_safe(...) \
  do { \
    auto start = rdtsc(); \
    __VA_ARGS__; \
    auto stop = rdtsc(); \
    if ((stop-start) > 10000000000ULL){ \
      fprintf(stderr, "call took way too long (%llu-%llu) at %s:%d\n", \
              start, stop, __FILE__, __LINE__); \
      ::abort(); \
    } \
  } while(0)

#define ibv_safe(...) \
  do { \
    auto start = rdtsc(); \
    int rc = __VA_ARGS__; \
    if (rc != 0){ \
      fprintf(stderr, "call failed with rc=%d at %s:%d\n", rc, __FILE__, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    }  \
    auto stop = rdtsc(); \
    if ((stop-start) > 100000000000ULL){ \
      fprintf(stderr, "call took way too long (%llu-%llu) at %s:%d\n", \
              start, stop, __FILE__, __LINE__); \
      ::abort(); \
    } \
  } while(0)

#define assert_ibv(elem) if (!elem){ \
    fprintf(stderr, "call failed at %s:%d\n", __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } else

#define debugf(str,...) printf("PE %d: " str "\n", request_tport->rank, __VA_ARGS__); fflush(stdout)

#ifdef KOKKOS_IBV_DEBUG
#define debug(...) debugf(__VA_ARGS__)
#else
#define debug(...)
#endif

namespace RACERlib{

#define NEW_REQUEST_BIT 0
#define NEW_REQUEST_MASK 1

template <class T>
struct SPSC_LockFree_Pool {
  uint64_t read_head;
  uint64_t write_head;
  uint32_t queue_size;
  T* queue;
  bool allocated;

  SPSC_LockFree_Pool() :
    read_head(0),
    write_head(0),
    queue_size(0),
    allocated(false),
    queue(nullptr)
  {
  }

  uint32_t size() const {
    return queue_size;
  }

  ~SPSC_LockFree_Pool(){
    if (allocated){
      delete[] queue;
    }
  }

  void fill_empty(uint32_t size){
    queue = new T[size];
    queue_size = size;
    allocated  = true;
  }

  void fill_append(T t){
    queue[write_head] = t;
    ++write_head;
  }

  template <class U=T>
  typename std::enable_if<std::is_pointer<U>::value>::type
  fill_from_storage(uint32_t size, U u){
    queue = new T[size];
    queue_size = size;
    allocated = true;
    for (uint32_t i=0; i < size; ++i){
      queue[i] = &u[i];
    }
    write_head = size;
  }

  void fill(uint32_t size, T* t){
    queue = t;
    write_head = size;
    queue_size = size;
  }

  void fill_iota(uint32_t size){
    queue = new T[size];
    for (uint32_t i=0; i < size; ++i){
      queue[i] = i;
    }
    write_head = size;
    queue_size = size;
    allocated = true;
  }

  void append(T t){
    // we guarantee to only put in what was there at the beginning
    // we therefore don't need to check for overruns
    auto idx = write_head % queue_size;
    queue[idx] = t;
    atomic_add(&write_head, uint64_t(1));
  }

  T pop(){
    while (read_head == volatile_load(&write_head));
    auto idx = read_head % queue_size;
    T t = queue[idx];
    atomic_add(&read_head, uint64_t(1));
    return t;
  }

};


struct RemoteWindowConfig {
  void*    reply_tx_buf;
  uint32_t reply_tx_key;
  void*    reply_rx_buf;
  uint32_t reply_rx_key;
};

struct RemoteWindow {
  RemoteWindowConfig cfg;
  /** The following are fixed for the entire run */
  uint32_t local_key;
  uint32_t elem_size;
  uint32_t requester;

  /** The following change with each new request */
  uint32_t epoch;
  uint32_t reply_token;
  uint32_t offset;
  uint32_t num_entries;
};

struct RdmaWorkRequest {
  enum Type {
    SEND_SG_REQUEST,
    RECV_SG_REQUEST,
    SEND_SG_RESPONSE,
    RECV_SG_RESPONSE
  };

  Type type;
  ibv_sge sge[3];
  ibv_srq* srq;
  ibv_qp* qp;

  union {
    ibv_send_wr sr;
    ibv_recv_wr rr;
  } req;

  union {
    ibv_send_wr* sr;
    ibv_recv_wr* rr;
  } bad_req;

  void* buf;
};

struct Transport {
  int nproc;
  int rank;
  ibv_context* ctx;
  ibv_cq* cq;
  ibv_qp** qps;
  ibv_device* dev;
  ibv_pd* pd;
  ibv_srq* srq;

  Transport(MPI_Comm comm);
  ~Transport();
};


struct RdmaScatterGatherEngine;

struct RdmaScatterGatherBuffer {
#ifdef KOKKOS_ENABLE_CUDA
  CUipcMemHandle handle;
#endif
  void* reply_tx_buffer;
  uint32_t reply_tx_key;
  char hostname[64];
};

struct PendingRdmaRequest {
  uint64_t start_idx;
  uint32_t num_entries;
  int pe;
  uint32_t token;
  RdmaScatterGatherEngine* sge;
};

struct RdmaScatterGatherEngine {

  //for optimization purposes, we have a statically sized queue
  constexpr static uint32_t queue_size = 1<<20;


  RemoteWindow* get_rx_remote_window(int idx) const {
    RemoteWindow* windows = (RemoteWindow*) rx_remote_windows_mr->addr;
    return &windows[idx];
  }

  void ack_scatter_gather(PendingRdmaRequest& req);

  void ack_completed(int pe, uint64_t num_completed);

  RdmaScatterGatherEngine(MPI_Comm comm, void* buf, size_t elem_size, size_t header_size);

  ~RdmaScatterGatherEngine();

  bool is_running(){
    return Kokkos::atomic_fetch_add(&terminate_signal,0) == 0;
  }

  void stop_running(){
    Kokkos::atomic_add(&terminate_signal, uint32_t(1));
  }

  void fence();

  void poll_requests();
  void poll_responses();
  void generate_requests();

 /** The data structures used on host and device */
 public: //these are public, safe to use on device
  uint64_t* tx_element_request_ctrs;
  uint64_t* tx_element_reply_ctrs;
  uint32_t* tx_element_request_trip_counts;
  uint64_t* tx_element_aggregate_ctrs;

  uint64_t* ack_ctrs_h;
  uint64_t* ack_ctrs_d;

  void** direct_ptrs_d;

  ibv_mr* tx_element_request_queue_mr;
  ibv_mr* tx_element_reply_queue_mr;
  ibv_mr* rx_element_request_queue_mr;
  ibv_mr* rx_element_reply_queue_mr;

  unsigned* request_done_flag;
  unsigned* response_done_flag;
  unsigned* fence_done_flag;
  MPI_Comm comm;
  int num_pes;
  int rank;
  Features::Cache::RemoteCache cache;

  uint64_t* tx_block_request_cmd_queue;
  uint64_t* rx_block_request_cmd_queue;
  uint64_t* tx_block_reply_cmd_queue;
  uint64_t tx_block_request_ctr;
  uint64_t rx_block_request_ctr;
  uint64_t tx_block_reply_ctr;

  uint32_t epoch;

  /** The data structures only used on the host */
 private:
  ibv_mr* rx_remote_windows_mr;
  ibv_mr* tx_remote_windows_mr;
  ibv_mr* all_request_mr;
#ifdef KOKKOS_ENABLE_CUDA
  CUipcMemHandle ipc_handle;
#endif

  pthread_t request_thread;
  pthread_t response_thread;
  pthread_t ack_thread;
  uint32_t terminate_signal;

  void** direct_ptrs_h;

  /** An array of size num_pes, contains a running count of the number
   *  of element requests actually sent each remote PE */
  uint64_t* tx_element_request_sent_ctrs;
  /** An array of size num_pes, contains a running count of the number
   *  of element requests received back from each remote PE
   *  and acked to the device */
  uint64_t* tx_element_request_acked_ctrs;

  SPSC_LockFree_Pool<RdmaWorkRequest*> available_send_request_wrs;
  SPSC_LockFree_Pool<RdmaWorkRequest*> available_send_response_wrs;
  SPSC_LockFree_Pool<RdmaWorkRequest*> available_recv_request_wrs;
  SPSC_LockFree_Pool<RdmaWorkRequest*> available_recv_response_wrs;
  std::vector<RemoteWindowConfig> tx_remote_window_configs;
  std::queue<RemoteWindow*> pending_replies;

#ifdef KOKKOS_ENABLE_CUDA
  cudaStream_t response_stream;
#endif

  struct SortAcks {
    bool operator()(PendingRdmaRequest* l, PendingRdmaRequest* r) const {
      return l->start_idx < r->start_idx;
    }
  };

  std::set<PendingRdmaRequest*, SortAcks> misordered_acks;

  SPSC_LockFree_Pool<RemoteWindow*> available_tx_windows;

  void remote_window_start_reply(RemoteWindow* win);
  void remote_window_finish_reply();
  void request_received(RdmaWorkRequest* req);
  void response_received(RdmaWorkRequest* req, uint32_t token);
  void send_remote_window(Transport* tport, int pe, uint32_t num_elements);
  void poll(Transport* tport);
  void check_for_new_block_requests();
  void ack_response(PendingRdmaRequest& req);

};

struct RdmaScatterGatherWorker
{
  struct RemoteCacheHolder;

  static constexpr uint32_t queue_size = RdmaScatterGatherEngine::queue_size;

  template <class T>
  KOKKOS_FUNCTION
  T get(int pe, uint32_t offset){
    uint64_t* tail_ctr = &tx_element_request_ctrs[pe];
    uint64_t idx = Kokkos::atomic_fetch_add((unsigned long long*) tail_ctr, 1);
    uint32_t trip_number = idx / queue_size;
    uint32_t buf_slot = idx % queue_size;
    uint64_t global_buf_slot = pe*queue_size + buf_slot;
    //if we have wrapped around the queue, wait for it to be free
    //this is a huge amount of extra storage, but it's the only way
    //to do it. I can't just use the ack counter to know when a slot
    //is free because I could overwrite the value before thread reads it
    while (volatile_load((unsigned int*)&tx_element_request_trip_counts[global_buf_slot]) != trip_number);

    //enough previous requests are cleared that we can join the queue
    uint32_t* req_ptr = &tx_element_request_queue[global_buf_slot];
    // the queue begins as all zeroes
    // on even passes we set a flag bit of 1 to indicate this is a new request
    // on odd passes we set a flag bit of 0 to indicate this is a new request
    //the requested offset is a combination of the actual offset
    //and flag indicate that this is a new request
    uint32_t offset_request = MAKE_ELEMENT_REQUEST(offset,trip_number);
    volatile_store(req_ptr, offset_request);
    //we now have to spin waiting for the request to be satisfied
    //we wil get the signal that this is ready when the ack count
    //exceeds our request idx
    
    uint64_t* ack_ptr = &ack_ctrs_d[pe];
    uint64_t ack = volatile_load(ack_ptr);
    while (ack <= idx){
      ack = volatile_load(ack_ptr);
    }
    
    //at this point, our data is now available in the reply buffer
    T* reply_buffer_T = (T*) rx_element_reply_queue;
    T ret = volatile_load(&reply_buffer_T[global_buf_slot]);
    // update the trip count to signal any waiting threads they can go
    atomic_add((unsigned int*)&tx_element_request_trip_counts[global_buf_slot], 1u);
    return ret;
  }

  template <class T>
  KOKKOS_FUNCTION
  T request(int pe, uint32_t offset){
  #ifdef KOKKOS_DISABLE_CACHE
    return get<T>(pe, offset);
  #else
    return cache.get<T>(pe, offset, this);
  #endif
  }

  Features::Cache::RemoteCache cache;
  /** Array of size num_pes, a running count of element requests
   * generated by worker threads */
  uint64_t* tx_element_request_ctrs;
  /** Array of size num_pes, a running count of element requests
   * completed and acked by the host. This is not read by the
   * worker threads to limit host-device contention */
  uint64_t* ack_ctrs_h;
  /** Array of size num_pes, a running count of element requests
   * completed and acked to worker threads. This is a mirror of
   * ack_ctrs_h that guarantees ack_ctrs_d[i] <= ack_cstr_h[i]
   * and that ack_ctrs_d will eventually equal ack_ctrs_h
  */
  uint64_t* ack_ctrs_d;
  /** Array of size num_pes, a running count of element replies
   * back to each PE */
  uint64_t* tx_element_reply_ctrs;
  /** Array of size num_pes, a running count of element requests
    * aggregated and processed by the request threads */
  uint64_t* tx_element_aggregate_ctrs;
  /** Array of size num_pes*queue_length */
  uint32_t* tx_element_request_queue;
  /** Array of size num_pes, the number of times we have wrapped around
   *  the circular buffer queue for each PE */
  uint32_t* tx_element_request_trip_counts;
  /** Arraqy of size num_pes*queue_length, requests from remote PEs
   *  are received here */
  uint32_t* rx_element_request_queue;
  /** Array of size num_pes*queue_length, data is gathered here
   *  and sent back to requesting PEs */
  void* tx_element_reply_queue;
  /** Array of size num_pes*queue_length, gathered data from the remote PE
   *  is received here */
  void* rx_element_reply_queue;
  /** Array of size num_pes, a pointer that can be directly read
   * to access peer data, nullptr if no peer pointer exists
   */
  void** direct_ptrs;
  int rank;
  int num_pes;

  /** Array of size queue_length
   *  Request threads on device write commands into this queue
   *  Progress threads on the CPU read command from this queue
   *  Indicates host should send a block of element requests to remote PE
   *  Combined queue for all PEs */
  uint64_t* tx_block_request_cmd_queue;
  /** Array of size queue_length
   *  Response threads on device read command from this queue
   *  Progress threads on the CPU write commands into this queue after receiving request
   *  Indicates GPU should gather data from scattered offsets into a contiguous block
   *  Combined queue for all PEs */
  uint64_t* rx_block_request_cmd_queue;
  /** Array of size queue_length
   *  Response threads on device write commands into this queue
   *  Progress threads on the CPU read command from this queue
   *  Indicates host a block of element replies is ready to send back to requesting PE
   *  Combined queue for all PEs */
  uint64_t* tx_block_reply_cmd_queue;
  /** A running count of the number of block requests sent to all PEs */
  uint64_t tx_block_request_ctr;
  /** A running count of the number of block requests received from all PEs */
  uint64_t rx_block_request_ctr;

  unsigned* request_done_flag;
  unsigned* response_done_flag;
  unsigned* fence_done_flag;
};


RdmaScatterGatherEngine* allocate_rdma_scatter_gather_engine(void* buf, size_t elem_size, size_t header_size, MPI_Comm comm);
void free_rdma_scatter_gather_engine(RdmaScatterGatherEngine* sge);


#ifdef RAW_CUDA

template<typename T, class Team>
__device__
void pack_response(T* local_values, RdmaScatterGatherWorker* sgw, unsigned* completion_flag,
                   Team&& team)
{
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t request;
  int my_thread = threadIdx.x * blockDim.y + threadIdx.y;
  int total_threads = blockDim.x * blockDim.y;
  uint32_t queue_size = RdmaScatterGatherEngine::queue_size;
  while (completion == 0){
    uint64_t idx = sgw->rx_block_request_ctr % queue_size;
    uint64_t trip_number = sgw->rx_block_request_ctr / queue_size;
    if (my_thread == 0){
      request = volatile_load(&sgw->rx_block_request_cmd_queue[idx]);
    }
    __syncthreads();

    if (GET_BLOCK_FLAG(request) == MAKE_READY_FLAG(trip_number)){
      uint32_t num_requests = GET_BLOCK_SIZE(request);
      uint32_t pe = GET_BLOCK_PE(request);
      uint32_t window = GET_BLOCK_WINDOW(request);
      uint32_t reply_offset = pe*queue_size + sgw->tx_element_reply_ctrs[pe] % queue_size;
      uint32_t* offsets = sgw->rx_element_request_queue + window*queue_size;
      T* reply_tx_buffer_T = ((T*)sgw->tx_element_reply_queue) + reply_offset;

      uint32_t num_packed = 0;
      while (num_packed < num_requests){
        uint32_t my_index = num_packed + my_thread; 
        if (my_index < num_requests){
          //this needs to be volatile to force visibility from the IB send
          uint32_t offset = GET_ELEMENT_OFFSET(volatile_load(&offsets[my_index]));
          reply_tx_buffer_T[my_index] = local_values[offset];
        }
        num_packed += total_threads;
      }
      if (my_thread == 0){
        ++sgw->rx_block_request_ctr;
        sgw->tx_element_reply_ctrs[pe] += num_requests;
      }
      //force visibility 
      __threadfence_system();
      if (my_thread == 0){
        volatile_store(&sgw->tx_block_reply_cmd_queue[idx], request);
      }
    }

    if (my_thread == 0){
      completion = volatile_load(completion_flag);
    }
    __syncthreads();
  }
  __syncthreads();
  if (my_thread == 0){
    volatile_store(completion_flag, 0u);
  }
}

template <class Team>
__device__
void aggregate_requests(RdmaScatterGatherWorker* sgw, Team&& team, unsigned num_worker_teams)
{
  int my_thread = threadIdx.x * blockDim.y + threadIdx.y;
  int total_threads = blockDim.x * blockDim.y;
  uint32_t queue_size = RdmaScatterGatherEngine::queue_size;
  static constexpr uint32_t mtu = 16384; //try to at least send 16K elements
  static constexpr uint32_t max_mtu_stalls = 4;
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t total_requests;
  KOKKOS_REMOTE_SHARED int misses[32]; // TODO, make this an array, I'm too lazy right now
  for (int i=0; i < 32; ++i) misses[i] = 0;
  completion = 0;
  __syncthreads();
  while (completion < num_worker_teams){
    for (int pe=0; pe < sgw->num_pes; ++pe){
      uint64_t head = sgw->tx_element_aggregate_ctrs[pe];
      if (my_thread == 0){
        uint64_t last_cleared_on_device = sgw->ack_ctrs_d[pe];
        if (head > last_cleared_on_device){
          uint64_t last_cleared_on_host = volatile_load(&sgw->ack_ctrs_h[pe]);
          if (last_cleared_on_device < last_cleared_on_host){
            volatile_store(&sgw->ack_ctrs_d[pe], last_cleared_on_host);
          }
        }
        uint64_t max_index = Kokkos::atomic_fetch_add(&sgw->tx_element_request_ctrs[pe], 0u);
        total_requests = max_index - head;
        if (total_requests < mtu && misses[pe] < max_mtu_stalls){
          total_requests = 0;
          ++misses[pe];
        } else {
          misses[pe] = 0;
        }
      }
      __syncthreads();
      if (total_requests > 0){
        unsigned requests_done = 0;
        while (requests_done < total_requests){
          uint64_t my_offset = head + requests_done + my_thread;
          if (my_offset < total_requests){
            uint64_t my_idx = my_offset % queue_size;
            uint64_t my_trip_number = my_offset / queue_size;
            uint32_t ready_flag = MAKE_READY_FLAG(my_trip_number);
            uint32_t req_slot = my_idx + pe*queue_size;
            uint32_t next_request = volatile_load(&sgw->tx_element_request_queue[req_slot]);
            while (GET_BLOCK_FLAG(next_request) != ready_flag){
              next_request = volatile_load(&sgw->tx_element_request_queue[req_slot]);
            }
            //this looks stupid, but is necessary to make visible to peer devices
            sgw->tx_element_request_queue[req_slot] = next_request;
          }
          requests_done += total_threads;
        }
        //we have written the requests, now make them peer visible
        __threadfence_system();

        if (my_thread == 0){
          uint64_t tail_idx = sgw->tx_block_request_ctr++;
          sgw->tx_element_aggregate_ctrs[pe] += total_requests;
          uint64_t queue_idx = tail_idx % queue_size;
          uint64_t trip_number = tail_idx / queue_size;
          uint64_t request = MAKE_BLOCK_GET_REQUEST(total_requests, pe, trip_number);
          volatile_store(&sgw->tx_block_request_cmd_queue[queue_idx], request);
        }
        __syncthreads();
      }
    }
    if (my_thread == 0){
      completion = volatile_load(sgw->request_done_flag);
    }
    __syncthreads();
  }
  __syncthreads();
  if (my_thread == 0){
    volatile_store(sgw->request_done_flag, 0u);
    volatile_store(sgw->response_done_flag, 1u);
  }

}

#else

template<typename T, class Team>
KOKKOS_FUNCTION 
void pack_response(T* local_values, RdmaScatterGatherWorker* sgw, unsigned* completion_flag,
                   Team&& team)
{
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t request;
  completion = 0;
  uint32_t queue_size = RdmaScatterGatherEngine::queue_size;
  while (completion == 0){
    uint64_t idx = sgw->rx_block_request_ctr % queue_size;
    uint64_t trip_number = sgw->rx_block_request_ctr / queue_size;
    Kokkos::single(Kokkos::PerTeam(team), [&] (){
      request = volatile_load(&sgw->rx_block_request_cmd_queue[idx]);
    });
    team.team_barrier();
    if (GET_BLOCK_FLAG(request) == MAKE_READY_FLAG(trip_number)){
      uint32_t num_requests = GET_BLOCK_SIZE(request);
      uint32_t pe = GET_BLOCK_PE(request);
      uint32_t window = GET_BLOCK_WINDOW(request);
      uint32_t reply_offset = pe*queue_size + sgw->tx_element_reply_ctrs[pe] % queue_size;
      uint32_t* offsets = sgw->rx_element_request_queue + window*queue_size;
      T* reply_tx_buffer_T = ((T*)sgw->tx_element_reply_queue) + reply_offset;

      auto vec_length = team.vector_length();
      uint64_t num_passes = num_requests / vec_length;
      if (num_requests % vec_length) num_passes++;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,num_passes),
        [&] (const int64_t pass) {
        uint64_t start = pass * vec_length;
        uint64_t stop = start + vec_length;
        if (stop > num_requests) stop = num_requests;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,start,stop), [=](uint64_t my_index){
          //this needs to be volatile to force visibility from the IB send
          uint32_t offset = GET_ELEMENT_OFFSET(volatile_load(&offsets[my_index]));
          reply_tx_buffer_T[my_index] = local_values[offset];
        });
      });
      Kokkos::single(Kokkos::PerTeam(team), [&](){
        ++sgw->rx_block_request_ctr;
        sgw->tx_element_reply_ctrs[pe] += num_requests;
      });

      KOKKOS_REMOTE_THREADFENCE_SYSTEM();
      Kokkos::single(Kokkos::PerTeam(team), [&](){
        volatile_store(&sgw->tx_block_reply_cmd_queue[idx], request);
      });
    }
    Kokkos::single(Kokkos::PerTeam(team), [&](){
      completion = volatile_load(completion_flag);
    });
    team.team_barrier();
  }
  team.team_barrier();
  Kokkos::single(Kokkos::PerTeam(team), [&](){
    volatile_store(completion_flag, 0u);
  });
}


template <class Team>
KOKKOS_INLINE_FUNCTION
void aggregate_requests(RdmaScatterGatherWorker* sgw, Team&& team, unsigned num_worker_teams)
{
  uint32_t queue_size = RdmaScatterGatherEngine::queue_size;
  static constexpr uint32_t mtu = 16384; //try to at least send 16K elements
  static constexpr uint32_t max_mtu_stalls = 4;
  KOKKOS_REMOTE_SHARED unsigned completion;
  KOKKOS_REMOTE_SHARED uint64_t total_requests;
  KOKKOS_REMOTE_SHARED int misses[32]; // TODO, make this an array, I'm too lazy right now
  for (int i=0; i < 32; ++i) misses[i] = 0;
  completion = 0;
  team.team_barrier();
  while (completion < num_worker_teams){
    for (int pe=0; pe < sgw->num_pes; ++pe){
      uint64_t head = sgw->tx_element_aggregate_ctrs[pe];
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        total_requests = 0;
        uint64_t last_cleared_on_device = sgw->ack_ctrs_d[pe];
        if (head > last_cleared_on_device){
          uint64_t last_cleared_on_host = volatile_load(&sgw->ack_ctrs_h[pe]);
          if (last_cleared_on_device < last_cleared_on_host){
            volatile_store(&sgw->ack_ctrs_d[pe], last_cleared_on_host);
          }
        }
        uint64_t max_index = Kokkos::atomic_fetch_add(&sgw->tx_element_request_ctrs[pe], 0u);
        total_requests = max_index - head;
        if (total_requests < mtu && misses[pe] < max_mtu_stalls){
          total_requests = 0;
          ++misses[pe];
        } else {
          misses[pe] = 0;
        }
      });
      team.team_barrier();
      if (total_requests > 0){
        auto vec_length = team.vector_length();
        uint64_t num_passes = total_requests / vec_length;
        if (total_requests % vec_length) num_passes++;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,num_passes),
          [&] (const int64_t pass) {
          uint64_t start = pass * vec_length;
          uint64_t stop = start + vec_length;
          if (stop > total_requests) stop = total_requests;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,start,stop), [=](uint64_t offset){
            uint64_t my_offset = head + offset;
            uint64_t my_trip_number = my_offset / queue_size;
            uint64_t my_idx = my_offset % queue_size;
            uint64_t ready_flag = MAKE_READY_FLAG(my_trip_number);
            uint64_t req_slot = my_idx + pe*queue_size;
            uint32_t next_request = volatile_load(&sgw->tx_element_request_queue[req_slot]);
            while (GET_BLOCK_FLAG(next_request) != ready_flag){
              next_request = volatile_load(&sgw->tx_element_request_queue[req_slot]);
            }
            //this looks stupid, but is necessary to make visible to peer devices
            sgw->tx_element_request_queue[req_slot] = next_request;
          });
        });
        //we have written the requests, now make them peer visible
        KOKKOS_REMOTE_THREADFENCE_SYSTEM();

        Kokkos::single(Kokkos::PerTeam(team), [&](){
          uint64_t tail_idx = sgw->tx_block_request_ctr++;
          sgw->tx_element_aggregate_ctrs[pe] += total_requests;
          uint64_t queue_idx = tail_idx % queue_size;
          uint64_t trip_number = tail_idx / queue_size;
          uint64_t request = MAKE_BLOCK_GET_REQUEST(total_requests, pe, trip_number);
          volatile_store(&sgw->tx_block_request_cmd_queue[queue_idx], request);
        });
        team.team_barrier();
      }
    }

    Kokkos::single(Kokkos::PerTeam(team), [&](){
      completion = volatile_load(sgw->request_done_flag);
    });
    team.team_barrier();
  }
  team.team_barrier();
  Kokkos::single(Kokkos::PerTeam(team), [&](){
    volatile_store(sgw->request_done_flag, 0u);
    volatile_store(sgw->response_done_flag, 1u);
  });
}

#endif

template <class Policy, class Lambda, class RemoteView>
struct RemoteParallelFor {
  KOKKOS_FUNCTION void operator()(const typename Policy::member_type& team) const {
    RdmaScatterGatherWorker* sgw = m_view(0,0).sg;
    if (team.league_rank() == 0){
      aggregate_requests(sgw,team,team.league_size() - 2);
    } else if (team.league_rank() == 1){
      pack_response(m_view(0,0).ptr,sgw,sgw->response_done_flag,team);
    } else {
      auto new_team = team.shrink_league(2);
      m_lambda(new_team);
      team.team_barrier();
      Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA (){
        atomic_fetch_add(sgw->request_done_flag, 1);
      });
    }
  }

  template <class L, class R>
  RemoteParallelFor(L&& lambda, R&& view) :
    m_lambda(std::forward<L>(lambda)),
    m_view(std::forward<R>(view))
  {
  }

 private:
  Lambda m_lambda;
  RemoteView m_view;
};

template <class Policy, class RemoteView>
struct RespondParallelFor {
  RespondParallelFor(const RemoteView& view) :
    m_view(view)
  {
  }

  KOKKOS_FUNCTION void operator()(const typename Policy::member_type& team) const {
    RdmaScatterGatherWorker* sgw = m_view(0,0).sg;
    pack_response(m_view(0,0).ptr, sgw, sgw->fence_done_flag, team);
  }

 private:
  RemoteView m_view;
};

template <class Policy, class Lambda, class RemoteView>
void remote_parallel_for(const std::string& name, Policy&& policy, Lambda&& lambda, const RemoteView& view){
    if (policy.league_size() == 0){
        return;
    }
    using PolicyType = typename std::remove_reference<Policy>::type;
    using LambdaType = typename std::remove_reference<Lambda>::type;
    RemoteParallelFor<PolicyType,LambdaType,RemoteView> 
          rpf(std::forward<Lambda>(lambda), view);
    int num_teams = policy.league_size();
#ifdef KOKKOS_ENABLE_CUDA
    int vector_length = policy.vector_length();
#else
    int vector_length = 1;
#endif
    PolicyType new_policy(num_teams+2, policy.team_size(), vector_length);
    using remote_space = typename RemoteView::memory_space;
    using exec_space = typename RemoteView::execution_space;
    Kokkos::parallel_for(name, new_policy, rpf);
    exec_space().fence();

    RespondParallelFor<PolicyType,RemoteView> txpf(view);

    auto respond_policy = Kokkos::TeamPolicy<>(1,policy.team_size()*vector_length);
    Kokkos::parallel_for("respond", respond_policy, txpf);
    remote_space().fence();
    view.impl_map().clear_fence(exec_space{});
}


struct BootstrapPort {
  uint16_t lid;
  uint8_t port;
  uint32_t qp_num;
};

Transport* request_tport = nullptr;
Transport* response_tport = nullptr;
ibv_pd* global_pd = nullptr;
ibv_context* global_ctx = nullptr;
int global_pd_ref_count = 0;

void rdma_ibv_init()
{
  if (!request_tport){
    request_tport = new Transport(MPI_COMM_WORLD);
  }
  if (!response_tport){
    response_tport = new Transport(MPI_COMM_WORLD);
  }
}

void rdma_ibv_finalize()
{
  if (request_tport){
    delete request_tport;
    request_tport = nullptr;
  }

  if (response_tport){
    delete response_tport;
    response_tport = nullptr;
  }

}

Transport::Transport(MPI_Comm comm)
{
  MPI_Comm_size(comm, &nproc);
  MPI_Comm_rank(comm, &rank);

  int num_devices;
  ibv_device** devices = ibv_get_device_list(&num_devices);
  assert_ibv(devices);
  dev = devices[0];
  if (global_pd_ref_count == 0){
    global_ctx = ibv_open_device(dev);
    global_pd = ibv_alloc_pd(global_ctx);
  }
  ++global_pd_ref_count;
  pd = global_pd;
  ctx = global_ctx;
  assert_ibv(pd);

  std::vector<BootstrapPort> global_ports;
  std::vector<BootstrapPort> local_ports;

  constexpr int num_cqe = 2048;

  cq = ibv_create_cq(ctx, num_cqe, nullptr, nullptr, 0);
  assert_ibv(cq);
  qps = new ibv_qp*[nproc];

  struct ibv_srq_init_attr srq_init_attr;

  memset(&srq_init_attr, 0, sizeof(srq_init_attr));

  srq_init_attr.attr.max_wr  = 1024;
  srq_init_attr.attr.max_sge = 2;
  srq = ibv_create_srq(pd, &srq_init_attr);
  assert_ibv(srq);

  ibv_device_attr dev_attr;
  ibv_query_device(ctx, &dev_attr);

  ibv_port_attr port_attr;
  ibv_query_port(ctx, 1, &port_attr);
  if (port_attr.state != IBV_PORT_ACTIVE || port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND){
    Kokkos::abort("queried ibv port is not active");
  }

  ibv_qp_init_attr qp_attr;
  memset(&qp_attr, 0, sizeof(struct ibv_qp_init_attr));
  qp_attr.srq = srq;
  qp_attr.send_cq = cq;
  qp_attr.recv_cq = cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = 1024;
  qp_attr.cap.max_recv_wr = 1024;
  qp_attr.cap.max_send_sge = 2;
  qp_attr.cap.max_recv_sge = 2;
  qp_attr.cap.max_inline_data = 0;
  for (int pe=0; pe < nproc; ++pe){
    qps[pe] = ibv_create_qp(pd, &qp_attr);
    assert_ibv(qps[pe]);
    uint8_t max_ports = 1; //dev_attr.phys_port_cnt
    for (uint8_t p=1; p <= max_ports; ++p){
       local_ports.push_back({port_attr.lid, p, qps[pe]->qp_num});

       ibv_qp_attr attr;
       memset(&attr, 0, sizeof(struct ibv_qp_attr));
       attr.qp_state = IBV_QPS_INIT;
       attr.pkey_index = 0;
       attr.port_num = p;
       attr.qp_access_flags =
           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
       int qp_flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

       ibv_safe(ibv_modify_qp(qps[pe], &attr, qp_flags));
    }
  }

  // exchange the data out-of-band using MPI for queue pairs
  global_ports.resize(local_ports.size());
  MPI_Alltoall(local_ports.data(), sizeof(BootstrapPort), MPI_BYTE,
               global_ports.data(), sizeof(BootstrapPort), MPI_BYTE,
               comm);

  //now put all the queue pairs into a ready state
  for (int pe=0; pe < nproc; ++pe){
    //if (pe == rank) continue;

    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.dest_qp_num = global_ports[pe].qp_num;
    attr.rq_psn = 0;
    attr.path_mtu = IBV_MTU_1024; //port_attr.active_mtu;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.dlid = global_ports[pe].lid;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = global_ports[pe].port;
    int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MIN_RNR_TIMER | IBV_QP_PATH_MTU | IBV_QP_MAX_DEST_RD_ATOMIC;
    ibv_safe(ibv_modify_qp(qps[pe], &attr, mask));

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 0x12;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    ibv_safe(ibv_modify_qp(qps[pe], &attr, mask));
  }
}

Transport::~Transport()
{
  for (int pe=0; pe < nproc; ++pe){
    if (pe != rank){
     ibv_destroy_qp(qps[pe]);
    }
  }

  ibv_destroy_cq(cq);
  --global_pd_ref_count;
  if (global_pd_ref_count == 0){
    ibv_dealloc_pd(global_pd);
    ibv_close_device(global_ctx);
    global_pd = nullptr;
    global_ctx = nullptr;
  }

}

} //RACERlib

#define heisenbug printf("%s:%d\n", __FILE__, __LINE__); fflush(stdout)

#endif // RACERLIB_RDMA_ENGINE_H