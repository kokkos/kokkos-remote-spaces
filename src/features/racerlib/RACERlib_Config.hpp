#define Kokkos_CUDA_ENABLED

#ifdef Kokkos_CUDA_ENABLED
#ifdef __CUDA_ARCH__
#define KOKKOS_REMOTE_THREADFENCE() __threadfence()
#define KOKKOS_REMOTE_THREADFENCE_SYSTEM() __threadfence_system()
#define KOKKOS_REMOTE_SHARED __shared__
#define KOKKOS_REMOTE_WORKER_THREAD_ID() threadIdx.x *blockDim.y + threadIdx.y
#define KOKKOS_REMOTE_NUM_WORKER_THREADS() blockDim.x *blockDim.y
#define KOKKOS_REMOTE_SYNCTHREADS()                                            \
  do {                                                                         \
  } while (0)
#else
#define KOKKOS_REMOTE_THREADFENCE()                                            \
  do {                                                                         \
  } while (0)
#define KOKKOS_REMOTE_THREADFENCE_SYSTEM()                                     \
  do {                                                                         \
  } while (0)
#define KOKKOS_REMOTE_SHARED thread_local
#define KOKKOS_REMOTE_WORKER_THREAD_ID() 0
#define KOKKOS_REMOTE_NUM_WORKER_THREADS() 1
#define KOKKOS_REMOTE_SYNCTHREADS()
#endif
#else
#define KOKKOS_REMOTE_THREADFENCE()                                            \
  do {                                                                         \
  } while (0)
#define KOKKOS_REMOTE_THREADFENCE_SYSTEM()                                     \
  do {                                                                         \
  } while (0)
#define KOKKOS_REMOTE_SHARED thread_local
#define KOKKOS_REMOTE_WORKER_THREAD_ID() 0
#define KOKKOS_REMOTE_NUM_WORKER_THREADS() 1
#define KOKKOS_REMOTE_SYNCTHREADS()
#endif
