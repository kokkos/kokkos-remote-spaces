## Kokkos Remote Spaces

Kokkos Remote Spaces (KRS) adds distributed memory support to the Kokkos parallel programming model. It supports MPI single-sided, SHMEM and NVSHMEM back-ends. Two examples are included, a RandomAccess micro benchmark as well as CGSolve. KRS implements global arrays. This unburdens the programmer from implementing explicit communication between processes or GPUs.

CGSolve is an example of a representative code in scientific computing that implements the conjugate gradient method in Kokkos. Taking a closer look shows that CGSolve involves a sequence of kernels that compute the product of two vectors (dot product), perform a matrix-vector multiplication (gemm), and compute a vector sum (axpy). While data can be partitioned for the dot product and the vector sum computations, the matrix-vector multiplication accesses vector elements across all partitions (memory spaces). Because of this, it can be challenging to maintain resource utilization when scaling to multiple processes or GPUs. RandomAccess implements a random-access to memory. Both applications showcase the use of the proposed API and its implementations.

*Note: Kokkos Remote Spaces is in an experimental development stage.*

## Dependencies

KRS has the following dependencies:

- MPI with single-sided communication or SHMEM enabled
- NVSHMEM for multiple Nvidia-based GPU support
- Kokkos

## Build

- Example: Building for MPI single-sided: export PATH=${KOKKOS_BUILD_DIR}/build/bin:$PATH cmake . -DKokkos_DIR=${KOKKOS_BUILD_DIR} -DKokkos_ENABLE_MPISPACE=ON -DCMAKE_CXX_COMPILER=nvcc_wrapper

- Example: Building for SHMEM: export PATH=${KOKKOS_BUILD_DIR}/build/bin:$PATH cmake . -DKokkos_DIR=${KOKKOS_BUILD_DIR} -DSHMEM_ROOT=${PATH_TO_MPI} -DKokkos_ENABLE_SHMEMSPACE=ON -DCMAKE_CXX_COMPILER=nvcc_wrapper

- Example: Building for NVSHMEM: export PATH=${KOKKOS_BUILD_DIR}/build/bin:$PATH cmake . -DKokkos_DIR=${KOKKOS_BUILD_DIR} -DKokkos_ENABLE_NVSHMEMSPACE=ON -DNVSHMEM_ROOT=${PATH_TO_NVSHMEM} -DCMAKE_CXX_COMPILER=nvcc_wrapper

## API

```C++
ViewType allocate_symmetric_remote_view(const char* const label, Args ... args)
```

## Example

```C++
using namespace Kokkos::Experimental;
using RemoteSpace = DefaultRemoteMemorySpace;
using RemoteView = Kokkos::View<int **, RemoteSpace>;
...
RemoteView v = allocate_symmetric_remote_view<RemoteView>(
        "RemoteView", size_per_rank);
v(pe,index)+=1;
printf("%i\n", v(pe,index));
```

Hint: Launching multiple processes per node requires the use of the '--kokkos-num-devices' Kokkos runtime flag. Please consult the Kokkos documentation for further information.

