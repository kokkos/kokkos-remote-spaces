## Kokkos Remote Spaces
Kokkos Remote Spaces (KRS) adds distributed memory support to the Kokkos parallel programming model. It does so by adding new memory space types to Kokkos. Memory spaces can be used to specialize Kokkos views. Specialized views (remote views) can be used to access local as well as remote memory, that is, memory in a different address space through the View type operator. KRS implements remote memory support via a set of back-ends that implement a PGAS access semantic. 

### Background
Traditionally, to support the execution on devices with disjoint memory address spaces, the developer is required to implements data exchange manually. This includes buffer allocation, data assembly (packing and unpacking), communication library invocation, and correct synchronization. PGAS programming models can simplify data management as they offer a globally accessible view of user data. While now a single-address space program can execute in parallel, spanning across many GPUs or cluster nodes, remote memory accesses are costly and special care is needed to achieve satisfactory performance. One strategy to reduce the exposure of memory access latencies, originating from per-element accesses, is the use of the block transfer API (deep_copy and local_deep_copy using remote views) in Kokkos. 

### Example 
KRS includes a set of examples including a conjugate gradient solver (CGSolve).  CGSolve invokes a sequence of kernels that compute the product of two vectors (dot product), matrix-vector multiplication (gemm), and vector sum (axpy). To support multiple GPUs and cluster nodes, the program input data is divided into partitions where each partition corresponds to one such device. While the dot product and the vector sum computations access local data only due to their regular memory access patterns, the parallel matrix-vector multiplication accesses vector elements across all partitions (memory spaces). This makes the matrix-vector multiplication a great example to showcase the use of the KRS API

*Note: Kokkos Remote Spaces is in an experimental development stage.*

### Dependencies
KRS requires MPI and Kokkos. At least one of the following dependencies must be enabled:
- MPI with one-sided support
- SHMEM (usually from OpenSHMEM or OpenMPI)
- NVSHMEM (required for communication in CUDA kernels)

## Build
For building KRS, you should build with the same compiler used to build Kokkos.
For CUDA, this is usually `nvcc_wrapper`, e.g.
````bash
KOKKOS_CXX=${KOKKOS_INSTALL_PREFIX}/bin/nvcc_wrapper
````
Code should be built out-of-source in a separate build directory.

#### MPI one-sided
Given a Kokkos installation at `KOKKOS_INSTALL_PREFIX` and a valid C++ compiler, an example configuration would be:
````bash
> cmake ${KRS_SOURCE_DIR} \
  -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
  -DKokkos_ENABLE_MPISPACE=ON \
  -DCMAKE_CXX_COMPILER=${KOKKOS_CXX}
````

#### SHMEM
Given a Kokkos installation at `KOKKOS_INSTALL_PREFIX`, a SHMEM installation at `SHMEM_INSTALL_PREFIX`, and a valid C++ compiler, an example configuration would be:
````bash
> cmake ${KRS_SOURCE_DIR} \
  -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
  -DSHMEM_ROOT=${SHMEM_INSTALL_PREFIX} \
  -DKokkos_ENABLE_SHMEMSPACE=ON \
  -DCMAKE_CXX_COMPILER=${KOKKOS_CXX}
````

#### NVSHMEM
Given a Kokkos installation at `KOKKOS_INSTALL_PREFIX`, an NVSHMEM installation at `NVSHMEM_INSTALL_PREFIX`, and a valid C++ compiler, an example configuration would be:
````bash
> cmake ${KRS_SOURCE_DIR} \
  -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
  -DNVSHMEM_ROOT=${SHMEM_INSTALL_PREFIX} \
  -DKokkos_ENABLE_NVSHMEMSPACE=ON \
  -DCMAKE_CXX_COMPILER=${KOKKOS_CXX}
````

### API
```C++
template <class DataType [, class LayoutType] [, class RemoteMemorySpace] [, class MemoryTraits]>
class View;
```

## Example
```C++
using namespace Kokkos::Experimental;
using RemoteSpace = DefaultRemoteMemorySpace;
using RemoteView = Kokkos::View<int **, RemoteSpace>;
...
RemoteView v ("RemoteView", num_ranks, size_per_rank);
v(pe,index)+=1;
printf("%i\n", v(pe,index));
```

Hint: Launching multiple processes per node requires the use of the '--kokkos-num-devices' Kokkos runtime flag. Please consult the Kokkos documentation for further information.

