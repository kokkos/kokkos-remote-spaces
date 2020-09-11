## Kokkos Remote Spaces

Kokkos Remote Spaces (KRS) adds distributed memory support to the Kokkos parallel programming model. It supports MPI single-sided, SHMEM and NVSHMEM back-ends. Two examples are included, a RandomAccess micro benchmark as well as CGSolve. KRS implements global arrays. This unburdens the programmer from implementing explicit communication between processes or GPUs.

CGSolve is an example of a representative code in scientific computing that implements the conjugate gradient method in Kokkos. Taking a closer look shows that CGSolve involves a sequence of kernels that compute the product of two vectors (dot product), perform a matrix-vector multiplication (gemm), and compute a vector sum (axpy). While data can be partitioned for the dot product and the vector sum computations, the matrix-vector multiplication accesses vector elements across all partitions (memory spaces). Because of this, it can be challenging to maintain resource utilization when scaling to multiple processes or GPUs. RandomAccess implements a random-access to memory. Both applications showcase the use of the proposed API and its implementations.

*Note: Kokkos Remote Spaces is in an experimental development stage.*

## Dependencies

KRS has MPI and Kokkos as required dependencies.
At least one of the following optional dependencies must also be enabled:

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

### MPI one-sided
Given a Kokkos installation at `KOKKOS_INSTALL_PREFIX` and a valid C++ compiler, an example configuration would be:
````bash
> cmake ${KRS_SOURCE_DIR} \
  -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
  -DKokkos_ENABLE_MPISPACE=ON \
  -DCMAKE_CXX_COMPILER=${KOKKOS_CXX}
````

### SHMEM
Given a Kokkos installation at `KOKKOS_INSTALL_PREFIX`, a SHMEM installation at `SHMEM_INSTALL_PREFIX`, and a valid C++ compiler, an example configuration would be:
````bash
> cmake ${KRS_SOURCE_DIR} \
  -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
  -DSHMEM_ROOT=${SHMEM_INSTALL_PREFIX} \
  -DKokkos_ENABLE_SHMEMSPACE=ON \
  -DCMAKE_CXX_COMPILER=${KOKKOS_CXX}
````

### NVSHMEM
Given a Kokkos installation at `KOKKOS_INSTALL_PREFIX`, an NVSHMEM installation at `NVSHMEM_INSTALL_PREFIX`, and a valid C++ compiler, an example configuration would be:
````bash
> cmake ${KRS_SOURCE_DIR} \
  -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
  -DNVSHMEM_ROOT=${SHMEM_INSTALL_PREFIX} \
  -DKokkos_ENABLE_NVSHMEMSPACE=ON \
  -DCMAKE_CXX_COMPILER=${KOKKOS_CXX}
````

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
        "RemoteView", num_ranks, size_per_rank);
v(pe,index)+=1;
printf("%i\n", v(pe,index));
```

Hint: Launching multiple processes per node requires the use of the '--kokkos-num-devices' Kokkos runtime flag. Please consult the Kokkos documentation for further information.

