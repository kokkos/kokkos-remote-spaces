## Kokkos Remote Spaces

<img src="https://user-images.githubusercontent.com/755191/120260364-ff398a80-c252-11eb-9f01-886bb888533a.png" width="500" align="right" >

Kokkos Remote Spaces adds distributed shared memory (DSM) support to the Kokkos parallel programming model. This enables a global view on data for a convenient multi-GPU, multi-node, and multi-device programming.

Under the hood, a new memory space type, namely the `DefaultRemoteMemorySpace` type, represent a Kokkos memory space with remote access semantic. Kokkos View specialized to this memory space through template arguments expose this semantic to the programmer. The underlying implementation of remote memory accesses relies on PGAS as a backend layer. 

Currently, three PGAS backends are supported namely SHMEM, NVSHMEM, and MPI One-sided (preview). SHMEM and MPI One-sided are host-only programming models and thus support distributed memory accesses on hosts only. This corresponds to Kokko's execution spaces such as Serial or OpenMP. NVSHMEM supports device-initiated communication on CUDA devices and consequently the Kokkos CUDA execution space. The following diagramm shows the fit of Kokkos Remote Spaces into the landscape of Kokkos and PGAS libraries.

## Examples

The following example illustrates the type definition of a Kokkos remote view. Further it shows the instatiation of a remote view, in this case of a 3-dimensional array of 20 elements per dimension and a subsequent instatiation of a subview that can span over a multiple of virtual address spaces. It is worth to point out that GlobalLayouts, per definition, distribute arrays by the left-most dimension. View data can be accesses simmilarly to Kokkos views.

<img src="https://user-images.githubusercontent.com/755191/120261884-10d06180-c256-11eb-9f07-9649a5331864.png" align="right" >

We have included more examples in the source code distribution namely RandomAccess as well as CGSolve. CGSolve is an example of a representative code in scientific computing that implements the conjugate gradient method in Kokkos. Taking a closer look shows that CGSolve involves a sequence of kernels that compute the product of two vectors (dot product), perform a matrix-vector multiplication (gemm), and compute a vector sum (axpy). While data can be partitioned for the dot product and the vector sum computations, the matrix-vector multiplication accesses vector elements across all partitions (memory spaces). Because of this, it can be challenging to maintain resource utilization when scaling to multiple processes or GPUs. RandomAccess implements a random-access to memory. 

## Build

Kokkos Remote Spaces is a stand-alone project with dependencies on Kokkos and a selected PGAS backend library. The following steps document the build process from within the Kokkos Remote Spaces root directory.

`SHMEM`
```
   $: export PATH=${KOKKOS_BUILD_DIR}/bin:$PATH
   $: cmake . -DKokkos_ENABLE_SHMEMSPACE=ON
           -DKokkos_DIR=${KOKKOS_BUILD_DIR} 
           -DSHMEM_ROOT=${PATH_TO_MPI}
           -DCMAKE_CXX_COMPILER=mpicxx
   $: make
```

`NVSHMEM`
```
   $: export PATH=${KOKKOS_BUILD_DIR}/bin:$PATH
   $: cmake . -DKokkos_ENABLE_NVSHMEMSPACE=ON
           -DKokkos_DIR=${KOKKOS_BUILD_DIR} 
           -DNVSHMEM_ROOT=${PATH_TO_NVSHMEM}
           -DCMAKE_CXX_COMPILER=nvcc_wrapper
   $: make
```

`MPI`
```
   $: export PATH=${KOKKOS_BUILD_DIR}/bin:$PATH
   $: cmake . -DKokkos_ENABLE_MPISPACE=ON
           -DKokkos_DIR=${KOKKOS_BUILD_DIR} 
           -DCMAKE_CXX_COMPILER=mpicxx
   $: make
```

*Note: Kokkos Remote Spaces is in an experimental development stage.*

