<img src="https://github.com/kokkos/kokkos-remote-spaces/assets/755191/401eedaf-df35-4a59-bcd8-24c793e0a3f4" width="200" align="right"></img>
# Kokkos Remote Spaces
*Kokkos Remote Spaces* adds distributed shared memory (DSM) support to [*Kokkos*](https://github.com/kokkos/kokkos/). This enables a global view on data for a convenient multi-GPU, multi-node, and multi-device programming.

A new memory space type, namely the `DefaultRemoteMemorySpace` type, represents a Kokkos memory space with remote access semantic. Kokkos View specialized to this memory space through template arguments expose this semantic to the programmer. The underlying implementation of remote memory accesses relies on PGAS as a backend layer.

Currently, four PGAS backends are supported namely SHMEM, NVSHMEM, ROCSHMEM and MPI One-sided (preview). SHMEM and MPI One-sided are host-only programming models and thus support distributed memory accesses on hosts only. This corresponds to Kokkos' execution spaces such as Serial or OpenMP. NVSHMEM or ROCSHMEM support device-initiated communication on compatible devices and consequently require the Kokkos CUDA or HIP execution space. The following diagram shows how Kokkos Remote Spaces fits into the landscape of Kokkos and PGAS libraries.

<img src="https://user-images.githubusercontent.com/755191/120260364-ff398a80-c252-11eb-9f01-886bb888533a.png" width="500" align="right" ></img>

## New APIs and Kokkos API overloads

```
Kokkos::Experimental::DefaultRemoteMemorySpace;
Kokkos::Experimental::DefaultRemoteMemorySpace::fence();
Kokkos::Experimental::PartitionedLayout;
Kokkos::Experimental::get_local_range(View_t);
Kokkos::Experimental::get_range(View_t, rank);
Kokkos::View(...);}
Kokkos::subview(...);}
Kokkos::deep_copy(...);
Kokkos::local_deep_copy(...);
Kokkos::MemoryTraits::Atomic;
```

## Examples

The following example illustrates the type definition of a Kokkos remote view (`ViewRemote_3D_t`). Further, it shows the instantiation of a remote view (`view`), in this case of a 3-dimensional array of 20 elements per dimension, and a subsequent instantiation of a subview that can span over multiple virtual address spaces (`sub_view`). It is worth pointing out that GlobalLayouts, per definition, distribute arrays by the left-most dimension. View data can be accesses similarly to Kokkos views.

<img src="https://user-images.githubusercontent.com/755191/132418921-c5b7210d-52a1-42ce-ae9f-c04f31923a55.png" align="right" >

We have included more examples in the source code distribution namely RandomAccess as well as CGSolve. CGSolve is an example of a representative code in scientific computing that implements the conjugate gradient method in Kokkos. Taking a closer look shows that CGSolve involves a sequence of kernels that compute the product of two vectors (dot product), perform a matrix-vector multiplication (gemm), and compute a vector sum (axpy). While data can be partitioned for the dot product and the vector sum computations, the matrix-vector multiplication accesses vector elements across all partitions (memory spaces). Because of this, it can be challenging to maintain resource utilization when scaling to multiple processes or GPUs. RandomAccess implements random access to memory.

## Insights

The following charts provide a perspective on performance and development complexity. They show measured performance expressed as bandwidth (GB/sec) and quantify development complexity with Kokkos Remote Spaces compared to other programming models by comparing LOC (lines of code). We compare different executions on the Lassen supercomputer and against a reference implementation with MPI+Cuda. Here, the executions correspond to configurations with one, two, and four Nvidia V100 GPUs. It can be observed that executions on >2 GPUs result in fine-grained memory accesses over the inter-processor communication interface which is significantly slower (~4x) than the NVLink interface connecting 2 GPUs each (NVLink complex). Reducing the access granularity by presorting remote non-regular accesses to the distributed data structure and moving data in bulk using local deep copies would be the appropriate optimization strategy in this case.

<img width="1301" alt="image" src="https://user-images.githubusercontent.com/755191/120720459-5a54c280-c489-11eb-96d0-842a26272019.png">

In this chart, GI (global indexing) and LI (local indexing) mark two implementations of CGSolve where one implementation uses Kokkos views with global layout (`GlobalLayoutLeft` and where the other uses `LayoutLeft` and thus relies on the programmer to compute the PE index. In the latter case, this computation can be implemented with a binary left-shift which yields a slight performance advantage.

## Building Kokkos Remote Spaces

Kokkos Remote Spaces is built using [CMake](https://cmake.org) version 3.17 or later. Is a stand-alone project with dependencies on *Kokkos* and a selected *PGAS backend library*. The following steps document the build process. Note that building in the root directory in not allowed.

#### CMake paths

| Path         | Description                                             |
| ------------ | ------------------------------------------------------- |
| Kokkos_ROOT  | Path to the root of the Kokkos install                  |
| SHMEM_ROOT   | Path to the root of a SHMEM installation if enabled     |
| NVSHMEM_ROOT | Path to the root of a NVSHMEM installation if enabled   |


#### Supported CMake Options

| Variable                | Default | Description                        |
| ----------------------- | ------- | ---------------------------------- |
| KRS_ENABLE_SHMEMSPACE| OFF     | Enables the SHMEM backend             |
| KRS_ENABLE_NVSHMEMSPACE| OFF     | Enables the NVSHMEM backend           |
| KRS_ENABLE_MPISPACE  | OFF     | Enables the MPI backend               |
| KRS_ENABLE_EXAMPLES  | OFF     | Enables building examples             |
| KRS_ENABLE_TESTS     | OFF     | Enables building tests                |


#### Examples

Building with `SHMEM`
```bash
   $: cmake . -DKRS_ENABLE_SHMEMSPACE=ON
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR}
           -DSHMEM_ROOT=${PATH_TO_SHMEM}
           -DCMAKE_CXX_COMPILER=oshc++
   $: make
```

Building with `NVSHMEM`
```bash
   $: cmake . -DKRS_ENABLE_NVSHMEMSPACE=ON
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR}
           -DNVSHMEM_ROOT=${PATH_TO_NVSHMEM}
           -DCMAKE_CXX_COMPILER=nvcc_wrapper
   $: make
```

Building with `MPI`
```bash
   $: cmake . -DKRS_ENABLE_MPISPACE=ON
           -DKokkos_ROOT=${KOKKOS_INSTALL_DIR}
           -DCMAKE_CXX_COMPILER=mpicxx
   $: make
```

## Building an Application with Kokkos Remote Spaces

Applications depend at least on Kokkos Remote Spaces and may depend on Kokkos Kernels or others. The following sample shows a cmake build file to generate the build scripts for "MyRemoteApp". It depends on Kokkos Remote Spaces and Kokkos Kernels.


```cmake
#Example
cmake_minimum_required(VERSION 3.13)

project(MyRemoteApp LANGUAGES CXX)

find_package(KokkosKernels REQUIRED)
find_package(KokkosRemote REQUIRED)

add_executable(MatVec matvec.cpp)
target_link_libraries(MatVec PRIVATE \
         Kokkos::kokkoskernels Kokkos::kokkosremotespaces)
```

This cmake build fike can be used as

```cmake
cmake .. -DKokkosKernels_ROOT=$KokkosKernels_INSTALL_PATH -DKokkosRemote_ROOT=$KokkosRemoteSpaces_INSTALL_PATH
```

*Note: Kokkos Remote Spaces is in an experimental development stage.*

### Contact
Jan Ciesko, jciesko@sandia.gov
