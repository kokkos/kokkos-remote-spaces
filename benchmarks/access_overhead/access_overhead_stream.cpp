//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_RemoteSpaces.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>

#define STREAM_ARRAY_SIZE 134217728
#define STREAM_NTIMES 20

using RemoteSpace_t     = Kokkos::Experimental::DefaultRemoteMemorySpace;
using StreamDeviceArray = Kokkos::View<double*, RemoteSpace_t>;
using StreamHostArray   = Kokkos::View<double*, Kokkos::HostSpace>;
using StreamIndex       = size_t;
using Policy            = Kokkos::RangePolicy<Kokkos::IndexType<StreamIndex>>;

void perform_set(StreamDeviceArray& a, const double scalar) {
  Kokkos::parallel_for(
      "set", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a(i) = scalar; });
  RemoteSpace_t().fence();
}

void perform_incr(StreamDeviceArray& a, const double scalar) {
  Kokkos::parallel_for(
      "set", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a(i) += scalar; });
  RemoteSpace_t().fence();
}

void perform_copy(StreamDeviceArray& a, StreamDeviceArray& b) {
  Kokkos::parallel_for(
      "copy", Policy(0, a.extent(0)), KOKKOS_LAMBDA(const StreamIndex i) {
        double tmp = a(i);
        b(i)       = tmp;
      });
  RemoteSpace_t().fence();
}

void perform_scale(StreamDeviceArray& b, StreamDeviceArray& c,
                   const double scalar) {
  Kokkos::parallel_for(
      "scale", Policy(0, b.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b(i) = scalar * c(i); });
  RemoteSpace_t().fence();
}

void perform_add(StreamDeviceArray& a, StreamDeviceArray& b,
                 StreamDeviceArray& c) {
  Kokkos::parallel_for(
      "add", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { c(i) = a(i) + b(i); });
  RemoteSpace_t().fence();
}

void perform_triad(StreamDeviceArray& a, StreamDeviceArray& b,
                   StreamDeviceArray& c, const double scalar) {
  Kokkos::parallel_for(
      "triad", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a(i) = b(i) + scalar * c(i); });
  RemoteSpace_t().fence();
}

int perform_validation(StreamHostArray& a, StreamHostArray& b,
                       StreamHostArray& c, const StreamIndex arraySize,
                       const double scalar) {
  double ai = 1.0;
  double bi = 2.0;
  double ci = 0.0;

  for (StreamIndex i = 0; i < arraySize; ++i) {
    ci = ai;
    bi = scalar * ci;
    ci = ai + bi;
    ai = bi + scalar * ci;
  };

  double aError = 0.0;
  double bError = 0.0;
  double cError = 0.0;

  for (StreamIndex i = 0; i < arraySize; ++i) {
    aError = std::abs(a[i] - ai);
    bError = std::abs(b[i] - bi);
    cError = std::abs(c[i] - ci);
  }

  double aAvgError = aError / (double)arraySize;
  double bAvgError = bError / (double)arraySize;
  double cAvgError = cError / (double)arraySize;

  const double epsilon = 1.0e-13;
  int errorCount       = 0;

  if (std::abs(aAvgError / ai) > epsilon) {
    fprintf(stderr, "Error: validation check on View a failed.\n");
    errorCount++;
  }

  if (std::abs(bAvgError / bi) > epsilon) {
    fprintf(stderr, "Error: validation check on View b failed.\n");
    errorCount++;
  }

  if (std::abs(cAvgError / ci) > epsilon) {
    fprintf(stderr, "Error: validation check on View c failed.\n");
    errorCount++;
  }

  if (errorCount == 0) {
    printf("All solutions checked and verified.\n");
  }

  return errorCount;
}

int run_benchmark(uint64_t size, uint64_t reps, uint64_t ls, uint64_t ts,
                  uint64_t vs) {
  StreamDeviceArray dev_a("a", size);
  StreamDeviceArray dev_b("b", size);
  StreamDeviceArray dev_c("c", size);

  StreamHostArray host_a("a", size);
  StreamHostArray host_b("b", size);
  StreamHostArray host_c("c", size);

  const double scalar = 3.0;

  double setTime   = std::numeric_limits<double>::max();
  double incrTime  = std::numeric_limits<double>::max();
  double copyTime  = std::numeric_limits<double>::max();
  double scaleTime = std::numeric_limits<double>::max();
  double addTime   = std::numeric_limits<double>::max();
  double triadTime = std::numeric_limits<double>::max();

  Kokkos::parallel_for(
      "init", Kokkos::RangePolicy<>(0, size), KOKKOS_LAMBDA(const int i) {
        dev_a(i) = 1.0;
        dev_b(i) = 2.0;
        dev_c(i) = 0.0;
      });

  Kokkos::Timer timer;

  for (StreamIndex k = 0; k < reps; ++k) {
    timer.reset();
    perform_set(dev_c, 1.5);
    setTime = std::min(setTime, timer.seconds());

    timer.reset();
    perform_incr(dev_c, 1);
    incrTime = std::min(incrTime, timer.seconds());

    timer.reset();
    perform_copy(dev_a, dev_c);
    copyTime = std::min(copyTime, timer.seconds());

    timer.reset();
    perform_scale(dev_b, dev_c, scalar);
    scaleTime = std::min(scaleTime, timer.seconds());

    timer.reset();
    perform_add(dev_a, dev_b, dev_c);
    addTime = std::min(addTime, timer.seconds());

    timer.reset();
    perform_triad(dev_a, dev_b, dev_c, scalar);
    triadTime = std::min(triadTime, timer.seconds());
  }

  printf("%lu,%lu,%lu,%lu,%li,%li,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", ls, ts, vs,
         static_cast<uint64_t>(size), (size * sizeof(double)) >> 20,
         ((size * sizeof(double)) >> 20) * 3,
         (1.0e-06 * 1.0 * (double)sizeof(double) * (double)size) / setTime,
         (1.0e-06 * 1.0 * (double)sizeof(double) * (double)size) / incrTime,
         (1.0e-06 * 2.0 * (double)sizeof(double) * (double)size) / copyTime,
         (1.0e-06 * 2.0 * (double)sizeof(double) * (double)size) / scaleTime,
         (1.0e-06 * 3.0 * (double)sizeof(double) * (double)size) / addTime,
         (1.0e-06 * 3.0 * (double)sizeof(double) * (double)size) / triadTime);

  Kokkos::deep_copy(host_a, dev_a);
  Kokkos::deep_copy(host_b, dev_b);
  Kokkos::deep_copy(host_c, dev_c);

  return perform_validation(host_a, host_b, host_c, size, scalar);
}

int main(int argc, char* argv[]) {
  int mpi_thread_level_available;
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;

#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
  mpi_thread_level_required = MPI_THREAD_SINGLE;
#endif

  MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);

#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

#ifdef KRS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  uint64_t array_size  = STREAM_ARRAY_SIZE;
  uint64_t repetitions = STREAM_NTIMES;
  uint64_t ls = 32, vs = 32, ts = 32;

  array_size  = argc > 1 ? atoi(argv[1]) : array_size;
  repetitions = argc > 2 ? atoi(argv[2]) : repetitions;
  ls          = argc > 3 ? atoi(argv[3]) : ls;
  ts          = argc > 4 ? atoi(argv[4]) : ts;
  vs          = argc > 5 ? atoi(argv[5]) : vs;

  Kokkos::initialize(argc, argv);
  const int rc = run_benchmark(array_size, repetitions, ls, ts, vs);
  Kokkos::finalize();
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#endif
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
  MPI_Finalize();
  return rc;
}