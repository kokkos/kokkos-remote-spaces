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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Random.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <getopt.h>
#include <limits>
#include <mpi.h>
#include <stdlib.h>

// Default values
#define SIZE 1024
#define NUM_ITER 1000
#define LEAGUE_SIZE 1
#define TEAM_SIZE 32
#define VEC_LEN 1
#define ORDINAL_T int64_t
#define SIGMA 1000

using RemoteSpace = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView = Kokkos::View<ORDINAL_T **, RemoteSpace>;
using Generator = Kokkos::Random_XorShift64_Pool<>;

KOKKOS_INLINE_FUNCTION
ORDINAL_T get(const ORDINAL_T mean, const float variance,
              Generator::generator_type &g) {

  /*
  Uncomment to select between random or linear access pattern.
  Random access pattern follow either a
  normal or linear distribution (randomly or linearly accessed).
  */
  return g.normal(mean, variance);
  //return g.urand64();
  // return mean;
}

int main(int argc, char *argv[]) {

  // Init
  MPI_Init(&argc, &argv);

#ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_init();
#endif
#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  // Vars
  float time = 0;
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  ORDINAL_T array_size = (SIZE << 10) / sizeof(ORDINAL_T);
  ORDINAL_T num_iters = NUM_ITER / num_ranks;
  ORDINAL_T next_iters = num_iters;
  int sigma = SIGMA;
  int league_size = LEAGUE_SIZE;
  int team_size = TEAM_SIZE;
  int vec_len = VEC_LEN;

  array_size =
      argc > 1 ? (atoi(argv[1]) << 10) / sizeof(ORDINAL_T) : array_size;
  sigma = argc > 2 ? atoi(argv[2]) : sigma;
  league_size = argc > 3 ? atoi(argv[3]) : league_size;
  team_size = argc > 4 ? atoi(argv[4]) : team_size;
  vec_len = argc > 5 ? atoi(argv[5]) : vec_len;

  // Set up variance based on sigma
  /*
   Compute variance: Three Sigma rule
   https://en.wikipedia.org/wiki/68–95–99.7_rule
  */
  const float sigma_fixed = 0.20;
  float variance = (float)array_size * sigma_fixed;
  variance = variance / 1000.0 * sigma;

  // Initialize Kokkos RT
  Kokkos::initialize(argc, argv);
  using TeamPolicy = Kokkos::TeamPolicy<>;
  TeamPolicy policy = TeamPolicy(league_size, team_size, vec_len);
  {
    ORDINAL_T elems_per_rank;
    ORDINAL_T iters_per_team;
    elems_per_rank = ceil(1.0 * array_size / num_ranks);
    int rank_list[num_ranks];
    RemoteView v = Kokkos::Experimental::allocate_symmetric_remote_view<RemoteView>(
        "RemoteView", elems_per_rank);

    do {
      Generator gen_pool(5374857);
      // Update execution parameters
      num_iters = next_iters;
      iters_per_team = ceil(num_iters / league_size);
      num_iters = iters_per_team * league_size;

      Kokkos::Timer timer;

      Kokkos::parallel_for(
          "Outer", policy, KOKKOS_LAMBDA(const TeamPolicy::member_type &team) {
            Generator::generator_type g = gen_pool.get_state();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, my_rank * iters_per_team,
                                        (my_rank + 1) * iters_per_team),
                [&](const ORDINAL_T i) {
                  int rank;
                  ORDINAL_T index = abs(get(i, variance, g));
                  rank = (index / elems_per_rank) % num_ranks;
                  index = index % elems_per_rank;                  
                  v(rank, index) ^= 0xC0FFEE;
                });
            gen_pool.free_state(g);
          });

      RemoteSpace().fence();
      time = timer.seconds();

      // Increase iteration space to reach a 2 second execution time.
      if (next_iters * 4 > std::numeric_limits<ORDINAL_T>::max() / 4)
        break;
      next_iters *= 4;
    } while (time <= 2.0);
  }

  if (my_rank == 0) {
    float GBs =
        (2.0 * num_ranks * num_iters * sizeof(ORDINAL_T)) / 1024.0 / 1024.0 / 1024.0 / time;
    float access_latency = time / (num_ranks * num_iters) * 1.0e6;

    printf("%i, %i, %i, %i, %lld, %.3f, %.2f, %.4f\n", num_ranks, league_size,
           team_size, vec_len, array_size, access_latency, time, GBs);
  }

  Kokkos::finalize();
  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_finalize();
  #endif
  #ifdef KOKKOS_ENABLE_MPISPACE
  MPI_Finalize();
  #endif
  
}
