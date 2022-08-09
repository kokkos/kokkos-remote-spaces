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
#include <Kokkos_Core.hpp>
#include <cassert>

using ORDINAL_T       = int64_t;
using CONST_ORDINAL_T = const ORDINAL_T;
using VALUE_T         = double;

#define DEFAULT_DIM_SIZE 4096
#define LEAGUE_SIZE 32
#define TEAM_SIZE 256
#define VEC_LEN 1

using VectorHost_t = Kokkos::View<VALUE_T *, Kokkos::HostSpace>;
using MatrixHost_t = Kokkos::View<VALUE_T **, Kokkos::HostSpace>;
using Vector_t     = Kokkos::View<VALUE_T *, Kokkos::CudaSpace>;
using Matrix_t     = Kokkos::View<VALUE_T **, Kokkos::CudaSpace>;

int main(int argc, char *argv[]) {
  // Vars
  float time = 0;
  ORDINAL_T nx;

  int league_size = LEAGUE_SIZE;
  int team_size   = TEAM_SIZE;
  int vec_len     = VEC_LEN;

  nx = argc > 1 ? atoi(argv[1]) : DEFAULT_DIM_SIZE;

  Kokkos::initialize(argc, argv);
  using TeamPolicy  = Kokkos::TeamPolicy<>;
  TeamPolicy policy = TeamPolicy(league_size, team_size, vec_len);
  {
    MatrixHost_t A_h("A_h", nx, nx);
    VectorHost_t b_h("b_h", nx);
    VectorHost_t x_h("x_h", nx);

    Kokkos::deep_copy(A_h, 2.0);
    Kokkos::deep_copy(b_h, 0.0);
    Kokkos::deep_copy(x_h, 1.0);

    auto A = Kokkos::create_mirror_view_and_copy(Kokkos::CudaSpace(), A_h);
    auto b = Kokkos::create_mirror_view_and_copy(Kokkos::CudaSpace(), b_h);
    auto x = Kokkos::create_mirror_view_and_copy(Kokkos::CudaSpace(), x_h);

    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "mv", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nx),
                               [&](CONST_ORDINAL_T row) {
                                 VALUE_T b_row = 0.0;
                                 Kokkos::parallel_reduce(
                                     Kokkos::ThreadVectorRange(team, nx),
                                     [=](CONST_ORDINAL_T col, VALUE_T &sum) {
                                       sum += A(row, col) * x(col);
                                     },
                                     b_row);
                                 b(row) = b_row;
                               });
        });

    Kokkos::fence();
    time = timer.seconds();

    Kokkos::deep_copy(b_h, b);
    for (ORDINAL_T i = 0; i < nx; ++i) assert(b_h(i) == 2 * nx);
    printf("%.2f sec, %.2f MB/sec\n", time,
           ((nx * nx + 2 * nx) * sizeof(VALUE_T) >> 10) / time);
  }
  Kokkos::finalize();
  return 0;
}
