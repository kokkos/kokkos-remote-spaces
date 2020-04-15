/*
 * Copyright (c) 2017      Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the libquo project. See the LICENSE file at the
 * top-level directory of this distribution.
 */

/*
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <math.h>

#include "mpi.h"

#include "eta.h"
#include "zeta.h"

static int
verify(double s,
       double ns,
       double zs)
{
    double expected = ns;

    double eta_zeta_relation = (1.0 - powl(2.0, 1.0 - s)) * zs;

    printf("expected  : %.16lf\ncalculated: %.16lf\n",
           expected, eta_zeta_relation);

    return 0;
}

int
main(int argc, char **argv)
{
    if (MPI_SUCCESS != MPI_Init(&argc, &argv)) return EXIT_FAILURE;
    int cid = 0;
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &cid)) return EXIT_FAILURE;

    const bool emit = (0 == cid);
    const int64_t sigma_n = 1e6;
    const double s = 2.0;

    double *z = NULL, *n = NULL;
    double zs = 0.0, ns = 0.0;

    int zrc = zeta(s, sigma_n, &z, MPI_COMM_WORLD, &zs);
    if (ZETA_SUCCESS != zrc) goto out;
    if (emit) printf("z(%lf) = %.16lf\n", s, zs);

    int erc = eta(s, sigma_n, &n, MPI_COMM_WORLD, &ns);
    if (ETA_SUCCESS != erc) goto out;
    if (emit) printf("n(%lf) = %.16lf\n", s, ns);

    if (emit) verify(s, ns, zs);

out:
    if (MPI_SUCCESS != MPI_Finalize()) zrc = ZETA_FAILURE;
    if (z) free(z);
    if (n) free(n);
    return (ZETA_SUCCESS == zrc ? EXIT_SUCCESS : EXIT_FAILURE);
}
