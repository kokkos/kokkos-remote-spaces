/*
 * Copyright (c) 2017      Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the libquo project. See the LICENSE file at the
 * top-level directory of this distribution.
 */

/*
 * Parallel library that calculates n(s) for any s > 0.0, where n is the
 * Dirichlet eta function.
 */

#include "eta.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <math.h>

#include "mpi.h"

static int
etai(double s,
     int64_t low,
     int64_t high,
     double *restrict n)
{
    for (int64_t ni = 0, i = low; i < high; ++i, ++ni) {
        n[ni] = powl(-1.0, i - 1) / powl(i, s);
    }

    return ETA_SUCCESS;
}

static double
sigma_eta(int64_t n_len,
          double *restrict n)
{
    double result = 0.0;
    for (int64_t i = 0; i < n_len; ++i) {
        result += n[i];
    }
    return result;
}

int
eta(double s,
    int64_t n_sigma,
    double **n,
    MPI_Comm comm,
    double *result)
{
    if (s < 0.0) return ETA_INVALID_ARG;

    int cid = 0, csize = 0;
    if (MPI_SUCCESS != MPI_Comm_rank(comm, &cid))   return ETA_FAILURE;
    if (MPI_SUCCESS != MPI_Comm_size(comm, &csize)) return ETA_FAILURE;

    /* My bounds. */
    int64_t low  = 1 + (cid * n_sigma);
    int64_t high = (cid + 1) * n_sigma;
    /* My array of zeta(i). */
    double *ntmp = calloc(n_sigma, sizeof(ntmp));
    if (!ntmp) return ETA_OOR;
    /* Fill in local z(i)s. */
    int rc = etai(s, low, high, ntmp);
    if (ETA_SUCCESS != rc) return rc;
    /* Calculate local sum */
    double partial_sum = sigma_eta(n_sigma, ntmp);
    /* Calculate global sum */
    if (MPI_SUCCESS != MPI_Reduce(&partial_sum, result,
                                  1, MPI_DOUBLE, MPI_SUM,
                                  0, comm)) {
        rc = ETA_FAILURE;
        goto out;
    }
out:
    if (ETA_SUCCESS != rc) free(ntmp);
    else *n = ntmp;
    return rc;
}
