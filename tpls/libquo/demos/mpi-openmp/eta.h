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

#pragma once

#include <inttypes.h>
#include "mpi.h"

enum {
    ETA_SUCCESS = 0,
    ETA_FAILURE,
    ETA_INVALID_ARG,
    ETA_OOR
};

int
eta(double s,
    int64_t n,
    double **z,
    MPI_Comm comm,
    double *result);
