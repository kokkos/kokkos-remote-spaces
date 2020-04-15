/*
 * Copyright (c) 2017      Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the libquo project. See the LICENSE file at the
 * top-level directory of this distribution.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

#include "mpi.h"
#include "omp.h"
#include "quo.h"

#define pprintf(p, va...)                                                      \
do {                                                                           \
    if ((p)) {                                                                 \
        printf(va);                                                            \
        fflush(stdout);                                                        \
    }                                                                          \
} while(0)

#define DDOT_OMP_CHUNK_SIZE 128

enum {
    SUCCESS = 0,
    FAILURE
};

static double
gettime(void) {
    return MPI_Wtime();
}

static double
timediff(double start,
         double end)
{
    return end - start;
}

typedef struct matrix_t {
    /* Local number of rows. */
    int64_t m;
    /* Local number of columns. */
    int64_t n;
    /* Points to dense data. */
    double **values;
} matrix_t;

typedef struct vector_t {
    /* Local length. */
    int64_t length;
    /* Values. */
    double *values;
} vector_t;

static int
vector_construct(vector_t *v,
                 int64_t len)
{
    if (!v) return FAILURE;

    v->length = len;
    v->values = calloc(len, sizeof(double));
    if (!v->values) return FAILURE;

    return SUCCESS;
}

static int
vector_destruct(vector_t *v)
{
    if (!v) return FAILURE;

    if (v->values) free(v->values);

    return SUCCESS;
}

static int
matrix_construct(matrix_t *mat,
                 int64_t m_local,
                 int64_t n_local)

{
    if (!mat) return FAILURE;

    mat->m = m_local;
    mat->n = n_local;

    mat->values = calloc(m_local, sizeof(double));
    if (!mat->values) return FAILURE;

    for (int64_t r = 0; r < m_local; ++r) {
        mat->values[r] = calloc(n_local, sizeof(double));
        if (!mat->values[r]) return FAILURE;
    }
    return SUCCESS;
}

static int
matrix_destruct(matrix_t *mat)
{
    if (!mat) return FAILURE;
    if (mat->values) {
        for (int64_t r = 0; r < mat->m; ++r) {
            if (mat->values[r]) free(mat->values[r]);
        }
    }
    return SUCCESS;
}

typedef struct dgemv_t {
    /* Number of MPI processes in MPI_COMM_WORLD. */
    int numpe;
    /* My rank in MPI_COMM_WORLD. */
    int pe;
    /* A QUO contex. */
    QUO_context qc;
    /* Local number of rows. */
    int64_t m;
    /* Local number of columns. */
    int64_t n;
    /* My local matrix. */
    matrix_t matrix;
    /* My local input vector. */
    vector_t vector_in;
    /* My local result vector. */
    vector_t vector_out;
} dgemv_t;

static int
init_mpi(dgemv_t *d,
         int argc,
         char **argv)
{
    if (MPI_SUCCESS != MPI_Init(&argc, &argv)) goto err;

    if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &(d->numpe))) goto err;
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(d->pe)))    goto err;

    return SUCCESS;
err:
    return FAILURE;
}

static int
fini_mpi(void)
{
    if (MPI_SUCCESS != MPI_Finalize()) return FAILURE;

    return SUCCESS;
}

static int
create_quo_context(dgemv_t *d)
{
    const bool emit = (0 == d->pe);

    pprintf(emit, "# Creating QUO Context...\n");

    const double start = gettime();

    if (QUO_SUCCESS != QUO_create(&(d->qc), MPI_COMM_WORLD)) return FAILURE;

    const double end = gettime();

    pprintf(emit, "# quo-context-create-time-s=%lf\n", timediff(start, end));

    return SUCCESS;
}

static int
free_quo_context(dgemv_t *d)
{
    const bool emit = (0 == d->pe);

    pprintf(emit, "# Freeing QUO Context...\n");

    const double start = gettime();

    if (QUO_SUCCESS != QUO_free(d->qc)) return FAILURE;

    const double end = gettime();

    pprintf(emit, "# quo-context-free-time-s=%lf\n", timediff(start, end));

    return SUCCESS;
}

static bool
starts_with(char *s,
            char *prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(s, prefix, n)) return false;
    return true;
}

static int
init_dgem(dgemv_t *d,
          int argc,
          char **argv)
{
    static const int n_params = 2;
    static const int64_t defaultv = 4096;
    int64_t iparams[] = {defaultv, defaultv};
    /* Local sizes. */
    static char *params[16] = {
        "--m=",
        "--n="
    };

    /* Parse argv. */
    for (int i = 1; i < argc; ++i) {
        for (int pi = 0; pi < n_params; ++pi) {
            if (starts_with(argv[i], params[pi])) {
                if (sscanf(argv[i] + strlen(params[pi]),
                           "%" PRId64, iparams + pi) != 1 || iparams[pi] < 1) {
                    iparams[pi] = defaultv;
                }
            }
        }
    }

    d->m = iparams[0];
    d->n = iparams[1];

    return SUCCESS;
}

static int
emit_config(dgemv_t *d)
{
    const bool emit = (0 == d->pe);

    pprintf(emit,
            "# MPI:\n"
            "# mpi-numpe=%d\n",
            d->numpe);

    pprintf(emit, "#\n");

    pprintf(emit,
            "# OpenMP:\n"
            "# omp-max-threads=%d\n",
            omp_get_max_threads());

    pprintf(emit, "#\n");

    pprintf(emit,
            "# Local Matrix:\n"
            "# local-m=%" PRId64 "\n"
            "# local-n=%" PRId64 "\n"
            "# Global Matrix:\n"
            "# global-m=%" PRId64 "\n"
            "# global-n=%" PRId64 "\n",
            d->m, d->n, d->m * d->numpe, d->n);

    pprintf(emit, "#\n");

    return SUCCESS;
}

/*
 * Basic Setup:
 *
 * Matrices:
 * - Globally (m x numpe) rows
 * - Globally n columns
 * - Decomposition: 1D along the rows. Each MPI process will have m elements.
 */
static int
gen_dgemv(dgemv_t *d)
{
    int rc = SUCCESS;
    const bool emit = (0 == d->pe);

    pprintf(emit, "# Generating Problem...\n");

    const double start = gettime();

    if (SUCCESS != (rc = matrix_construct(&d->matrix, d->m, d->n))) goto out;
    if (SUCCESS != (rc = vector_construct(&d->vector_in,  d->n)))   goto out;
    if (SUCCESS != (rc = vector_construct(&d->vector_out, d->n)))   goto out;

    const double end = gettime();

    pprintf(emit, "# prob-gen-time-s=%lf\n", timediff(start, end));
out:
    return rc;
}

static int
teardown_dgemv(dgemv_t *d)
{
    int rc = SUCCESS;
    if (SUCCESS != (rc = matrix_destruct(&d->matrix))) goto out;
    if (SUCCESS != (rc = vector_destruct(&d->vector_in))) goto out;
    if (SUCCESS != (rc = vector_destruct(&d->vector_out))) goto out; 
out:
    return rc;
}

static double
comp_ddot(const double *restrict x,
          const double *restrict y,
          int64_t n)
{
    double res = 0.0;
    int64_t i = 0;

#pragma omp parallel for                                                       \
        default(shared) private(i)                                             \
        schedule(static, DDOT_OMP_CHUNK_SIZE)                                  \
        reduction(+:res)
    for (i = 0; i < n; ++i) {
        res += x[i] * y[i];
    }

    return res;
}

static int
comp_dgemv(dgemv_t *d)
{
    const bool emit = (0 == d->pe);

    pprintf(emit, "# Calculating y=Ax...\n");

    const double start = gettime();

    double        *y  = d->vector_out.values;
    const double  **A = (const double **)d->matrix.values;
    const double  *x  = d->vector_in.values;

    const int64_t nrow = d->matrix.m;
    const int64_t ncol = d->matrix.n;

    for (int64_t r = 0; r < nrow; ++r) {
        y[r] = comp_ddot(A[r], x, ncol);
    }

    const double end = gettime();

    pprintf(emit, "# comp-dgemv-s=%lf\n", timediff(start, end));

    return SUCCESS;
}

int
main(int argc, char **argv)
{
    int rc = SUCCESS;
    dgemv_t dgem;
    /* Init MPI library. */
    if (SUCCESS != (rc = init_mpi(&dgem, argc, argv))) goto out;
    /* Problem init. */
    if (SUCCESS != (rc = init_dgem(&dgem, argc, argv))) goto out;
    /* Display basic setup info. */
    if (SUCCESS != (rc = emit_config(&dgem))) goto out;
    /* Generate the problem. */
    if (SUCCESS != (rc = gen_dgemv( &dgem))) goto out;
    /* Create a QUO context (can be done anytime after MPI_Init). */
    if (SUCCESS != (rc = create_quo_context(&dgem))) goto out;
    /* Perform the parallel calculation. */
    if (SUCCESS != (rc = comp_dgemv(&dgem))) goto out;
    /* Cleanup. */
    if (SUCCESS != (rc = teardown_dgemv(&dgem))) goto out;
    if (SUCCESS != (rc = free_quo_context(&dgem))) goto out;
    if (SUCCESS != (rc = fini_mpi())) goto out;
out:
    return (SUCCESS == rc) ? EXIT_SUCCESS : EXIT_FAILURE;
}
