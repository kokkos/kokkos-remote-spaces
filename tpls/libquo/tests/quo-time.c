/**
 * Copyright (c) 2013-2016 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 * National Security, LLC for the U.S.  Department of Energy. The U.S.
 * Government has rights to use, reproduce, and distribute this software.
 * NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
 * WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
 * SOFTWARE.  If software is modified to produce derivative works, such modified
 * software should be clearly marked, so as not to confuse it with the version
 * available from LANL.
 *
 * Additionally, redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 *
 * · Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * · Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * · Neither the name of Los Alamos National Security, LLC, Los Alamos
 *   National Laboratory, LANL, the U.S. Government, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 * SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "quo.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "mpi.h"

/**
 * Measures cost of QUO calls.
 */

typedef struct context_t {
    /* my rank */
    int rank;
    /* number of ranks in MPI_COMM_WORLD */
    int nranks;
    /* number of nodes in our job */
    int nnodes;
    /* number of ranks that share this node with me (includes myself) */
    int nnoderanks;
    /* my node rank */
    int noderank;
    /* whether or not mpi is initialized */
    bool mpi_inited;
    /* number of sockets on the node */
    int nsockets;
    /* number of cores on the node */
    int ncores;
    /* number of pus (processing units - e.g hw threads) */
    int npus;
    /* quo major version */
    int qv;
    /* quo minor version */
    int qsv;
    /* pointer to initial stringification of our cpuset */
    char *cbindstr;
    /* flag indicating whether or not we are initially bound */
    int bound;
    /* our quo context (the thing that gets passed around all over the place).
     * filler words that make this comment line look mo better... */
    QUO_context quo;
} context_t;

typedef struct experiment_t {
    context_t *c;
    char *name;
    int (*fun)(context_t *, int , double *);
    int n_trials;
    int res_len;
    double *results;
} experiment_t;

/**
 * rudimentary "pretty print" routine. not needed in real life...
 */
static inline void
demo_emit_sync(const context_t *c)
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep((c->rank) * 1000);
}

static int
fini(context_t *c)
{
    if (!c) return 1;
    if (QUO_SUCCESS != QUO_free(c->quo)) return 1;
    /* finalize mpi AFTER QUO_destruct - we may mpi in our destruct */
    if (c->mpi_inited) MPI_Finalize();
    if (c->cbindstr) free(c->cbindstr);
    free(c);
    return 0;
}

/**
 * i'm being really sloppy here. ideally, one should probably save the rc and
 * then display or do some other cool thing with it. don't be like this code. if
 * QUO_construct or QUO_init fail, then someone using this could just continue
 * without the awesomeness that is libquo. they cleanup after themselves, so
 * things *should* be in an okay state after an early failure. the failures may
 * be part of a larger problem, however. */
static int
init(context_t **c)
{
    context_t *newc = NULL;
    /* alloc our context */
    if (NULL == (newc = calloc(1, sizeof(*newc)))) return 1;
    /* libquo requires that MPI be initialized before its init is called */
    if (MPI_SUCCESS != MPI_Init(NULL, NULL)) return 1;
    /* gather some basic job info from our mpi lib */
    if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &(newc->nranks))) goto err;
    /* ...and more */
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(newc->rank))) goto err;
    /* can be called at any point -- even before init and construct. */
    if (QUO_SUCCESS != QUO_version(&(newc->qv), &(newc->qsv))) goto err;
    /* relatively expensive call. you only really want to do this once at the
     * beginning of time and pass the context all over the place within your
     * code. */
    if (QUO_SUCCESS != QUO_create(&newc->quo, MPI_COMM_WORLD)) goto err;
    newc->mpi_inited = true;
    *c = newc;
    return 0;
err:
    (void)fini(newc);
    return 1;
}

/**
 * gather system and job info from libquo.
 */
static int
sys_grok(context_t *c)
{
    char *bad_func = NULL;

    /* this interface is more powerful, but the other n* calls can be more
     * convenient. at any rate, this is an example of the
     * QUO_nobjs_in_type_by_type interface to get the number of sockets on
     * the machine. note: you can also use the QUO_nsockets or
     * QUO_nobjs_by_type to get the same info. */
    if (QUO_SUCCESS != QUO_nobjs_in_type_by_type(c->quo,
                                                 QUO_OBJ_MACHINE,
                                                 0,
                                                 QUO_OBJ_SOCKET,
                                                 &c->nsockets)) {
        bad_func = "QUO_nobjs_in_type_by_type";
        goto out;
    }
    if (QUO_SUCCESS != QUO_ncores(c->quo, &c->ncores)) {
        bad_func = "QUO_ncores";
        goto out;
    }
    if (QUO_SUCCESS != QUO_npus(c->quo, &c->npus)) {
        bad_func = "QUO_npus";
        goto out;
    }
    if (QUO_SUCCESS != QUO_bound(c->quo, &c->bound)) {
        bad_func = "QUO_bound";
        goto out;
    }
    if (QUO_SUCCESS != QUO_stringify_cbind(c->quo, &c->cbindstr)) {
        bad_func = "QUO_stringify_cbind";
        goto out;
    }
    if (QUO_SUCCESS != QUO_nnodes(c->quo, &c->nnodes)) {
        bad_func = "QUO_nnodes";
        goto out;
    }
    if (QUO_SUCCESS != QUO_nqids(c->quo, &c->nnoderanks)) {
        bad_func = "QUO_nqids";
        goto out;
    }
    if (QUO_SUCCESS != QUO_id(c->quo, &c->noderank)) {
        bad_func = "QUO_id";
        goto out;
    }
out:
    if (bad_func) {
        fprintf(stderr, "%s: %s failure :-(\n", __func__, bad_func);
        return 1;
    }
    return 0;
}

static int
emit_node_basics(const context_t *c)
{
    (void)c;
#if 0
    /* one proc per node will emit this info */
    if (0 == c->noderank) {
        printf("### %d: quo version: %d.%d ###\n", c->rank, c->qv, c->qsv);
        printf("### %d: nnodes: %d\n", c->rank, c->nnodes);
        printf("### %d: nnoderanks: %d\n", c->rank, c->nnoderanks);
        printf("### %d: nsockets: %d\n", c->rank, c->nsockets);
        printf("### %d: ncores: %d\n", c->rank, c->ncores);
        printf("### %d: npus: %d\n", c->rank, c->npus);
        fflush(stdout);
    }
    demo_emit_sync(c);
#endif
    return 0;
}

static int
qcreate(
    context_t *c,
    int n_trials,
    double *res
) {
    (void)c;
    //
    QUO_context *ctx = calloc(n_trials, sizeof(*ctx));
    if (!ctx) return 1;
    //
    for (int i = 0; i < n_trials; ++i) {
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_create(&(ctx[i]), MPI_COMM_WORLD)) return 1;
        double end = MPI_Wtime();
        res[i] = end - start;
    }
    for (int i = 0; i < n_trials; ++i) {
        QUO_free(ctx[i]);
    }
    return 0;
}

static int
qfree(
    context_t *c,
    int n_trials,
    double *res
) {
    (void)c;
    //
    QUO_context *ctx = calloc(n_trials, sizeof(*ctx));
    if (!ctx) return 1;
    //
    for (int i = 0; i < n_trials; ++i) {
        if (QUO_SUCCESS != QUO_create(&(ctx[i]), MPI_COMM_WORLD)) return 1;
    }
    for (int i = 0; i < n_trials; ++i) {
        double start = MPI_Wtime();
        QUO_free(ctx[i]);
        double end = MPI_Wtime();
        res[i] = end - start;
    }
    return 0;
}

static int
qnpus(
    context_t *c,
    int n_trials,
    double *res
) {
    //
    QUO_context ctx;
    if (QUO_SUCCESS != QUO_create(&ctx, MPI_COMM_WORLD)) return 1;
    //
    int n = 0;
    for (int i = 0; i < n_trials; ++i) {
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_npus(ctx, &n)) return 1;
        double end = MPI_Wtime();
        res[i] = end - start;
    }
    // Don't want compiler to optimize this away. Will never print.
    if (c->rank == (c->nranks + 1)) printf("### NPUS: %d\n", n);
    if (QUO_SUCCESS != QUO_free(ctx)) return 1;
    return 0;
}

static int
qquids_in_type(
    context_t *c,
    int n_trials,
    double *res
) {
    //
    QUO_context ctx;
    if (QUO_SUCCESS != QUO_create(&ctx, MPI_COMM_WORLD)) return 1;
    //
    int n = 0;
    for (int i = 0; i < n_trials; ++i) {
        int nq = 0;
        int *qids = NULL;
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_qids_in_type(
                               ctx, QUO_OBJ_PU, 0,
                               &nq, &qids)) return 1;
        double fs = MPI_Wtime();
        free(qids);
        double fe = MPI_Wtime();
        double end = MPI_Wtime();
        res[i] = (end - start) - (fe - fs);
    }
    // Don't want compiler to optimize this away. Will never print.
    if (c->rank == (c->nranks + 1)) printf("%d\n", n);
    if (QUO_SUCCESS != QUO_free(ctx)) return 1;
    return 0;
}

static int
qbind_push(
    context_t *c,
    int n_trials,
    double *res
) {
    (void)c;
    //
    QUO_context ctx;
    if (QUO_SUCCESS != QUO_create(&ctx, MPI_COMM_WORLD)) return 1;
    //
    for (int i = 0; i < n_trials; ++i) {
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_bind_push(
                               ctx, QUO_BIND_PUSH_OBJ,
                               QUO_OBJ_MACHINE, -1)) return 1;
        double end = MPI_Wtime();
        res[i] = end - start;
        if (QUO_SUCCESS != QUO_bind_pop(ctx)) return 1;
    }
    if (QUO_SUCCESS != QUO_free(ctx)) return 1;
    return 0;
}

static int
qbind_pop(
    context_t *c,
    int n_trials,
    double *res
) {
    (void)c;
    //
    QUO_context ctx;
    if (QUO_SUCCESS != QUO_create(&ctx, MPI_COMM_WORLD)) return 1;
    //
    for (int i = 0; i < n_trials; ++i) {
        if (QUO_SUCCESS != QUO_bind_push(
                               ctx, QUO_BIND_PUSH_OBJ,
                               QUO_OBJ_MACHINE, -1)) return 1;
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_bind_pop(ctx)) return 1;
        double end = MPI_Wtime();
        res[i] = end - start;
    }
    if (QUO_SUCCESS != QUO_free(ctx)) return 1;
    return 0;
}

static int
qauto_distrib(
    context_t *c,
    int n_trials,
    double *res
) {
    //
    QUO_context ctx;
    if (QUO_SUCCESS != QUO_create(&ctx, MPI_COMM_WORLD)) return 1;
    //
    int sel = 0;
    for (int i = 0; i < n_trials; ++i) {
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_auto_distrib(ctx, QUO_OBJ_PU,
                                            c->nranks, &sel)) return 1;
        double end = MPI_Wtime();
        res[i] = end - start;
    }
    // Don't want compiler to optimize this away. Will never print.
    if (c->rank == (c->nranks + 1)) printf("### NPUS: %d\n", sel);
    if (QUO_SUCCESS != QUO_free(ctx)) return 1;
    return 0;
}

static int
qbarrier(
    context_t *c,
    int n_trials,
    double *res
) {
    (void)c;
    //
    QUO_context ctx;
    if (QUO_SUCCESS != QUO_create(&ctx, MPI_COMM_WORLD)) return 1;
    //
    for (int i = 0; i < n_trials; ++i) {
        double start = MPI_Wtime();
        if (QUO_SUCCESS != QUO_barrier(ctx)) return 1;
        double end = MPI_Wtime();
        res[i] = end - start;
    }
    if (QUO_SUCCESS != QUO_free(ctx)) return 1;
    return 0;
}

/**
 *
 */
static int
time_fun(
    context_t *c,
    int (*fun)(context_t *, int , double *),
    int n_trials,
    int *out_res_len,
    double **out_results
) {
    double *res = NULL;
    int res_len = 0;
    if (c->rank != 0) {
        *out_res_len = n_trials;
        res_len = *out_res_len;
        res = calloc(res_len, sizeof(*res));
    }
    else {
        *out_res_len = n_trials * c->nranks;
        res_len = *out_res_len;
        res = calloc(res_len, sizeof(*res));
    }
    if (!res) return 1;
    //
    if (fun(c, n_trials, res)) return 1;
    //
    if (MPI_SUCCESS != MPI_Gather(0 == c->rank ? MPI_IN_PLACE : res,
                                  n_trials, MPI_DOUBLE, res, n_trials,
                                  MPI_DOUBLE, 0, MPI_COMM_WORLD)) {
        return 1;
    }
#if 0 // DEBUG
    if (0 == c->rank) {
        for (int i = 0; i < res_len; ++i) {
            printf("%lf, ", res[i]);
        }
        printf("\n");
    }
#endif
    *out_results = res;
    return 0;
}

static int
emit_stats(
    context_t *c,
    char *name,
    int res_len,
    double *results)
{
    if (c->rank != 0) goto out;
    //
    double tot = 0.0;
    for (int i = 0; i < res_len; ++i) {
        tot += results[i];
    }
    double ave = tot / (double)res_len;
    // Calculate standard deviation
    double a = 0.0;
    for (int i = 0; i < res_len; ++i) {
        a += powf(results[i] - ave, 2.0);
    }
    double stddev = sqrtl(a / (res_len - 1));

    double sem = stddev / sqrtl((double)res_len);

    printf("NUMPE : %d\n"      , c->nranks);
    printf("Test Name : %s\n"      , name);
    printf("Number of Entries : %d\n"      , res_len);
    printf("Average Time (us) : %.10lf\n"  , ave    * 1e6);
    printf("Standard Deviation (us) : %.10lf\n"  , stddev * 1e6);
    printf("Standard Error of Mean (us) : %.10lf\n\n", sem    * 1e6);

out:
    demo_emit_sync(c);
    return 0;
}

static int
run_experiment(experiment_t *e) {
    char *bad_func = NULL;
    if (time_fun(e->c, e->fun, e->n_trials, &(e->res_len), &(e->results))) {
        bad_func = e->name;
        goto out;
    }
    if (emit_stats(e->c, e->name, e->res_len, e->results)) {
        bad_func = "emit_stats";
        goto out;
    }
    free(e->results);
out:
    if (bad_func) {
        fprintf(stderr, "%s failed!\n", bad_func);
        return 1;
    }
    return 0;
}


int
main(void)
{
    int erc = EXIT_SUCCESS;
    char *bad_func = NULL;
    context_t *context = NULL;

    /* ////////////////////////////////////////////////////////////////////// */
    /* init code */
    /* ////////////////////////////////////////////////////////////////////// */
    if (init(&context)) {
        bad_func = "init";
        goto out;
    }
    /* ////////////////////////////////////////////////////////////////////// */
    /* libquo is now ready for service */
    /* ////////////////////////////////////////////////////////////////////// */

    /* first gather some info so we can base our decisions on our current
     * situation. */
    if (sys_grok(context)) {
        bad_func = "sys_grok";
        goto out;
    }
    if (emit_node_basics(context)) {
        bad_func = "emit_node_basics";
        goto out;
    }
    if (0 == context->rank) {
        printf("### Starting QUO Timing Tests...\n");
        fflush(stdout);
    }
    demo_emit_sync(context);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    static const int n_trials = 100;
    //
    experiment_t experiments[] =
    {
        {context, "QUO_create",       qcreate,        n_trials, 0, NULL},
        {context, "QUO_free",         qfree,          n_trials, 0, NULL},
        {context, "QUO_npus",         qnpus,          n_trials, 0, NULL},
        {context, "QUO_qids_in_type", qquids_in_type, n_trials, 0, NULL},
        {context, "QUO_bind_push",    qbind_push,     n_trials, 0, NULL},
        {context, "QUO_bind_pop",     qbind_pop,      n_trials, 0, NULL},
        {context, "QUO_auto_distrib", qauto_distrib,  n_trials, 0, NULL},
        {context, "QUO_barrier",      qbarrier,       n_trials, 0, NULL}
    };

    for (unsigned i = 0; i < sizeof(experiments)/sizeof(experiment_t); ++i) {
        run_experiment(&experiments[i]);
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "XXX %s failure in: %s\n", __FILE__, bad_func);
        erc = EXIT_FAILURE;
    }
    (void)fini(context);
    return erc;
}
