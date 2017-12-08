/**
 * Copyright (c) 2013-2016 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 * National Security, LLC for the U.S. Department of Energy. The U.S. Government
 * has rights to use, reproduce, and distribute this software.  NEITHER THE
 * GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
 * OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If
 * software is modified to produce derivative works, such modified software
 * should be clearly marked, so as not to confuse it with the version available
 * from LANL.
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

/**
 * a simple demo code that quiesces all the MPI processes on the node where
 * MPI_COMM_WORLD rank 0 resides with a QUO_barrier and expands the cpuset of
 * rank 0 to accommodate OpenMP threading.
 */

#include "quo.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpi.h"

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
emit_bind_state(const context_t *c)
{
    char *cbindstr = NULL, *bad_func = NULL;
    int bound = 0;

    demo_emit_sync(c);
    if (QUO_SUCCESS != QUO_stringify_cbind(c->quo, &cbindstr)) {
        bad_func = "QUO_stringify_cbind";
        goto out;
    }
    if (QUO_SUCCESS != QUO_bound(c->quo, &bound)) {
        bad_func = "QUO_bound";
        goto out;
    }
    printf("### process %d rank %d [%s] bound: %s\n",
           (int)getpid(), c->rank, cbindstr, bound ? "true" : "false");
    fflush(stdout);
out:
    demo_emit_sync(c);
    if (cbindstr) free(cbindstr);
    if (bad_func) {
        fprintf(stderr, "%s: %s failure :-(\n", __func__, bad_func);
        return 1;
    }
    return 0;
}

static int
emit_node_basics(const context_t *c)
{
    /* one proc per node will emit this info */
    if (0 == c->noderank) {
        printf("### quo version: %d.%d ###\n", c->qv, c->qsv);
        printf("### nnodes: %d\n", c->nnodes);
        printf("### nnoderanks: %d\n", c->nnoderanks);
        printf("### nsockets: %d\n", c->nsockets);
        printf("### ncores: %d\n", c->ncores);
        printf("### npus: %d\n", c->npus);
        fflush(stdout);
    }
    demo_emit_sync(c);
    return 0;
}

/**
 * expands the caller's cpuset to all available resources on the node.
 */
static int
bindup_node(const context_t *c)
{
    /* if you are going to change bindings often, then cache this */
    if (QUO_SUCCESS != QUO_bind_push(c->quo, QUO_BIND_PUSH_OBJ,
                                     QUO_OBJ_MACHINE, -1)) {
        return 1;
    }
    return 0;
}

/**
 * revert pushed binding policy to the previous policy.
 */
static int
binddown_node(const context_t *c)
{
    if (QUO_SUCCESS != QUO_bind_pop(c->quo)) {
        return 1;
    }
    return 0;
}

int
do_omp_things(const context_t *c)
{
    char *cbindstr = NULL;

    printf("rank %d about to do OpenMP things!\n", c->rank);
    if (QUO_SUCCESS != QUO_stringify_cbind(c->quo, &cbindstr)) {
        return 1;
    }
    printf("rank %d's cpuset: %s\n", c->rank, cbindstr);
    free(cbindstr);
    printf("rank %d is now threading up life in OMP land...\n", c->rank);
    sleep(2); /* do real work here... */
    printf("rank %d is now done threading up life in OMP land...\n", c->rank);
    return 0;
}

int
enter_omp_region(const context_t *c)
{
    /* FIXME - assumes that ranks are assigned by filling in a node at a
     * time. Easy to create a new, more general version, but this is a demo
     * code, so why are you doing weird mapping things anyway!?!
     */
    bool on_rank_0s_node = c->rank < c->nnoderanks;
    if (on_rank_0s_node) {
        if (0 == c->rank) {
            fprintf(stdout, "getting ready for OMP region...\n");
            /* change policy before the OMP calculation */
            if (bindup_node(c)) return 1;
            /* do the calculation */
            if (do_omp_things(c)) return 1;
            /* revert to old binding policy */
            if (binddown_node(c)) return 1;
        }
        /* everyone else wait for rank 0 completion. QUO_barrier because it's
         * cheaper than MPI_Barrier on a node. */
        if (QUO_SUCCESS != QUO_barrier(c->quo)) {
            return 1;
        }
        if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD)) return 1;
    }
    /* not on rank 0's node, so wait in MPI barrier */
    else {
        if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD)) return 1;
    }
    return 0;
}

int
main(void)
{
    int erc = EXIT_SUCCESS;
    char *bad_func = NULL;
    context_t *context = NULL;

    setbuf(stdout, NULL);

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
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
    /* change binding policies to accommodate OMP threads on node 0 */
    if (enter_omp_region(context)) {
        bad_func = "enter_omp_region";
        goto out;
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "XXX %s failure in: %s\n", __FILE__, bad_func);
        erc = EXIT_FAILURE;
    }
    (void)fini(context);
    return erc;
}
