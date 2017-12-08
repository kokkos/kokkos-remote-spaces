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

#include "callee-driven-ex-p1.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>

#include "callee-driven-ex-p1.h"

static int
emit_bind_state(const p1_context_t *c,
                char *msg_prefix)
{
    char *cbindstr = NULL, *bad_func = NULL;
    int bound = 0;

    if (QUO_SUCCESS != QUO_stringify_cbind(c->quo, &cbindstr)) {
        bad_func = "QUO_stringify_cbind";
        goto out;
    }
    if (QUO_SUCCESS != QUO_bound(c->quo, &bound)) {
        bad_func = "QUO_bound";
        goto out;
    }
    printf("%s [rank %d] process %d [%s] bound: %s\n",
           msg_prefix, c->rank, (int)getpid(),
           cbindstr, bound ? "true" : "false");
    fflush(stdout);
out:
    if (cbindstr) free(cbindstr);
    if (bad_func) {
        fprintf(stderr, "%s: %s failure :-(\n", __func__, bad_func);
        return 1;
    }
    return 0;
}

/**
 * rudimentary "pretty print" routine. not needed in real life...
 */
static void
demo_emit_sync(const p1_context_t *c)
{
    MPI_Barrier(c->init_comm_dup);
    usleep((c->rank) * 1000);
}

/**
 * gather system and job info from libquo.
 */
static int
sys_grok(p1_context_t *c)
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
    /* for NUMA nodes */
    if (QUO_SUCCESS != QUO_nnumanodes(c->quo, &c->nnumanodes)) {
        bad_func = "QUO_nnumanodes";
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
emit_node_basics(const p1_context_t *c)
{
    demo_emit_sync(c);
    /* one proc per node will emit this info */
    if (0 == c->noderank) {
        printf("### [rank %d] quo version: %d.%d ###\n",
                c->rank, c->qv, c->qsv);
        printf("### [rank %d] nnodes: %d\n", c->rank, c->nnodes);
        printf("### [rank %d] nnoderanks: %d\n", c->rank, c->nnoderanks);
        printf("### [rank %d] nnumanodes: %d\n", c->rank, c->nnumanodes);
        printf("### [rank %d] nsockets: %d\n", c->rank, c->nsockets);
        printf("### [rank %d] ncores: %d\n", c->rank, c->ncores);
        printf("### [rank %d] npus: %d\n", c->rank, c->npus);
        fflush(stdout);
    }
    demo_emit_sync(c);
    return 0;
}

/**
 * this is where we set our policy regarding who will actually do work. the
 * others will sit in a quo barrier an wait for the workers to finish.
 *
 * this particular example distributes the workers among all the sockets on the
 * system, but you can imagine doing the same for NUMA nodes, for example. if
 * there are no NUMA nodes on the system, then fall back to something else.
 */
static int
get_worker_pes(p1_context_t *c,
               int *nworkers,
               int **workers)
{
    int res_assigned = 0, tot_workers = 0;
    int rc = QUO_ERR;
    /* array that hold whether or not a particular rank is going to do work */
    int *work_contribs = NULL;
    int *worker_ranks = NULL;

    /* let quo distribute workers over the sockets. if res_assigned is 1 after
     * this call, then i have been chosen. */
    if (QUO_SUCCESS != QUO_auto_distrib(c->quo, QUO_OBJ_SOCKET,
                                        2, &res_assigned)) {
        return 1;
    }
    /* array that hold whether or not a particular rank is going to do work */
    work_contribs = calloc(c->nranks, sizeof(*work_contribs));
    if (!work_contribs) {
        rc = QUO_ERR_OOR;
        goto out;
    }
    if (MPI_SUCCESS != (rc = MPI_Allgather(&res_assigned, 1, MPI_INT,
                                           work_contribs, 1, MPI_INT,
                                           c->init_comm_dup))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* now iterate over the array and count the total number of workers */
    for (int i = 0; i < c->nranks; ++i) {
        if (1 == work_contribs[i]) ++tot_workers;
    }
    worker_ranks = calloc(tot_workers, sizeof(*worker_ranks));
    if (!worker_ranks) {
        rc = QUO_ERR_OOR;
        goto out;
    }
    /* populate the array with the worker comm world ranks */
    for (int i = 0, j = 0; i < c->nranks; ++i) {
        if (1 == work_contribs[i]) {
            worker_ranks[j++] = i;
        }
    }
    *nworkers = tot_workers;
    *workers = worker_ranks;
out:
    if (work_contribs) free(work_contribs);
    if (QUO_SUCCESS != rc) {
        if (worker_ranks) free(worker_ranks);
    }
    return (QUO_SUCCESS == rc) ? 0 : 1;
}

static int
push_bind(const p1_context_t *c)
{
    if (QUO_SUCCESS != QUO_bind_push(c->quo, QUO_BIND_PUSH_OBJ,
                                     QUO_OBJ_SOCKET, -1)) {
        return 1;
    }
    return 0;
}

/* revert our binding policy so p0 can go about its business with its own
 * binding policy... */
static int
pop_bind(const p1_context_t *c)
{
    if (QUO_SUCCESS != QUO_bind_pop(c->quo)) return 1;
    return 0;
}

static int
quo_init(p1_context_t *ctx)
{
    /* can be called at any point -- even before construct. */
    if (QUO_SUCCESS != QUO_version(&(ctx->qv), &(ctx->qsv))) return 1;
    /* relatively expensive call. you only really want to do this once at the
     * beginning of time and pass the context all over the place within your
     * code.
     */
    if (QUO_SUCCESS != QUO_create(&ctx->quo, ctx->init_comm_dup)) return 1;
    //
    return 0;
}

int
gen_libcomm(p1_context_t *c,
            int np1s /* number of participants |p1who| */,
            int *p1who /* the participating ranks (initializing comm) */)
{
    int rc = QUO_SUCCESS;
    if (0 == c->noderank) {
        printf("### [rank %d] %d p1pes slated for work\n", c->rank, np1s);
        printf("### [rank %d] and they are: ", c->rank);
        if (0 == np1s) printf("\n");
        fflush(stdout);
        for (int i = 0; i < np1s; ++i) {
            printf("%d ", p1who[i]); fflush(stdout);
            if (i + 1 == np1s) printf("\n"); fflush(stdout);
        }
    }
    /* ////////////////////////////////////////////////////////////////////// */
    /* now create our own communicator based on the rank ids passed here */
    /* ////////////////////////////////////////////////////////////////////// */
    MPI_Group init_comm_grp;
    MPI_Group p1_group;
    if (MPI_SUCCESS != MPI_Comm_group(c->init_comm_dup, &init_comm_grp)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Group_incl(init_comm_grp, np1s,
                                      p1who, &p1_group)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Comm_create(c->init_comm_dup,
                                       p1_group,
                                       &(c->quo_comm))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* am i in the new communicator? */
    c->in_quo_comm = (MPI_COMM_NULL == c->quo_comm) ? false : true;
    if (c->in_quo_comm) {
        if (MPI_SUCCESS != MPI_Comm_size(c->quo_comm, &c->qc_size)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
        if (MPI_SUCCESS != MPI_Comm_rank(c->quo_comm, &c->qc_rank)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
    }
out:
    if (MPI_SUCCESS != MPI_Group_free(&init_comm_grp)) return 1;
    return (QUO_SUCCESS == rc) ? 0 : 1;
}

int
p1_init(p1_context_t **p1_ctx,
        MPI_Comm comm)
{
    if (!p1_ctx) return 1;
    //
    int inited = 0;
    if (MPI_SUCCESS != MPI_Initialized(&inited)) return 1;
    /* QUO requires that MPI be initialized before its use. */
    if (!inited) return 1;
    //
    p1_context_t *newc = NULL;
    if (NULL == (newc = calloc(1, sizeof(*newc)))) return 1;
    // dup initializing comm */
    if (MPI_SUCCESS != MPI_Comm_dup(comm, &newc->init_comm_dup)) return 1;
    /* gather some basic info about initializing communicator */
    if (MPI_SUCCESS != MPI_Comm_size(newc->init_comm_dup, &newc->nranks)) {
        return 1;
    }
    if (MPI_SUCCESS != MPI_Comm_rank(newc->init_comm_dup, &newc->rank)) {
        return 1;
    }
    //
    if (quo_init(newc)) return 1;
    //
    if (sys_grok(newc)) return 1;
    //
    if (emit_node_basics(newc)) return 1;
    //
    demo_emit_sync(newc);
    if (emit_bind_state(newc, "###")) return 1;
    demo_emit_sync(newc);
    //
    int n_workers = 0;
    int *worker_comm_ids = NULL;
    if (get_worker_pes(newc, &n_workers, &worker_comm_ids)) return 1;
    if (gen_libcomm(newc, n_workers, worker_comm_ids)) return 1;
    //
    *p1_ctx = newc;
    return 0;
}

int
p1_fini(p1_context_t *c)
{
    if (!c) return 0;
    MPI_Comm_free(&c->init_comm_dup);
    if (c->in_quo_comm) MPI_Comm_free(&c->quo_comm);
    QUO_free(c->quo);
    free(c);
    return 0;
}

int
p1_entry_point(p1_context_t *c)
{
    /* actually do threaded work? */
    if (c->in_quo_comm) {
        /* prep runtime environment */
        push_bind(c);
        /* DO WORK HERE */
        if (emit_bind_state(c, "-->")) return 1;
        /* revert to previous runtime environment */
        pop_bind(c);
    }
    if (QUO_SUCCESS != QUO_barrier(c->quo)) return 1;
    /* via mpi, share results with processes that were not in quo_comm, or any
     * other processes that need those data. sharing is caring. */
    return 0;
}
