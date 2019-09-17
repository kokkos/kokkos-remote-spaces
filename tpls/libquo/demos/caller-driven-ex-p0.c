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

#include "caller-driven-ex-common.h"
#include "caller-driven-ex-p1.h"
#include "quo.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpi.h"

/**
 * libquo demo code that illustrates how two libraries interact. in this code
 * the caller makes all the decisions about how its target library's environment
 * should be set up.
 */

/**
 * demo finalize.
 */
static int
fini(context_t *c)
{
    if (!c) return 1;
    if (QUO_SUCCESS != QUO_free(c->quo)) return 1;
    /* finalize mpi AFTER QUO_free - we may mpi in our free */
    if (c->mpi_inited) {
        MPI_Finalize();
    }
    if (c->cbindstr) free(c->cbindstr);
    free(c);
    return 0;
}

/**
 * i'm being really sloppy here. ideally, one should probably save the rc and
 * then display or do some other cool thing with it. don't be like this code. if
 * QUO_create fails, then someone using this could just continue without the
 * awesomeness that is libquo. they cleanup after themselves, so things *should*
 * be in an okay state after an early failure. the failures may be part of a
 * larger problem, however. if you are reading this, you are probably the first
 * -- thanks! */
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

    /* ////////////////////////////////////////////////////////////////////// */
    /* now libquo can be initialized. libquo uses mpi, so that needs to be
     * initialized first. */
    /* ////////////////////////////////////////////////////////////////////// */

    /* can be called at any point -- even before construct. */
    if (QUO_SUCCESS != QUO_version(&(newc->qv), &(newc->qsv))) goto err;
    /* relatively expensive call. you only really want to do this once at the
     * beginning of time and pass the context all over the place within your
     * code.
     */
    if (QUO_SUCCESS != QUO_create(&newc->quo, MPI_COMM_WORLD)) goto err;
    /* mpi initialized at this point */
    newc->mpi_inited = true;
    /* return pointer to allocated resources */
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
emit_node_basics(const context_t *c)
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

#if 0 /* old way */
/**
 * this is where we set our policy regarding who will actually call into p1 and
 * do work. the others will sit in a barrier an wait for the workers to finish.
 *
 * this particular example distributes the workers among all the sockets on the
 * system, but you can imagine doing the same for NUMA nodes, for example. if
 * there are no NUMA nodes on the system, then fall back to something else.
 */
static int
get_p1pes(context_t *c,
                 bool *working,
                 int *nworkers,
                 int **workers)
{
    /* points to an array that stores the number of elements in the
     * rank_ids_bound_to_socket array at a particular socket index */
    int *nranks_bound_to_socket = NULL;
    /* array of pointers that point to the smp ranks that cover a particular
     * socket at a particular socket index. you can think of this as a 2D
     * matrix where [i][j] is the ith socket that smp rank j covers. */
    int **rank_ids_bound_to_socket = NULL;
    int rc = QUO_ERR;
    int work_contrib = 0;
    /* array that hold whether or not a particular rank is going to do work */
    int *work_contribs = NULL;
    int *worker_ranks = NULL;

    *nworkers = 0; *workers = NULL; *working = false;
    /* allocate some memory for our arrays */
    nranks_bound_to_socket = calloc(c->nsockets,
                                    sizeof(*nranks_bound_to_socket));
    if (!nranks_bound_to_socket) return 1;
    /* allocate pointer array */
    rank_ids_bound_to_socket = calloc(c->nsockets,
                                      sizeof(*rank_ids_bound_to_socket));
    if (!rank_ids_bound_to_socket) {
        free(nranks_bound_to_socket); nranks_bound_to_socket = NULL;
        return 1;
    }
    /* grab the smp ranks (node ranks) that are in each socket */
    for (int socket = 0; socket < c->nsockets; ++socket) {
        rc = QUO_qids_in_type(c->quo,
                                  QUO_OBJ_SOCKET,
                                  socket,
                                  &(nranks_bound_to_socket[socket]),
                                  &(rank_ids_bound_to_socket[socket]));
        if (QUO_SUCCESS != rc) {
            if (rank_ids_bound_to_socket) free(rank_ids_bound_to_socket);
            if (nranks_bound_to_socket) free(nranks_bound_to_socket);
            return 1;
        }
    }
    /* everyone has the same info on the node, so just have node rank 0 display
     * the list of smp ranks that cover each socket on the system. */
    for (int socket = 0; socket < c->nsockets; ++socket) {
        for (int rank = 0; rank < nranks_bound_to_socket[socket]; ++rank) {
            if (0 == c->noderank) {
                printf("### [rank %d] rank %d covers socket %d\n", c->rank,
                       rank_ids_bound_to_socket[socket][rank], socket);
            }
        }
    }
    demo_emit_sync(c);
    /* ////////////////////////////////////////////////////////////////////// */
    /* now elect the workers. NOTE: the setup for this was fairly ugly and not
     * cheap, so caching the following result may be a good thing.
     * o setup.
     * o elect workers for a particular regime.
     * o cache that result for later use to avoid setup and query costs.
     */
    /* ////////////////////////////////////////////////////////////////////// */
    /* maximum number of workers for a given resource (socket in this case). we
     * are statically setting this number, but one could imagine this number
     * being exchanged at the beginning of time in a real application. */
    int tot_workers = 0;
    int max_workers_per_res = 2;
    /* whether or not we are already assigned to a particular resource */
    bool res_assigned = false;
    for (int socket = 0; socket < c->nsockets; ++socket) {
        for (int rank = 0; rank < nranks_bound_to_socket[socket]; ++rank) {
            /* if i'm not already assigned to a particular resource and
             * my current cpuset covers the resource in question and
             * someone else won't be assigned to that resource
             */
            if (!res_assigned &&
                c->noderank == rank_ids_bound_to_socket[socket][rank] &&
                rank < max_workers_per_res) {
                res_assigned = true;
                printf("### [rank %d] smp rank %d assigned to socket %d\n",
                        c->rank, c->noderank, socket);
            }
        }
    }
    work_contrib = res_assigned ? 1 : 0;
    /* array that hold whether or not a particular rank is going to do work */
    work_contribs = calloc(c->nranks, sizeof(*work_contribs));
    if (!work_contribs) {
        rc = QUO_ERR_OOR;
        goto out;
    }
    if (MPI_SUCCESS != (rc = MPI_Allgather(&work_contrib, 1, MPI_INT,
                                           work_contribs, 1, MPI_INT,
                                           MPI_COMM_WORLD))) {
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
    *working = (bool)work_contrib;
    *nworkers = tot_workers;
    *workers = worker_ranks;
    demo_emit_sync(c);
out:
    /* the resources returned by QUO_qids_in_type must be freed by us */
    for (int i = 0; i < c->nsockets; ++i) {
        if (rank_ids_bound_to_socket[i]) free(rank_ids_bound_to_socket[i]);
    }
    if (rank_ids_bound_to_socket) free(rank_ids_bound_to_socket);
    if (nranks_bound_to_socket) free(nranks_bound_to_socket);
    if (work_contribs) free(work_contribs);
    if (QUO_SUCCESS != rc) {
        if (worker_ranks) free(worker_ranks);
    }
    return (QUO_SUCCESS == rc) ? 0 : 1;
}
#else /* new way */
/**
 * this is where we set our policy regarding who will actually call into p1 and
 * do work. the others will sit in a barrier an wait for the workers to finish.
 *
 * this particular example distributes the workers among all the sockets on the
 * system, but you can imagine doing the same for NUMA nodes, for example. if
 * there are no NUMA nodes on the system, then fall back to something else.
 */
static int
get_p1pes(context_t *c,
                 bool *working,
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
                                           MPI_COMM_WORLD))) {
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
    *working = (bool)res_assigned;
    *nworkers = tot_workers;
    *workers = worker_ranks;
    demo_emit_sync(c);
out:
    if (work_contribs) free(work_contribs);
    if (QUO_SUCCESS != rc) {
        if (worker_ranks) free(worker_ranks);
    }
    return (QUO_SUCCESS == rc) ? 0 : 1;
}
#endif

int
main(void)
{
    int erc = EXIT_SUCCESS;
    char *bad_func = NULL;
    context_t *context = NULL;
    /* flag indicating whether or not i'm a p1pe (calling into p1) */
    bool p1pe = false;
    /* total number of p1pes */
    int tot_p1pes = 0;
    /* the MPI_COMM_WORLD ranks of the p1pes */
    int *p1pes = NULL;

    /* ////////////////////////////////////////////////////////////////////// */
    /* init code -- note that the top-level package must do this */
    /* ////////////////////////////////////////////////////////////////////// */
    if (init(&context)) {
        bad_func = "init";
        goto out;
    }

    /* ////////////////////////////////////////////////////////////////////// */
    /* libquo is now ready for service and mpi is good to go. */
    /* ////////////////////////////////////////////////////////////////////// */

    /* first gather some info so we can base our decisions on our current
     * situation. */
    if (sys_grok(context)) {
        bad_func = "sys_grok";
        goto out;
    }
    /* show some info that we got about our nodes - one per node */
    if (emit_node_basics(context)) {
        bad_func = "emit_node_basics";
        goto out;
    }
    /* display our binding */
    if (emit_bind_state(context, "###")) {
        bad_func = "emit_bind_state";
        goto out;
    }
    demo_emit_sync(context);
    /* ////////////////////////////////////////////////////////////////////// */
    /* setup needed before we can init p1 */
    /* ////////////////////////////////////////////////////////////////////// */
    if (get_p1pes(context, &p1pe, &tot_p1pes, &p1pes)) {
        bad_func = "get_p1pes";
        goto out;
    }
    /* ////////////////////////////////////////////////////////////////////// */
    /* init p1 by letting it know the ranks that are going to do work.
     * EVERY ONE IN MPI_COMM_WORLD CALLS THIS (sorry about the yelling) */
    /* ////////////////////////////////////////////////////////////////////// */
    if (p1_init(context, tot_p1pes, p1pes)) {
        bad_func = "p1_init";
        goto out;
    }
    if (0 == context->noderank) {
        printf("### [rank %d] %d p0pes doing science in p0!\n",
               context->rank, context->nnoderanks);
    }
    /* time for p1 to do some work with some of the ranks */
    if (p1pe) {
        if (p1_entry_point(context)) {
            bad_func = "p1_entry_point";
            goto out;
        }
        /* signals completion within p1 */
        if (QUO_SUCCESS != QUO_barrier(context->quo)) {
            bad_func = "QUO_barrier";
            goto out;
        }
    }
    else {
        /* non p1pes wait in a barrier */
        if (QUO_SUCCESS != QUO_barrier(context->quo)) {
            bad_func = "QUO_barrier";
            goto out;
        }
    }
    demo_emit_sync(context);
    /* display our binding */
    if (emit_bind_state(context, "###")) {
        bad_func = "emit_bind_state";
        goto out;
    }
    demo_emit_sync(context);
    if (0 == context->noderank) {
        printf("### [rank %d] %d p0pes doing science in p0!\n",
               context->rank, context->nnoderanks);
    }
    if (p1_fini()) {
        bad_func = "p1_fini";
        goto out;
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "XXX %s failure in: %s\n", __FILE__, bad_func);
        erc = EXIT_FAILURE;
    }
    if (p1pes) free(p1pes);
    (void)fini(context);
    return erc;
}
