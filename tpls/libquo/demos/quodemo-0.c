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

#include "quo.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpi.h"

/**
 * libquo demo code. enjoy.
 */

/**
 * SUGGESTED USE
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
 * elects some node ranks and distributes them onto all the sockets on the node
 */
static int
bindup_sockets(const context_t *c)
{
    /* if you are going to change bindings often, then cache this */
    if (c->noderank + 1 <= c->nsockets) {
        if (QUO_SUCCESS != QUO_bind_push(c->quo, QUO_BIND_PUSH_PROVIDED,
                                         QUO_OBJ_SOCKET, c->noderank)) {
            return 1;
        }
    }
    return 0;
}

/**
 * we can only safely pop bindings that were pushed, so those who were elected
 * to be the socket master can now revert their binding by calling pop.
 */
static int
binddown_sockets(const context_t *c)
{
    if (c->noderank + 1 <= c->nsockets) {
        if (QUO_SUCCESS != QUO_bind_pop(c->quo)) {
            return 1;
        }
    }
    return 0;
}

/**
 * returns whether or not our current cpuset (binding) falls within a particular
 * type and index. like: am i bound within socket 1? kinda thing.
 *
 * for example: if you are current bound to a core within socket 1, then this
 * routine will return 1. if you are not bound at all, this routine will also
 * return 1.
 */
static int
type_in_cur_bind(const context_t *c,
                 QUO_obj_type_t type,
                 int type_id,
                 int *in_cur_bind)
{
    if (QUO_SUCCESS != QUO_cpuset_in_type(c->quo, type, type_id, in_cur_bind)) {
        return 1;
    }
    return 0;
}

static int
cores_in_cur_bind_test(const context_t *c)
{
    int b0 = -1, blast = -1;
    if (type_in_cur_bind(c, QUO_OBJ_CORE, 0, &b0)) return 1;
    if (type_in_cur_bind(c, QUO_OBJ_CORE, c->ncores - 1, &blast)) return 1;

    printf("### [rank %d] core %d in current bind policy: %s\n",
           c->rank, 0, b0 ? "true" : "false");
    printf("### [rank %d] core %d in current bind policy: %s\n",
           c->rank, c->ncores - 1, blast ? "true" : "false");
    demo_emit_sync(c);
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
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
    if (0 == context->rank) {
        fprintf(stdout, "changing binding...\n");
        fflush(stdout);
    }
    if (bindup_sockets(context)) {
        bad_func = "bindup_sockets";
        goto out;
    }
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
    /* now test to see if core 0 and the last core are in the socket that we are
     * currently bound. */
    if (cores_in_cur_bind_test(context)) {
        bad_func = "cores_in_cur_bind_test";
        goto out;
    }
    /* now revert the previous policy */
    if (binddown_sockets(context)) {
        bad_func = "binddown_sockets";
        goto out;
    }
    if (0 == context->rank) {
        fprintf(stdout, "reverting binding change...\n");
        fflush(stdout);
    }
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
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
