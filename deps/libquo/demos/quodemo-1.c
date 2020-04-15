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
 * another quo demo code that shows how one might implement binding policies
 * based on node process information, hardware info, and code policy.
 */

typedef struct context_t {
    /* my host's name */
    char hostname[MPI_MAX_PROCESSOR_NAME];
    /* my rank */
    int rank;
    /* number of ranks in MPI_COMM_WORLD */
    int nranks;
    /* number of nodes in our job */
    int nnodes;
    /* number of numa nodes on this node */
    int nnuma;
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
    /* flag indicating whether or not i've pushed a policy that hasn't been
     * popped yet. this is hacky, but illustrates a nice point. */
    bool pushed_policy;
    /* our quo context (the thing that gets passed around all over the place).
     * filler words that make this comment line look mo better... */
    QUO_context quo;
} context_t;

static const char *
stringify_type(QUO_obj_type_t typ)
{
    switch (typ) {
        case QUO_OBJ_MACHINE: return "machine";
        case QUO_OBJ_NUMANODE: return "numa node";
        case QUO_OBJ_SOCKET: return "socket";
        case QUO_OBJ_CORE: return "core";
        case QUO_OBJ_PU: return "processing unit";
        default: return "???";
    }
    return "???";
}

/**
 * rudimentary "pretty print" routine. not needed in real life...
 */
static inline void
demo_emit_sync(const context_t *c)
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep((c->rank) * 5000);
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
    int name_len = 0;
    context_t *newc = NULL;
    /* alloc our context */
    if (NULL == (newc = calloc(1, sizeof(*newc)))) return 1;
    /* libquo requires that MPI be initialized before its init is called */
    if (MPI_SUCCESS != MPI_Init(NULL, NULL)) return 1;
    /* gather some basic job info from our mpi lib */
    if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &(newc->nranks))) goto err;
    /* ...and more */
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(newc->rank))) goto err;
    /* stash my host's name */
    if (MPI_SUCCESS != MPI_Get_processor_name(newc->hostname, &name_len)) {
        goto err;
    }
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
    if (QUO_SUCCESS != QUO_nobjs_in_type_by_type(c->quo,
                                                 QUO_OBJ_MACHINE,
                                                 0,
                                                 QUO_OBJ_NUMANODE,
                                                 &c->nnuma)) {
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
        printf("### nnumanodes: %d\n", c->nnuma);
        printf("### nsockets: %d\n", c->nsockets);
        printf("### ncores: %d\n", c->ncores);
        printf("### npus: %d\n", c->npus);
        fflush(stdout);
    }
    demo_emit_sync(c);
    return 0;
}

/**
 *
 */
static int
bindup_node(context_t *c)
{
    /* choose any node rank here. i use node rank 0 (quid 0) because it's easy.
     * */
    if (0 == c->noderank) {
        printf("### [rank %d on %s] expanding my cpuset for threading!\n",
                c->rank, c->hostname);
        /* QUO_BIND_PUSH_OBJ, so the last argument doesn't matter */
        int rc = QUO_bind_push(c->quo, QUO_BIND_PUSH_OBJ,
                               QUO_OBJ_MACHINE, -1);
        if (QUO_SUCCESS != rc) {
            fprintf(stderr, "%s fails with rc: %d\n",
                    "QUO_bind_push", rc);
            return 1;
        }
        /* i pushed a policy */
        c->pushed_policy = true;
        demo_emit_sync(c);
    }
    else {
        printf("--- [rank %d on %s] going to sleep...\n",
                c->rank, c->hostname);
        demo_emit_sync(c);
    }
    return 0;
}

/**
 * bind to what is provided. the assumption is that the binding policy for all
 * procs that are calling this are "binding up." this demo binds all ranks on a
 * node to the provided "this" that encloses the current policy. for example, if
 * i'm bound to a core and "bind up" to a socket, then i'll be bound to my
 * enclosing socket.
 */
static int
bindup_to_this(context_t *c,
               QUO_obj_type_t this)
{
    demo_emit_sync(c);
    printf("### [rank %d on %s] expanding my cpuset for threading!\n",
            c->rank, c->hostname);
    /* QUO_BIND_PUSH_OBJ, so the last argument doesn't matter. QUO will do smart
     * things here (ish). This is where how the things were launched matters. */
    int rc = QUO_bind_push(c->quo, QUO_BIND_PUSH_OBJ, this, -1);
    if (QUO_SUCCESS != rc) {
        fprintf(stderr, "%s fails with rc: %d\n",
                "QUO_bind_push", rc);
        return 1;
    }
    c->pushed_policy = true;
    /* i pushed a policy */
    demo_emit_sync(c);
    return 0;
}

/**
 * we can only safely pop bindings that were pushed, so those who were elected
 * to be the socket master can now revert their binding by calling pop.
 */
static int
pop_bind_policy(context_t *c)
{
    /* if i pushed a policy, then revert it. only one deep in our stack in this
     * demo, so that's why the bool works. more complicated binding stack
     * situations will require a bit more code. that is, quo let's you push an
     * arbitrary amount of binding policies, but this code only ever pushed one
     * at a time. */
    if (c->pushed_policy) {
        if (QUO_SUCCESS != QUO_bind_pop(c->quo)) {
            return 1;
        }
        c->pushed_policy = false;
    }
    return 0;
}

static int
node_process_info(const context_t *c)
{
    demo_emit_sync(c);
    /* if i'm node rank (from quo's perspective), then print out some info. */
    if (0 == c->noderank) {
        printf("### [rank %d] %04d mpi processes share this node: %s\n",
              c->rank, c->nnoderanks, c->hostname);
    }
    demo_emit_sync(c);
    return 0;
}

static int
one_rank_all_res(context_t *context)
{
    int erc = 0;
    char *bad_func = NULL;
    demo_emit_sync(context);
    /* for the first test, we will just use ALL the resources on the node for
     * threading and quiesce all but one mpi process on each node. NOTE: doing
     * so will show that the process is NOT bound -- because it isn't.  its
     * cpuset is the widest it can be on a particular server, so it is NOT
     * bound. */
    if (0 == context->rank) {
        fprintf(stdout,
                "*****************************************\n"
                "*** 1 rank given all node resources *****\n"
                "*****************************************\n");
        fflush(stdout);
    }
    if (bindup_node(context)) {
        bad_func = "bindup_node";
        goto out;
    }
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
    /* now revert the previous policy */
    if (pop_bind_policy(context)) {
        bad_func = "pop_bind_policy";
        goto out;
    }
    if (0 == context->rank) {
        fprintf(stdout, "reverted binding change...\n");
        fflush(stdout);
    }
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
out:
    if (bad_func) {
        fprintf(stderr, "XXX %s failure in: %s\n", __FILE__, bad_func);
        erc = 1;
    }
    return erc;
}

static int
all_ranks_some_res(context_t *context)
{
    int erc = 0;
    char *bad_func = NULL;
    demo_emit_sync(context);
    /* everyone has one of these -- default */
    QUO_obj_type_t what_to_bindup_to = QUO_OBJ_MACHINE;
    /* pick some resource to "bind up" to. */
    if (0 != context->nnuma) {
        what_to_bindup_to = QUO_OBJ_NUMANODE;
    }
    else if (0 != context->nsockets) {
        what_to_bindup_to = QUO_OBJ_SOCKET;
    }
    if (0 == context->rank) {
        fprintf(stdout,
                "*****************************************\n"
                "*** all ranks given some resources       \n"
                "*** resource is: %s                      \n"
                "*****************************************\n",
                stringify_type(what_to_bindup_to));
        fflush(stdout);
    }
    if (bindup_to_this(context, what_to_bindup_to)) {
        bad_func = "bindup_node";
        goto out;
    }
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
    /* now revert the previous policy */
    if (pop_bind_policy(context)) {
        bad_func = "pop_bind_policy";
        goto out;
    }
    if (0 == context->rank) {
        fprintf(stdout, "reverted binding change...\n");
        fflush(stdout);
    }
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "XXX %s failure in: %s\n", __FILE__, bad_func);
        erc = 1;
    }
    return erc;
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
    /* let the people know what's going on... one per node. this info is on a
     * per-node basis, folks... so, running on a set of heterogeneous nodes will
     * yeild different results. policy can be set on a per node basis. */
    if (emit_node_basics(context)) {
        bad_func = "emit_node_basics";
        goto out;
    }
    /* now show what the default bind state of each process in the job is */
    if (emit_bind_state(context)) {
        bad_func = "emit_bind_state";
        goto out;
    }
    /* ////////////////////////////////////////////////////////////////////// */
    /* ////////////////////////////////////////////////////////////////////// */
    /* this is where the real work begins ...                                 */
    /* ////////////////////////////////////////////////////////////////////// */
    /* ////////////////////////////////////////////////////////////////////// */
    demo_emit_sync(context);
    if (0 == context->rank) {
        fprintf(stdout,
                "*****************************************\n"
                "***      starting the demo pieces     ***\n"
                "*****************************************\n");
        fflush(stdout);
    }
    demo_emit_sync(context);
    /* first emit how many mpi processes share a node with me.
     * NOTE: this is a per-node thing and the result may differ across nodes.
     * Processes that share a node will of course see the same answer, but the
     * answer may be different across nodes (i.e. servers/compute nodes). */
    if (node_process_info(context)) {
        bad_func = "node_process_info";
        goto out;
    }
    /* demo 0 */
    if (one_rank_all_res(context)) {
        bad_func = "one_rank_all_res";
        goto out;
    }
    /* demo 1 */
    if (all_ranks_some_res(context)) {
        bad_func = "all_ranks_some_res";
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
