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

#include "caller-driven-ex-p1.h"

typedef struct p1context_t {
    /* communicator used by p1 */
    MPI_Comm comm;
    /* size of p1_comm */
    int comm_size;
    /* my rank in p1_comm */
    int comm_rank;
    /* flag indicating whether or not i'm in the p1 group */
    int incomm;
} p1context_t;

static p1context_t p1;

static void
p1_emit_sync(const p1context_t *p1c)
{
    MPI_Barrier(p1c->comm);
    usleep((p1c->comm_rank) * 1000);
}

static int
push_bind(const context_t *c)
{
    /* p1 wants each pe to expand their bindings to the socket in which they are
     * currently bound. the idea here is that p0 will call us with a particular
     * binding policy, but we need a different one. we'll "bind up" to the
     * closest socket. notice that with QUO_BIND_PUSH_OBJ, the last argument
     * (the obj index [e.g socket 1]) is ignored. this is NOT the case when
     * using the QUO_BIND_PUSH_PROVIDED option. */
    if (QUO_SUCCESS != QUO_bind_push(c->quo, QUO_BIND_PUSH_OBJ,
                                     QUO_OBJ_SOCKET, -1)) {
        return 1;
    }
    return 0;
}

/* revert our binding policy so p0 can go about its business with its own
 * binding policy... */
static int
pop_bind(const context_t *c)
{
    if (QUO_SUCCESS != QUO_bind_pop(c->quo)) return 1;
    return 0;
}

int
p1_init(context_t *c,
        int np1s /* number of participants |p1who| */,
        int *p1who /* the participating ranks (MPI_COMM_WORLD) */)
{
    int rc = QUO_SUCCESS;
    if (0 == c->noderank) {
        printf("ooo [rank %d] %d p1pes initializing p1\n", c->rank, np1s);
        printf("ooo [rank %d] and they are: ", c->rank);
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
    MPI_Group world_group;
    MPI_Group p1_group;
    if (MPI_SUCCESS != MPI_Comm_group(MPI_COMM_WORLD, &world_group)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Group_incl(world_group, np1s,
                                      p1who, &p1_group)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Comm_create(MPI_COMM_WORLD,
                                       p1_group,
                                       &(p1.comm))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* am i in the new communicator? */
    p1.incomm = (MPI_COMM_NULL == p1.comm) ? 0 : 1;
    if (p1.incomm) {
        if (MPI_SUCCESS != MPI_Comm_size(p1.comm, &p1.comm_size)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
        if (MPI_SUCCESS != MPI_Comm_rank(p1.comm, &p1.comm_rank)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
    }
    /* for pretty print */
    usleep((c->rank) * 1000);
out:
    if (MPI_SUCCESS != MPI_Group_free(&world_group)) return 1;
    return (QUO_SUCCESS == rc) ? 0 : 1;
}

int
p1_fini(void)
{
    if (p1.incomm) {
        if (MPI_SUCCESS != MPI_Comm_free(&p1.comm)) return 1;
    }
    return 0;
}


int
p1_entry_point(context_t *c)
{
    /* change our binding */
    if (push_bind(c)) {
        fprintf(stderr, "push_bind failure in %s\n", __func__);
        return 1;
    }
    if (emit_bind_state(c, "ooo")) {
        fprintf(stderr, "emit_bind_state failure in %s\n", __func__);
        return 1;
    }
    p1_emit_sync(&p1);
    printf("ooo [rank %d] p1pe rank %d doing science in p1!\n",
           c->rank, p1.comm_rank);
    fflush(stdout);
    /* revert our binding policy */
    if (pop_bind(c)) {
        fprintf(stderr, "pop_bind failure in %s\n", __func__);
        return 1;
    }
    p1_emit_sync(&p1);
    return 0;
}
