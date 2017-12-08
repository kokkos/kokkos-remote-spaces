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
#include <assert.h>
#include <unistd.h>
#include <stdbool.h>

#include "mpi.h"

/**
 * this code tests the QUO_auto_distrib routine.
 */

typedef struct info_t {
    QUO_context q;
    int rank;
    int nranks;
    int noderank;
    int nnoderanks;
    int ncores;
    QUO_obj_type_t tres;
} info_t;

typedef void (*bp)(info_t *q);

typedef struct bind_t {
    /* bind name */
    char *bname;
    /* bind function pointer */
    bp bfp;
    /* pop function pointer */
    bp pfp;
} bind_t;

static void
emit_bind(info_t *i)
{
    char *binds = NULL;
    assert(QUO_SUCCESS == QUO_stringify_cbind(i->q, &binds));
    printf("bbb %d %s\n", i->rank, binds);
    free(binds);
}

static void
tsync(const info_t *i)
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep((i->rank) * 1000);
}

static void
popbind(info_t *i)
{
    assert(QUO_SUCCESS == QUO_bind_pop(i->q));
}

static void
nobind(info_t *i)
{
    assert(QUO_SUCCESS == QUO_bind_push(i->q, QUO_BIND_PUSH_PROVIDED,
                                        QUO_OBJ_MACHINE, 0));
}

static void
tightbind(info_t *i)
{
    assert(QUO_SUCCESS == QUO_bind_push(i->q, QUO_BIND_PUSH_PROVIDED,
                                        QUO_OBJ_CORE, i->noderank));
}

static void
insanity(info_t *i)
{
    if (0 == (i->noderank % 2)) {
        assert(QUO_SUCCESS == QUO_bind_push(i->q, QUO_BIND_PUSH_OBJ,
                                            QUO_OBJ_MACHINE, -1));
    }
    else {
        assert(QUO_SUCCESS == QUO_bind_push(i->q, QUO_BIND_PUSH_PROVIDED,
                                            QUO_OBJ_CORE, i->noderank));
    }
}

static bind_t binds[] =
{
    {"complete set overlap", nobind, popbind},
    {"no set overlap", tightbind, popbind},
    {"some set overlap", insanity, popbind},
    {NULL, NULL, NULL}
};

int
main(int argc, char **argv)
{
    info_t info;
    int work_member = 0, max_members_per_res = 2;
    int nres = 0, rc = EXIT_SUCCESS;
    info.tres = QUO_OBJ_NUMANODE;

    assert(MPI_SUCCESS == MPI_Init(&argc, &argv));
    assert(QUO_SUCCESS == QUO_create(&info.q, MPI_COMM_WORLD));
    assert(MPI_SUCCESS == MPI_Comm_size(MPI_COMM_WORLD, &info.nranks));
    assert(MPI_SUCCESS == MPI_Comm_rank(MPI_COMM_WORLD, &info.rank));
    assert(QUO_SUCCESS == QUO_id(info.q, &info.noderank));
    assert(QUO_SUCCESS == QUO_nqids(info.q, &info.nnoderanks));
    assert(QUO_SUCCESS == QUO_nnumanodes(info.q, &nres));
    assert(QUO_SUCCESS == QUO_ncores(info.q, &info.ncores));
    setbuf(stdout, NULL);
    if (info.ncores < info.nnoderanks) {
        if (0 == info.noderank) {
            fprintf(stderr, "xxx cannot continue: %d core(s) < %d rank(s).\n",
                    info.ncores, info.nranks);
        }
        rc = EXIT_FAILURE;
        goto done;
    }
    if (0 == nres) {
        assert(QUO_SUCCESS == QUO_nsockets(info.q, &nres));
        info.tres = QUO_OBJ_SOCKET;
        if (0 == nres) {
            fprintf(stderr, "xxx cannot continue with test! xxx\n");
            rc = EXIT_FAILURE;
            goto done;
        }
    }
    if (0 == info.rank) {
        printf("ooo starting test: max mems per %d = %d (see quo.h) ooo\n",
               (int)info.tres, max_members_per_res);
    }
    for (int i = 0; NULL != binds[i].bname; ++i) {
        tsync(&info);
        if (0 == info.rank) printf("--- %s\n", binds[i].bname);
        binds[i].bfp(&info);
        emit_bind(&info);
        tsync(&info);
        assert(QUO_SUCCESS == QUO_auto_distrib(info.q, info.tres,
                                               max_members_per_res,
                                               &work_member));
        printf("*** rank %d work member: %d\n", info.rank, work_member);
        tsync(&info);
        binds[i].pfp(&info);
        tsync(&info);
    }
done:
    assert(QUO_SUCCESS == QUO_free(info.q));
    assert(MPI_SUCCESS == MPI_Finalize());
    return rc;
}
