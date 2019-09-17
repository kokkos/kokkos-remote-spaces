/**
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * Copyright 2013. Los Alamos National Security, LLC. This software was produced
 * under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National
 * Laboratory (LANL), which is operated by Los Alamos National Security, LLC for
 * the U.S. Department of Energy. The U.S. Government has rights to use,
 * reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS
 * ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 * to produce derivative works, such modified software should be clearly marked,
 * so as not to confuse it with the version available from LANL.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mpi.h"
#include "quo.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_OMP_H
#include <omp.h>
#endif
#ifdef HAVE_SCHED_H
#include <sched.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYSCALL_H
#include <sys/syscall.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

// TODO FIXME - global state is bad. Don't do this.
QUO_context context;
int rank, size;

void toString(char *text){
    printf("%s: Rank %d, thread: %d, PID: %d, CPU: %u\n",
           text, rank, omp_get_thread_num(), getpid(), sched_getcpu());
    fflush(stdout);
}

int main() {
    int number, i;

    if (MPI_SUCCESS != MPI_Init(NULL, NULL)) return 1;

    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &rank)) goto err;

    if (MPI_SUCCESS !=  MPI_Comm_size(MPI_COMM_WORLD, &size)) goto err;

    if (QUO_SUCCESS != QUO_create(&context)) goto err;

    if (QUO_SUCCESS !=
        QUO_bind_push(context, QUO_BIND_PUSH_PROVIDED, QUO_OBJ_SOCKET, rank%2))
    {
        printf("QUO_bind failed\n");
        goto err;
    }

    omp_set_nested(1);

#pragma omp parallel num_threads(4)
    {
        if (QUO_SUCCESS != QUO_bind_threads(context, QUO_OBJ_SOCKET, rank%2))
            printf("QUO_bind_threads failed\n");

        toString("First configuration");

#pragma omp parallel num_threads(2)
        {
            if (QUO_SUCCESS != QUO_bind_threads(context,
                                                QUO_OBJ_SOCKET, rank%2))
                printf("QUO_bind_threads failed\n");

            toString("Second configuration");
        }
    }

err:
    if(context)
        QUO_free(context);

    MPI_Finalize();
    return 0;
}
