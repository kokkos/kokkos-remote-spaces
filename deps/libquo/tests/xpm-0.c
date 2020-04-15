/*
 * Copyright (c) 2017      Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the libquo project. See the LICENSE file at the
 * top-level directory of this distribution.
 */

#include "quo.h"
#include "quo-xpm.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi.h"

int
main(int argc, char **argv)
{
    QUO_context q = NULL;
    QUO_xpm_context xpm = NULL;

    int numpe = 0, rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numpe);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int qid = 0, nqid = 0;
    QUO_create(&q, MPI_COMM_WORLD);
    QUO_id(q, &qid);
    QUO_nqids(q, &nqid);

    int ni = 1 + rank;
    size_t local_size = ni * sizeof(int);

    QUO_xpm_allocate(q, local_size, &xpm);

    QUO_xpm_view_t my_view;

    QUO_xpm_view_local(xpm, &my_view);

    int *my_array = (int *)my_view.base;
    int nlocal = my_view.extent / sizeof(int);

    for (int i = 0; i < nlocal; ++i) {
        my_array[i] = qid;
    }

    for (int i = 0; i < nlocal; ++i) {
        printf("%d: my_array[i] = %d\n", qid, my_array[i]);
    }

    QUO_barrier(q);

    if (0 == qid) {
        printf("xxxxxxxxxxxxxxxxxxx\n");
        for (int r = 0; r < nqid; ++r) {
            QUO_xpm_view_t r_view;
            QUO_xpm_view_by_qid(xpm, r, &r_view);
            int *r_array = (int *)r_view.base;
            int n_remote = r_view.extent / sizeof(int);

            for (int i = 0; i < n_remote; ++i) {
                r_array[i] = n_remote;
            }
        }
    }

    QUO_barrier(q);

    for (int i = 0; i < nlocal; ++i) {
        printf("%d: my_array'[i] = %d\n", qid, my_array[i]);
    }

    QUO_barrier(q);

    if (0 == qid) {
        printf("xxxxxxxxxxxxxxxxxxx\n");
        QUO_xpm_view_t range_view;
        QUO_xpm_view_by_qid_range(xpm, 0, nqid - 1, &range_view);
        int *r_array = (int *)range_view.base;
        int n_remote = range_view.extent / sizeof(int);

        for (int i = 0; i < n_remote; ++i) {
            r_array[i] = -1;
        }
    }

    QUO_barrier(q);

    for (int i = 0; i < ni; ++i) {
        printf("%d: my_array''[i] = %d\n", qid, my_array[i]);
    }

    QUO_xpm_free(xpm);

    QUO_free(q);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
