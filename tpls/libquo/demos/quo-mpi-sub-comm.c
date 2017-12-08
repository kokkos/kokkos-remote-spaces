/**
 * Copyright (c) 2013-2017 Los Alamos National Security, LLC
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
#include <string.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpi.h"

typedef struct inf_t {
    int rank;
    int nranks;
    bool mpi_inited;
} inf_t;

typedef struct hw_info_t {
    int nnodes;
    int nnoderanks;
    int nsockets;
    int ncores;
    int npus;
} hw_info_t;

static int
init(inf_t *inf)
{
    if (MPI_SUCCESS != MPI_Init(NULL, NULL)) {
        return 1;
    }
    if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &(inf->nranks))) {
        return 1;
    }
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(inf->rank))) {
        return 1;
    }
    inf->mpi_inited = true;
    return 0;
}

static int
get_hw_info(QUO_context quo,
            hw_info_t *info)
{
    int qrc = QUO_SUCCESS;
    char *bad_func = NULL;

    if (QUO_SUCCESS != (qrc = QUO_nsockets(quo, &info->nsockets))) {
        bad_func = "QUO_nsockets";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_ncores(quo, &info->ncores))) {
        bad_func = "QUO_ncores";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_npus(quo, &info->npus))) {
        bad_func = "QUO_npus";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_nnodes(quo, &info->nnodes))) {
        bad_func = "QUO_nnodes";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_nqids(quo, &info->nnoderanks))) {
        bad_func = "QUO_nnodes";
        goto out;
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "xxx %s failure in: %s\n", __FILE__, bad_func);
        return 1;
    }
    return 0;
}

static int
print_comm_ranks(MPI_Comm comm)
{
    char *bad_func = NULL;
    int *ranks = NULL, *world_ranks = NULL;

    MPI_Group grp, world_grp;

    if (MPI_SUCCESS != MPI_Comm_group(MPI_COMM_WORLD, &world_grp)) {
        bad_func = "MPI_Comm_group";
        goto out;
    }
    if (MPI_SUCCESS != MPI_Comm_group(comm, &grp)) {
        bad_func = "MPI_Comm_group";
        goto out;
    }

    int grp_size = 0;
    if (MPI_SUCCESS != MPI_Group_size(grp, &grp_size)) {
        bad_func = "MPI_Group_size";
        goto out;
    }

    ranks = malloc(grp_size * sizeof(int));
    world_ranks = malloc(grp_size * sizeof(int));

    if (NULL == ranks || NULL == world_ranks) {
        bad_func = "malloc";
        goto out;
    }

    for (int i = 0; i < grp_size; i++) ranks[i] = i;

    if (MPI_SUCCESS != MPI_Group_translate_ranks(grp, grp_size, ranks,
                                                 world_grp, world_ranks)) {
        bad_func = "MPI_Group_translate_ranks";
        goto out;
    }

    for (int i = 0; i < grp_size; i++)
        printf("comm[%d] has world rank %d\n", i, world_ranks[i]);

    if (MPI_SUCCESS != MPI_Group_free(&grp)) {
        bad_func = "MPI_Group_free";
        goto out;
    }
    if (MPI_SUCCESS != MPI_Group_free(&world_grp)) {
        bad_func = "MPI_Group_free";
        goto out;
    }
out:
    if (ranks) free(ranks);
    if (world_ranks) free(world_ranks);

    if (NULL != bad_func) {
        fprintf(stderr, "xxx %s failure in: %s\n", __FILE__, bad_func);
        return 1;
    }
    return 0;
}

static int
test_QUO_get_mpi_comm_by_type(QUO_context quoc,
                              bool print)
{
    char *bad_func = NULL;
    int qrc = QUO_SUCCESS;
    if (print) {
        printf("### testing QUO_get_mpi_comm_by_type...\n");
    }
    //
    MPI_Comm quo_node_comm;
    if (QUO_SUCCESS != (qrc =
        QUO_get_mpi_comm_by_type(quoc, QUO_OBJ_MACHINE, &quo_node_comm))) {
        bad_func = "QUO_get_mpi_comm_by_type";
        goto out;
    }
    if (print) {
        if (print_comm_ranks(quo_node_comm)) {
            bad_func = "print_comm_ranks";
            goto out;
        }
    }
    if (MPI_SUCCESS != MPI_Comm_free(&quo_node_comm)) {
        bad_func = "MPI_Comm_free";
        goto out;
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "xxx %s failure in: %s\n", __FILE__, bad_func);
        return 1;
    }
    return 0;
}

static int
fini(inf_t *inf)
{
    if (inf->mpi_inited) MPI_Finalize();
    return 0;
}

int
main(void)
{
    int qrc = QUO_SUCCESS, erc = EXIT_SUCCESS;
    int qv = 0, qsv = 0;
    char *bad_func = NULL;
    QUO_context quo = NULL;
    inf_t info;
    memset(&info, 0, sizeof(inf_t));
    hw_info_t hw_info;

    if (init(&info)) {
        bad_func = "info";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_version(&qv, &qsv))) {
        bad_func = "QUO_version";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_create(&quo, MPI_COMM_WORLD))) {
        bad_func = "QUO_create";
        goto out;
    }
    if (get_hw_info(quo, &hw_info)) {
        bad_func = "get_hw_info";
        goto out;
    }
    int quo_id = 0;
    if (QUO_SUCCESS != (qrc = QUO_id(quo, &quo_id))) {
        bad_func = "QUO_id";
        goto out;
    }
    /* one process per node prints hw info. */
    if (0 == quo_id) {
        printf("### quo version: %d.%d ###\n", qv, qsv);
        printf("### nnodes: %d\n", hw_info.nnodes);
        printf("### nnoderanks: %d\n", hw_info.nnoderanks);
        printf("### nsockets: %d\n", hw_info.nsockets);
        printf("### ncores: %d\n", hw_info.ncores);
        printf("### npus: %d\n", hw_info.npus);
    }
    if (test_QUO_get_mpi_comm_by_type(quo, (0 == info.rank))) {
        bad_func = "test_QUO_get_mpi_comm_by_type";
        goto out;
    }
    if (QUO_SUCCESS != (qrc = QUO_free(quo))) {
        bad_func = "QUO_free";
        goto out;
    }
out:
    if (NULL != bad_func) {
        fprintf(stderr, "xxx %s failure in: %s\n", __FILE__, bad_func);
        erc = EXIT_FAILURE;
    }
    (void)fini(&info);
    return erc;
}
