/*
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

/**
 * @file quo-auto-distrib.c
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "quo.h"
#include "quo-private.h"
#include "quo-set.h"
#include "quo-mpi.h"
#include "quo-sm.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif
#ifdef HAVE_STDBOOL_H
#include <stdbool.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

/**
 * \note Caller is responsible for freeing returned resources.
 */
static int
get_qids_in_target_type(QUO_t *q,
                        QUO_obj_type_t target,
                        int n_target,
                        int **out_nranks_in_res,
                        int ***out_rank_ids_in_res)
{
    if (!q || n_target <= 0 || !out_nranks_in_res || !out_rank_ids_in_res) {
        return QUO_ERR_INVLD_ARG;
    }
    /* Set nice default return values. */
    *out_nranks_in_res = NULL;
    *out_rank_ids_in_res = NULL;

    int *nranks_in_res = NULL;
    int **rank_ids_in_res = NULL;
    int rc = QUO_ERR;
    /* Communication structures. */
    MPI_Comm node_comm;
    /* Points to a unique (node-local) path name for inter-process affinity
     * info. */
    char *sm_seg_path = NULL;
    quo_sm_t *smseg = NULL;

    /* Get node communicator so we can chat with our friends. */
    if (QUO_SUCCESS != (rc = quo_mpi_get_node_comm(q->mpi, &node_comm))) {
        QUO_ERR_MSGRC("quo_mpi_get_node_comm", rc);
        goto out;
    }
    /* Generate and agree upon a unique (node-local) path name. */
    if (QUO_SUCCESS != (rc = quo_mpi_xchange_uniq_path(q->mpi,
                                                       "qaff",
                                                       &sm_seg_path))) {
        QUO_ERR_MSGRC("quo_mpi_xchange_uniq_path", rc);
        goto out;
    }
    /* Build the shared-memory instance. */
    if (QUO_SUCCESS != (rc = quo_sm_construct(&smseg))) {
        QUO_ERR_MSGRC("quo_sm_construct", rc);
        goto out;
    }
    /* Allocate some memory for our arrays. */
    nranks_in_res = calloc(n_target, sizeof(*nranks_in_res));
    if (!nranks_in_res) {
        QUO_OOR_COMPLAIN();
        rc = QUO_ERR_OOR;
        goto out;
    }
    /* Allocate pointer array. */
    rank_ids_in_res = calloc(n_target, sizeof(*rank_ids_in_res));
    if (!rank_ids_in_res) {
        QUO_OOR_COMPLAIN();
        rc = QUO_ERR_OOR;
        goto out;
    }
    /* Let one process query for QUO_qids_in_type info that will then be shared
     * via shared-memory. */
    if (0 == q->qid) {
        /* Query for the smp ranks (node ranks) that cover each resource. */
        for (int rid = 0; rid < n_target; ++rid) {
            rc = QUO_qids_in_type(q, target, rid,
                                  &(nranks_in_res[rid]),
                                  &(rank_ids_in_res[rid]));
            if (QUO_SUCCESS != rc) goto out;
        }
        /* Now that we have that info, now calculate how large of a
         * shared-memory segment is needed (in bytes). */
        int sm_seg_len = 0;
        const int header_len = (n_target * sizeof(*nranks_in_res));
        /* The first bit will be the "header" info. */
        sm_seg_len += header_len;
        /* The next bit will be dedicated to the rank_ids_in_res table. We have
         * enough information in the header to reconstruct the table, which we
         * are going to store as a flatten 1D array in the shared-memory
         * segment. */
        int num_entries = 0;
        for (int i = 0; i < n_target; ++i) {
            num_entries += nranks_in_res[i];
        }
        sm_seg_len += (num_entries * sizeof(*nranks_in_res));
        /* Now that we know the size of the buffer, share that info. */
        if (QUO_SUCCESS != (rc = quo_mpi_bcast(&sm_seg_len, 1,
                                               MPI_INT, 0, node_comm))) {
            QUO_ERR_MSGRC("quo_mpi_bcast", rc);
            goto out;
        }
        /* Create the shared-memory segment. */
        if (QUO_SUCCESS != (rc = quo_sm_segment_create(smseg,
                                                       sm_seg_path,
                                                       sm_seg_len))) {
            QUO_ERR_MSGRC("quo_sm_segment_create", rc);
            goto out;
        }
        /* Get base of the shared-memory segment (starting point for header). */
        char *headerp = (char *)quo_sm_get_basep(smseg);
        /* Fill in the header. */
        (void)memmove(headerp, nranks_in_res, header_len);
        /* Copy a flattened version of the rank_ids_in_res table into the
         * shared-memory segment. Note that the tabular data starting point is
         * offset by header_len bytes. */
        char *tabp = (headerp + header_len);
        for (int i = 0; i < n_target; ++i) {
            const size_t nbytes = nranks_in_res[i] * sizeof(*nranks_in_res);
            (void)memmove(tabp, rank_ids_in_res[i], nbytes);
            tabp += nbytes;
        }
        /* Signal completion. */
        if (QUO_SUCCESS != (rc = quo_mpi_sm_barrier(q->mpi))) {
            QUO_ERR_MSGRC("quo_mpi_sm_barrier", rc);
            goto out;
        }
        /* Wait for attach completion. */
        if (QUO_SUCCESS != (rc = quo_mpi_sm_barrier(q->mpi))) {
            QUO_ERR_MSGRC("quo_mpi_sm_barrier", rc);
            goto out;
        }
        /* Cleanup after everyone is done. */
        (void)quo_sm_unlink(smseg);
    }
    else {
        int sm_seg_len = 0;
        /* Get shared-memory segment size. */
        if (QUO_SUCCESS != (rc = quo_mpi_bcast(&sm_seg_len, 1,
                                               MPI_INT, 0, node_comm))) {
            QUO_ERR_MSGRC("quo_mpi_bcast", rc);
            goto out;
        }
        /* Wait for the data to be published. */
        if (QUO_SUCCESS != (rc = quo_mpi_sm_barrier(q->mpi))) {
            QUO_ERR_MSGRC("quo_mpi_sm_barrier", rc);
            goto out;
        }
        if (QUO_SUCCESS!= (rc = quo_sm_segment_attach(smseg,
                                                      sm_seg_path,
                                                      sm_seg_len))) {
            QUO_ERR_MSGRC("quo_sm_segment_attach", rc);
            goto out;
        }
        /* Signal attach attach completion. */
        if (QUO_SUCCESS != (rc = quo_mpi_sm_barrier(q->mpi))) {
            QUO_ERR_MSGRC("quo_mpi_sm_barrier", rc);
            goto out;
        }
        /* Reconstruct structures to pass to caller. */
        /* Get base of the shared-memory segment (starting point for header). */
        char *headerp = (char *)quo_sm_get_basep(smseg);
        /* Get first bit of info from the header. */
        const int header_len = (n_target * sizeof(*nranks_in_res));
        (void)memmove(nranks_in_res, headerp, header_len);
        /* Copy from a flattened version of the rank_ids_in_res table into the
         * "real" 2D table that will be passed to the caller. */
        char *tabp = (headerp + header_len);
        for (int i = 0; i < n_target; ++i) {
            const size_t nranks = nranks_in_res[i];
            const size_t nbytes = nranks * sizeof(*nranks_in_res);
            rank_ids_in_res[i] = calloc(nranks, sizeof(*nranks_in_res));
            if (NULL == rank_ids_in_res[i]) {
                QUO_OOR_COMPLAIN();
                rc = QUO_ERR_OOR;
                goto out;
            }
            /* Copy out. */
            (void)memmove(rank_ids_in_res[i], tabp, nbytes);
            tabp += nbytes;
        }
    }
out:
    if (QUO_SUCCESS != rc) {
        if (rank_ids_in_res) {
            for (int i = 0; i < n_target; ++i) {
                if (rank_ids_in_res[i]) free(rank_ids_in_res[i]);
            }
            free(rank_ids_in_res);
        }
        if (nranks_in_res) free(nranks_in_res);
    }
    else {
        *out_nranks_in_res = nranks_in_res;
        *out_rank_ids_in_res = rank_ids_in_res;
    }
    /* General cleanup. */
    if (sm_seg_path) free(sm_seg_path);
    (void)quo_sm_destruct(smseg);

    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
QUO_auto_distrib(QUO_t *q,
                 QUO_obj_type_t distrib_over_this,
                 int max_qids_per_res_type,
                 int *out_selected)
{
    /* total number of target resources. */
    int nres = 0;
    /* points to an array that stores the number of elements in the
     * rank_ids_in_res array at a particular resource index. */
    int *nranks_in_res = NULL;
    /* array of pointers that point to the smp ranks that cover a particular
     * hardware resource at a particular index. you can think of this as a 2D
     * matrix where [i][j] is the ith hardware resource that smp rank j covers.
     */
    int **rank_ids_in_res = NULL;
    int rc = QUO_ERR;
    /* my node (smp) rank */
    int my_smp_rank = 0, nsmp_ranks = 0;
    /* holds k set intersection info */
    int *k_set_intersection = NULL, k_set_intersection_len = 0;

    if (!q || !out_selected || max_qids_per_res_type <= 0) {
        return QUO_ERR_INVLD_ARG;
    }
    QUO_NO_INIT_ACTION(q);
    *out_selected = 0; /* set default */
    /* get total number of processes that share a node with me (includes me). */
    nsmp_ranks = q->nqid;
    /* what is my node rank? */
    my_smp_rank = q->qid;
    /* figure out how many target things are on the system. */
    if (QUO_SUCCESS != (rc = QUO_nobjs_by_type(q, distrib_over_this,
                                               &nres))) {
        return rc;
    }
    /* if there are no resources, then return not found */
    if (0 == nres) return QUO_ERR_NOT_FOUND;
    /* Populate arrays with data required to perform the intersection
     * calculation. */
    if (QUO_SUCCESS != (rc = get_qids_in_target_type(q, distrib_over_this, nres,
                                                     &nranks_in_res,
                                                     &rank_ids_in_res))) {
            QUO_ERR_MSGRC("get_qids_in_target_type", rc);
            goto out;
    }
    /* calculate the k set intersection of ranks on resources. the returned
     * array will be the set of ranks that currently share a particular
     * resource. */
    rc = quo_set_get_k_set_intersection(nres, nranks_in_res,
                                        rank_ids_in_res,
                                        &k_set_intersection,
                                        &k_set_intersection_len);
    if (QUO_SUCCESS != rc) goto out;
    /* ////////////////////////////////////////////////////////////////////// */
    /* distribute workers over target resources. */
    /* ////////////////////////////////////////////////////////////////////// */

    /* !!! remember: always maintain "max workers per resource" invariant !!! */

    /* completely disjoint sets, so making a local decision is easy */
    if (0 == k_set_intersection_len) {
        for (int rid = 0; rid < nres; ++rid) {
            /* if already a member, stop search */
            if (1 == *out_selected) break;
            for (int rank = 0; rank < nranks_in_res[rid]; ++rank) {
                /* if my current cpuset covers the resource in question and
                 * someone won't be assigned to that particular resource */
                if (my_smp_rank == rank_ids_in_res[rid][rank] &&
                    rank < max_qids_per_res_type) {
                    *out_selected = 1;
                }
            }
        }
    }
    /* all processes overlap - really no hope of doing anything sane. we
     * typically see this in the "no one is bound case." */
    else if (nsmp_ranks == k_set_intersection_len) {
        if (my_smp_rank < max_qids_per_res_type * nres) *out_selected = 1;
    }
    /* only a few ranks share a resource. i don't know if this case will ever
     * happen in practice, but i've seen stranger things... in any case, favor
     * unshared resources. */
    else {
        /* construct a "hash table" large enough to accommodate all possible
         * values up to nnoderanks - 1. note: these arrays are typically small,
         * so who cares. if this ever changes, then update the code to use a
         * proper hash table. */
        int *big_htab = NULL, rmapped = 0;
        size_t bhts = nsmp_ranks * sizeof(*big_htab);
        if (NULL == (big_htab = malloc(bhts))) {
            QUO_OOR_COMPLAIN();
            rc = QUO_ERR_OOR;
            goto out;
        }
        /* -1 = spot not taken */
        (void)memset(big_htab, -1, bhts);
        /* populate the hash table */
        for (int i = 0; i < k_set_intersection_len; ++i) {
            big_htab[k_set_intersection[i]] = k_set_intersection[i];
        }
        /* first only consider ranks that aren't sharing resources */
        for (int rid = 0; rid < nres; ++rid) {
            /* if already a member, stop search */
            if (1 == *out_selected) break;
            rmapped = 0;
            for (int rank = 0; rank < nranks_in_res[rid]; ++rank) {
                /* this thing is shared - skip */
                if (-1 != big_htab[rank_ids_in_res[rid][rank]]) continue;
                /* if my current cpuset covers the resource in question */
                if (my_smp_rank == rank_ids_in_res[rid][rank] &&
                    rmapped < max_qids_per_res_type) {
                        *out_selected = 1;
                        break;
                }
                ++rmapped;
            }
        }
        if (big_htab) free(big_htab);
    }
out:
    /* the resources returned by get_qids_in_target_type must be freed by us */
    if (rank_ids_in_res) {
        for (int i = 0; i < nres; ++i) {
            if (rank_ids_in_res[i]) free(rank_ids_in_res[i]);
        }
        free(rank_ids_in_res);
    }
    if (nranks_in_res) free(nranks_in_res);
    if (k_set_intersection) free(k_set_intersection);

    return rc;
}
