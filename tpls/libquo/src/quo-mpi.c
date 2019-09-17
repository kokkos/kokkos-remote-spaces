/*
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

/**
 * @file quo-mpi.c
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "quo-mpi.h"
#include "quo-sm.h"
#include "quo-utils.h"

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
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif
#ifdef HAVE_TIME_H
#include <time.h>
#endif

#include "mpi.h"

/* don't forget that the upper layer will make sure that all the right stuff
 * will be called in the right order, so we don't have to be so careful
 * about checking if everything has been setup before continuing with the
 * operation. */

/** Pthread-based inter-process quiescence structure that is embedded in a
 * shared-memory segment (one per node per context). */
typedef struct quo_shmem_barrier_segment_t {
    /** The barrier structure. */
    pthread_barrier_t barrier;
} quo_shmem_barrier_segment_t;

/**
 * Maintains pid to smprank mapping.
 */
typedef struct pid_smprank_map_t {
    /** PID */
    long pid;
    /** Node-local ID. */
    int smprank;
} pid_smprank_map_t;

/* ////////////////////////////////////////////////////////////////////////// */
struct quo_mpi_t {
    /** Whether or not MPI is initialized. */
    int mpi_inited;
    /** My host's name. */
    char hostname[MPI_MAX_PROCESSOR_NAME];
    /**
     * Communication channel for libquo MPI bits.
     * A dup of initializing comm.
     */
    MPI_Comm commchan;
    /** Node communicator. */
    MPI_Comm smpcomm;
    /** Number of nodes in the current job. */
    int nnodes;
    /** My rank in MPI_COMM_WORLD. */
    int rank;
    /** Number of ranks in MPI_COMM_WORLD. */
    int nranks;
    /** My smp (node) rank. */
    int smprank;
    /** Number of ranks that share a node with me - |pid_smprank_map|. */
    int nsmpranks;
    /** PID to smprank map for all ranks that share a node with me. */
    pid_smprank_map_t *pid_smprank_map;
    /** Array of comm world ranks that share a node with me (includes me). */
    int *node_ranks;
    /** Shared-memory barrier segment path. */
    char *bseg_path;
    /** Base address of the shared memory segment used for our barrier. */
    quo_shmem_barrier_segment_t *bsegp;
    /** Shared memory instance for node-local barrier. */
    quo_sm_t *barrier_sm;
};

/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* private utility routines */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */

/* ////////////////////////////////////////////////////////////////////////// */
static int
cmp_uli(const void *p1,
        const void *p2)
{
    return (*(unsigned long int *)p1 - *(unsigned long int *)p2);
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
get_netnum(const char *hstn,
           unsigned long int *net_num)
{
    struct hostent *host = NULL;

    if (!hstn || !net_num) return QUO_ERR_INVLD_ARG;
    if (NULL == (host = gethostbyname(hstn))) {
        fprintf(stderr, QUO_ERR_PREFIX"%s failed. Cannot continue.\n",
                "gethostbyname");
        return QUO_ERR_SYS;
    }
    /* htonl used here because nodes could be different architectures */
    *net_num = htonl(inet_network(inet_ntoa(*(struct in_addr *)host->h_addr)));
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
get_my_color(unsigned long int *net_nums,
             int net_num_len,
             unsigned long int my_net_num,
             int *my_color)
{
    int i = 0, node_i = 0;
    unsigned long int prev_num;

    if (!net_nums || !my_color) return QUO_ERR_INVLD_ARG;
    qsort(net_nums, (size_t)net_num_len, sizeof(unsigned long int), cmp_uli);
    prev_num = net_nums[0];
    while (i < net_num_len && prev_num != my_net_num) {
        while (net_nums[i] == prev_num) {
            ++i;
        }
        ++node_i;
        prev_num = net_nums[i];
    }
    *my_color = node_i;
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
smprank_setup(quo_mpi_t *mpi)
{
    int rc = QUO_ERR, mycolor = 0, nnode_contrib = 0;
    unsigned long int my_netnum = 0, *netnums = NULL;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (NULL == (netnums = calloc(mpi->nranks, sizeof(*netnums)))) {
        QUO_OOR_COMPLAIN();
        return QUO_ERR_OOR;
    }
    if (QUO_SUCCESS != (rc = get_netnum(mpi->hostname, &my_netnum))) {
        /* rc set */
        goto out;
    }
    /* get everyone else's netnum */
    if (MPI_SUCCESS != (rc = MPI_Allgather(&my_netnum, 1, MPI_UNSIGNED_LONG,
                                           netnums, 1, MPI_UNSIGNED_LONG,
                                           mpi->commchan))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (QUO_SUCCESS != (rc = get_my_color(netnums, mpi->nranks, my_netnum,
                                          &mycolor))) {
        goto out;
    }
    /* split into local node groups */
    if (MPI_SUCCESS != (rc = MPI_Comm_split(mpi->commchan, mycolor, mpi->rank,
                                            &(mpi->smpcomm)))) {
        goto out;
    }
    /* get basic smpcomm info */
    if (MPI_SUCCESS != MPI_Comm_size(mpi->smpcomm, &(mpi->nsmpranks))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Comm_rank(mpi->smpcomm, &(mpi->smprank))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* calculate how many nodes are in our allocation */
    nnode_contrib = (0 == mpi->smprank) ? 1 : 0;
    if (MPI_SUCCESS != MPI_Allreduce(&nnode_contrib, &mpi->nnodes, 1, MPI_INT,
                                     MPI_SUM, mpi->commchan)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
out:
    if (netnums) free(netnums);
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
/** communication channel setup used for quo communication. */
static int
commchan_setup(quo_mpi_t *mpi,
               MPI_Comm comm)
{
    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (MPI_SUCCESS != MPI_Comm_dup(comm, &(mpi->commchan))) {
        return QUO_ERR_MPI;
    }
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
init_setup(quo_mpi_t *mpi,
           MPI_Comm comm)
{
    int rc = QUO_ERR, hostname_len = 0;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (QUO_SUCCESS != (rc = commchan_setup(mpi, comm))) goto out;
    /* gather some basic info that we need */
    if (MPI_SUCCESS != MPI_Comm_size(mpi->commchan, &(mpi->nranks))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Comm_rank(mpi->commchan, &(mpi->rank))) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* get my host's name */
    memset(mpi->hostname, 0, sizeof(mpi->hostname));
    if (MPI_SUCCESS != MPI_Get_processor_name(mpi->hostname, &hostname_len)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
out:
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
/**
 * pid_smprank_map allocation, setup, and exchange.
 */
static int
pid_smprank_xchange(quo_mpi_t *mpi)
{
    int rc = QUO_SUCCESS;
    pid_smprank_map_t my_info;
    my_info.pid = (long)getpid();
    my_info.smprank = mpi->smprank;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (NULL == (mpi->pid_smprank_map = calloc(mpi->nsmpranks,
                                               sizeof(pid_smprank_map_t)))) {
        QUO_OOR_COMPLAIN();
        return QUO_ERR_OOR;
    }
    /* ////////////////////////////////////////////////////////////////////// */
    /* if you update pid_smprank_map, then update this code                   */
    /* ////////////////////////////////////////////////////////////////////// */
    /* number of items in the struct */
    int nitems = 2;
    int block_lens[2] = {1, 1};
    /* array of base mpi types that the struct is made up of */
    MPI_Datatype types[2] = {MPI_LONG, MPI_INT};
    /* the mpi datatype that we are creating */
    MPI_Datatype pid_smprank_type;
    /* type offsets */
    MPI_Aint offsets[2];
    offsets[0] = offsetof(pid_smprank_map_t, pid);
    offsets[1] = offsetof(pid_smprank_map_t, smprank);
    /* create the thing */
    if (MPI_SUCCESS != MPI_Type_create_struct(nitems,
                                              block_lens,
                                              offsets,
                                              types,
                                              &pid_smprank_type)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    if (MPI_SUCCESS != MPI_Type_commit(&pid_smprank_type)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* now exchange the data between all ranks on the node (via smpcomm) */
    if (MPI_SUCCESS != MPI_Allgather(&my_info, 1, pid_smprank_type,
                                     mpi->pid_smprank_map, 1, pid_smprank_type,
                                     mpi->smpcomm)) {
        return QUO_ERR_MPI;
    }
out:
    /* error path */
    if (QUO_SUCCESS != rc) {
        if (mpi->pid_smprank_map) {
            free(mpi->pid_smprank_map);
            mpi->pid_smprank_map = NULL;
        }
    }
    /* no longer needed */
    if (MPI_SUCCESS != MPI_Type_free(&pid_smprank_type)) rc = QUO_ERR_MPI;
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
node_rank_xchange(quo_mpi_t *mpi)
{
    int rc = QUO_SUCCESS;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    mpi->node_ranks = calloc(mpi->nsmpranks, sizeof(int));
    if (!mpi->node_ranks) return QUO_ERR_OOR;
    if (MPI_SUCCESS != MPI_Allgather(&(mpi->rank), 1, MPI_INT,
                                     mpi->node_ranks, 1, MPI_INT,
                                     mpi->smpcomm)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
out:
    if (QUO_SUCCESS != rc) {
        if (mpi->node_ranks) {
            free(mpi->node_ranks);
            mpi->node_ranks = NULL;
        }
    }
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
get_barrier_segment_name(quo_mpi_t *mpi,
                         const char *module_name,
                         char **segname)
{
    int rc = QUO_SUCCESS, err = 0;
    bool tmpdir_usable = false;
    char *usern = NULL, *tmpdir = NULL;
    int randn = 0;
    int my_pid = (int)getpid();

    randn = rand();

    if (!mpi || !module_name || !segname) return QUO_ERR_INVLD_ARG;
    /* get base dir */
    if (QUO_SUCCESS != (rc = quo_utils_tmpdir(&tmpdir))) goto out;
    /* get user name */
    if (QUO_SUCCESS != (rc = quo_utils_whoami(&usern))) goto out;
    /* make sure that the provided base is usable */
    if (QUO_SUCCESS != (rc = quo_utils_path_usable(tmpdir, &tmpdir_usable,
                                                   &err))) goto out;
    if (!tmpdir_usable) {
        fprintf(stderr, QUO_ERR_PREFIX"cannot use: %s (errno: %d (%s.))\n",
                tmpdir, err, strerror(err));
        rc = QUO_ERR_INVLD_ARG;
        goto out;
    }
    /* all is well, so build the file name - caller must free this */
    if (-1 == asprintf(segname, "%s/%s-%s-%s-%d-%d.%s",
                       tmpdir, PACKAGE, mpi->hostname, usern,
                       my_pid, randn, module_name)) {
        rc = QUO_ERR_OOR;
        goto out;
    }
out:
    if (tmpdir) free(tmpdir);
    if (usern) free(usern);
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
ptmc_init(quo_mpi_t *mpi)
{
    int rc = QUO_ERR_SYS;
    pthread_barrierattr_t battr;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (0 != pthread_barrierattr_init(&battr)) goto out;
    if (0 != pthread_barrierattr_setpshared(&battr, PTHREAD_PROCESS_SHARED)) {
        goto out;
    }
    if (0 != pthread_barrier_init(&(mpi->bsegp->barrier), &battr,
                                  (unsigned)mpi->nsmpranks)) {
        goto out;
    }
    if (0 != pthread_barrierattr_destroy(&battr)) goto out;
    /* all is well */
    rc = QUO_SUCCESS;
out:
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
bseg_create(quo_mpi_t *mpi)
{
    int rc = QUO_SUCCESS;
    char *badfunc = NULL;

    if (!mpi) return QUO_ERR_INVLD_ARG;

    if (QUO_SUCCESS != (rc =
            quo_sm_segment_create(mpi->barrier_sm,
                                  mpi->bseg_path,
                                  sizeof(quo_shmem_barrier_segment_t)))) {
        badfunc = "quo_sm_segment_create";
        goto out;
    }
    mpi->bsegp = quo_sm_get_basep(mpi->barrier_sm);
    /*setup mutex, condition, and barrier counter */
    if (QUO_SUCCESS != (rc = ptmc_init(mpi))) goto out;
out:
    if (badfunc) {
        fprintf(stderr, QUO_ERR_PREFIX"%s failure: rc=%d\n", badfunc, rc);
    }
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
bseg_attach(quo_mpi_t *mpi)
{
    int rc = QUO_SUCCESS;
    char *badfunc = NULL;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (QUO_SUCCESS != (rc =
            quo_sm_segment_attach(mpi->barrier_sm,
                                  mpi->bseg_path,
                                  sizeof(quo_shmem_barrier_segment_t)))) {
        badfunc = "quo_sm_segment_attach";
        goto out;
    }
    mpi->bsegp = quo_sm_get_basep(mpi->barrier_sm);
out:
    if (badfunc) {
        fprintf(stderr, QUO_ERR_PREFIX"%s failure: rc=%d\n", badfunc, rc);
    }
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_xchange_uniq_path(quo_mpi_t *mpi,
                          const char *module_name,
                          char **result)
{
    int rc = QUO_SUCCESS, plen = 0;

    if (!mpi || !module_name || !result) return QUO_ERR_INVLD_ARG;
    //
    if (0 == mpi->smprank) {
        rc = get_barrier_segment_name(mpi, module_name, result);
        if (QUO_SUCCESS != rc) {
            plen = -rc; /* indicates an error occurred */
        }
        else {
            plen = strlen(*result) + 1;
        }
        if (MPI_SUCCESS != MPI_Bcast(&plen, 1, MPI_INT, 0, mpi->smpcomm)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
        /* an error occurred, so just bail */
        if (QUO_SUCCESS != rc) goto out;
        if (MPI_SUCCESS != MPI_Bcast(*result, plen,
                                     MPI_CHAR, 0, mpi->smpcomm)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
    }
    else {
        if (MPI_SUCCESS != MPI_Bcast(&plen, 1, MPI_INT, 0, mpi->smpcomm)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
        /* something bad happened during setup, so just bail */
        if (plen < 0) {
            rc = -plen;
            goto out;
        }
        /* we are good, recv the path */
        if (NULL == (*result = calloc(plen, sizeof(char)))) {
            QUO_OOR_COMPLAIN();
            rc = QUO_ERR_OOR;
            goto out;
        }
        if (MPI_SUCCESS != MPI_Bcast(*result, plen,
                                     MPI_CHAR, 0, mpi->smpcomm)) {
            rc = QUO_ERR_MPI;
            goto out;
        }
    }
out:
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
static int
sm_setup(quo_mpi_t *mpi)
{
    int rc = QUO_SUCCESS;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    /* first exchange segment path information */
    if (QUO_SUCCESS != (rc =
            quo_mpi_xchange_uniq_path(mpi, "bseg", &mpi->bseg_path))) {
        goto out;
    }
    /* node rank 0 sets up the segment */
    if (0 == mpi->smprank) {
        if (QUO_SUCCESS != (rc = bseg_create(mpi))) goto out;
    }
    /* sync */
    if (MPI_SUCCESS != MPI_Barrier(mpi->smpcomm)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* everyone else attach to the shared memory segment */
    if (0 != mpi->smprank) {
        if (QUO_SUCCESS != (rc = bseg_attach(mpi))) goto out;
    }
    /* sync */
    if (MPI_SUCCESS != MPI_Barrier(mpi->smpcomm)) {
        rc = QUO_ERR_MPI;
        goto out;
    }
    /* cleanup */
    if (0 == mpi->smprank) {
        (void)quo_sm_unlink(mpi->barrier_sm);
    }
out:
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_smprank2pid(quo_mpi_t *mpi,
                    int smprank,
                    pid_t *out_pid)
{
    if (!mpi || !out_pid) return QUO_ERR_INVLD_ARG;

    /* MPI_Allgather guarantees rank ordering. Since we used the SMP
     * communicator for this exchange, the ith item will always correspond to
     * the SMP rank i. */
    *out_pid = (pid_t)mpi->pid_smprank_map[smprank].pid;

    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_construct(quo_mpi_t **nmpi)
{
    int rc = QUO_SUCCESS;
    quo_mpi_t *m = NULL;

    if (!nmpi) return QUO_ERR_INVLD_ARG;
    if (NULL == (m = calloc(1, sizeof(*m)))) {
        QUO_OOR_COMPLAIN();
        return QUO_ERR_OOR;
    }
    if (QUO_SUCCESS != (rc = quo_sm_construct(&(m->barrier_sm)))) {
        fprintf(stderr, QUO_ERR_PREFIX"%s failed. Cannot continue.\n",
                "quo_sm_construct");
        goto out;
    }
    /* Set new seed for rand() things. */
    srand((unsigned int)time(NULL));

    *nmpi = m;
out:
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_init(quo_mpi_t *mpi,
             MPI_Comm comm)
{
    int rc = QUO_ERR;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (MPI_SUCCESS != MPI_Initialized(&(mpi->mpi_inited))) return QUO_ERR_MPI;
    /* if mpi isn't initialized, then we can't continue */
    if (!mpi->mpi_inited) {
        fprintf(stderr, QUO_ERR_PREFIX"MPI has not been initialized and %s "
                "uses MPI. Cannot continue.\n", PACKAGE);
        rc = QUO_ERR_MPI;
        goto err;
    }
    /* if we are here, then mpi is initialized */
    mpi->mpi_inited = 1;
    /* first perform basic initialization */
    if (QUO_SUCCESS != (rc = init_setup(mpi, comm))) goto err;
    /* setup node rank info */
    if (QUO_SUCCESS != (rc = smprank_setup(mpi))) goto err;
    /* mpi is setup and we know about our node neighbors and all the jive, so
     * setup and exchange node pids and node ranks. */
    if (QUO_SUCCESS != (rc = pid_smprank_xchange(mpi))) goto err;
    /* now cache the initializing comm ranks that are the node with me */
    if (QUO_SUCCESS != (rc = node_rank_xchange(mpi))) goto err;
    /* now setup shared memory stuff for our barrier */
    if (QUO_SUCCESS != (rc = sm_setup(mpi))) goto err;
    return QUO_SUCCESS;
err:
    quo_mpi_destruct(mpi);
    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_destruct(quo_mpi_t *mpi)
{
    int nerrs = 0;

    if (!mpi) return QUO_ERR_INVLD_ARG;
    if (mpi->mpi_inited) {
        if (MPI_SUCCESS != MPI_Comm_free(&(mpi->commchan))) nerrs++;
        if (MPI_SUCCESS != MPI_Comm_free(&(mpi->smpcomm))) nerrs++;
    }
    if (mpi->pid_smprank_map) {
        free(mpi->pid_smprank_map);
        mpi->pid_smprank_map = NULL;
    }
    if (mpi->node_ranks) {
        free(mpi->node_ranks);
        mpi->node_ranks = NULL;
    }
    if (mpi->bseg_path) {
        free(mpi->bseg_path);
        mpi->bseg_path = NULL;
    }
    if (QUO_SUCCESS != quo_sm_destruct(mpi->barrier_sm)) nerrs++;

    free(mpi);

    return 0 == nerrs ? QUO_SUCCESS : QUO_ERR;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_nnodes(const quo_mpi_t *mpi,
               int *nnodes)
{
    if (!mpi || !nnodes) return QUO_ERR_INVLD_ARG;
    *nnodes = mpi->nnodes;
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_nnoderanks(const quo_mpi_t *mpi,
                   int *nnoderanks)
{
    if (!mpi || !nnoderanks) return QUO_ERR_INVLD_ARG;
    *nnoderanks = mpi->nsmpranks;
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
/* SKG - i think we can get this info from MPI-3. so in the MPI-3 case, just use
 * the mpi interface to get this info -- should be much faster and more
 * scalable. */
int
quo_mpi_noderank(const quo_mpi_t *mpi,
                 int *noderank)
{
    if (!mpi || !noderank) return QUO_ERR_INVLD_ARG;
    *noderank = mpi->smprank;
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_ranks_on_node(const quo_mpi_t *mpi,
                      int *out_nranks,
                      int **out_ranks)
{
    int *ta = NULL;
    if (!mpi || !out_nranks || !out_ranks) return QUO_ERR_INVLD_ARG;
    *out_nranks = mpi->nsmpranks; *out_ranks = NULL;
    if (NULL == (ta = calloc(mpi->nsmpranks, sizeof(int)))) return QUO_ERR_OOR;
    (void)memmove(ta, mpi->node_ranks, mpi->nsmpranks * sizeof(int));
    *out_nranks = mpi->nsmpranks;
    *out_ranks = ta;
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_sm_barrier(const quo_mpi_t *mpi)
{
    int rc = 0;
    if (!mpi) return QUO_ERR_INVLD_ARG;
    rc = pthread_barrier_wait(&(mpi->bsegp->barrier));
    if (PTHREAD_BARRIER_SERIAL_THREAD != rc && 0 != rc) return QUO_ERR_SYS;
    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_get_comm_by_type(const quo_mpi_t *mpi,
                         QUO_obj_type_t target_type,
                         MPI_Comm *out_comm)
{
    if (!mpi || !out_comm) return QUO_ERR_INVLD_ARG;

    switch (target_type) {
        case QUO_OBJ_MACHINE:
        {
            /* this case is easy. just return a dup of the smp communicator that
             * we already maintain internally. */
            if (MPI_SUCCESS != MPI_Comm_dup(mpi->smpcomm, out_comm)) {
                return QUO_ERR_MPI;
            }
            break;
        }
        /* TODO add support for other obj types */
        default:
            return QUO_ERR_NOT_SUPPORTED;
    }

    return QUO_SUCCESS;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_bcast(void *buffer,
              int count,
              MPI_Datatype datatype,
              int root,
              MPI_Comm comm)
{
    int rc = QUO_SUCCESS;

    if (!buffer) return QUO_ERR_INVLD_ARG;

    if (MPI_SUCCESS != MPI_Bcast(buffer, count, datatype, root, comm)) {
        rc = QUO_ERR_MPI;
    }

    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_allgather(const void *sendbuf,
                  int sendcount,
                  MPI_Datatype sendtype,
                  void *recvbuf,
                  int recvcount,
                  MPI_Datatype recvtype,
                  MPI_Comm comm)
{
    int rc = QUO_SUCCESS;

    if (!sendbuf || !recvbuf) return QUO_ERR_INVLD_ARG;

    if (MPI_SUCCESS != MPI_Allgather(sendbuf, sendcount, sendtype,
                                     recvbuf, recvcount, recvtype, comm)) {
        rc = QUO_ERR_MPI;
    }

    return rc;
}

/* ////////////////////////////////////////////////////////////////////////// */
int
quo_mpi_get_node_comm(quo_mpi_t *mpi,
                      MPI_Comm *comm)
{
    if (!mpi || !comm) return QUO_ERR_INVLD_ARG;

    *comm = mpi->smpcomm;

    return QUO_SUCCESS;
}
