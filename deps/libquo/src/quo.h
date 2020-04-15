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
 * @file quo.h
 */

/* I do a pretty terrible job explaining the interface. play around with the
 * demo codes; they are simple and pretty clearly illustrate how to use QUO. */

#ifndef QUO_H_INCLUDED
#define QUO_H_INCLUDED

/* For MPI_Comm type */
#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Convenience definition (in case you need this). */
#define LIBQUO 1

/** Opaque QUO context. */
struct QUO_t;
/** Convenience typedef. */
typedef struct QUO_t QUO_t;
/** External QUO context type. */
typedef QUO_t * QUO_context;

/**
 * QUO return codes:
 * - fatal = libquo can no longer function.
 * - not fatal = libquo can continue functioning, but an error occurred.
 */
enum {
    /** Success. */
    QUO_SUCCESS = 0,
    /** Success, but already done. */
    QUO_SUCCESS_ALREADY_DONE,
    /** General error -- fatal. */
    QUO_ERR,
    /** System error -- fatal. */
    QUO_ERR_SYS,
    /** Out of resources error  -- fatal. */
    QUO_ERR_OOR,
    /** Invalid argument provided to library -- usually fatal. */
    QUO_ERR_INVLD_ARG,
    /** Library call before QUO_init was called -- improper use of library. */
    QUO_ERR_CALL_BEFORE_INIT,
    /** Topology error -- fatal. */
    QUO_ERR_TOPO,
    /** MPI error -- fatal. */
    QUO_ERR_MPI,
    /** Action not supported -- usually not fatal. */
    QUO_ERR_NOT_SUPPORTED,
    /** Pop error -- not fatal, but usually indicates improper use. */
    QUO_ERR_POP,
    /** The thing that you were looking for wasn't found -- not fatal. */
    QUO_ERR_NOT_FOUND
};

/** Hardware resource types. */
typedef enum {
    /** The machine. */
    QUO_OBJ_MACHINE = 0,
    /** NUMA node. */
    QUO_OBJ_NUMANODE,
    /** Socket. */
    QUO_OBJ_SOCKET,
    /** Core. */
    QUO_OBJ_CORE,
    /** Processing unit (e.g. hardware thread). */
    QUO_OBJ_PU
} QUO_obj_type_t;

/** Push policies that influence QUO_bind_push behavior. */
typedef enum {
    /** Push the exact binding policy that was provided. */
    QUO_BIND_PUSH_PROVIDED = 0,
    /** Push to the enclosing QUO_obj_type_t provided. */
    QUO_BIND_PUSH_OBJ
} QUO_bind_push_policy_t;

/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* QUO API */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */

/**
 * Version query routine.
 *
 * @param[out] version Major version.
 *
 * @param[out] subversion Subversion.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \note
 * This routine can be called before QUO_init.
 */
int
QUO_version(int *version,
            int *subversion);

/**
 * Context handle construction and initialization routine.
 *
 * @param[in] comm Initializing MPI communicator.
 * @param[out] q Reference to a new QUO_context.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \note
 * This is typically the first "real" call into the library. A relatively
 * expensive routine that must be called AFTER MPI_Init. Call QUO_free to free
 * returned resources.
 *
 * \code{.c}
 * QUO_context quo = NULL;
 * if (QUO_SUCCESS != QUO_create(&quo, MPI_COMM_WORLD)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_create(QUO_context *q,
           MPI_Comm comm);

/**
 * Context handle destruction routine.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \note
 * This is typically the last "real" call into the library.  A relatively
 * inexpensive routine that must be called BEFORE MPI_Finalize.  Once a call to
 * this routine is made, it is an error to use any libquo services associated
 * with the freed libquo context from any other participating process.
 *
 * \code{.c}
 * if (QUO_SUCCESS != QUO_free(quo)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_free(QUO_context q);

/**
 * Context query routine that returns the total number of hardware
 * resource objects that are on the caller's system.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] target_type Hardware object type that is being queried.
 *
 * @param[out] out_nobjs Total number of hardware object types found on the
 *             system.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * int nsockets = 0;
 * if (QUO_SUCCESS != QUO_nobjs_by_type(q, QUO_OBJ_SOCKET, &nsockets)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_nobjs_by_type(QUO_context q,
                  QUO_obj_type_t target_type,
                  int *out_nobjs);

/**
 * Context query routine that returns the total number of hardware
 * resource objects that are in another hardware resource (e.g. cores in a
 * socket).
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] in_type Container hardware object type.
 *
 * @param[in] in_type_index in_type's ID (base 0).
 *
 * @param[in] type Target hardware object found in in_type[in_type_index].
 *
 * @param[out] out_result Total number of hardware object types found by the query.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * int ncores_in_first_socket = 0;
 * if (QUO_SUCCESS != QUO_nobjs_in_type_by_type(q, QUO_OBJ_SOCKET, 0
 *                                              QUO_OBJ_CORE,
 *                                              &ncores_in_first_socket)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_nobjs_in_type_by_type(QUO_context q,
                          QUO_obj_type_t in_type,
                          int in_type_index,
                          QUO_obj_type_t type,
                          int *out_result);

/**
 * Context handle query routine that returns whether or not my current
 * binding policy falls within a particular system hardware resource (is
 * enclosed).
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] type Hardware object type.
 *
 * @param[in] in_type_index type's ID (base 0).
 *
 * @param[out] out_result Flag indicating whether or not my current binding policy
 *             falls within type[in_type_index].
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * int cur_bind_covers_sock3 = 0;
 * if (QUO_SUCCESS != QUO_cpuset_in_type(q, QUO_OBJ_SOCKET, 2
 *                                       &cur_bind_enclosed_in_sock3)) {
 *     // error handling //
 * }
 * if (cur_bind_enclosed_in_sock3) {
 *     // do stuff //
 * }
 * \endcode
 */
int
QUO_cpuset_in_type(QUO_context q,
                   QUO_obj_type_t type,
                   int in_type_index,
                   int *out_result);

/**
 * Similar to QUO_cpuset_in_type, but returns the "SMP_COMM_WORLD" QUO IDs that
 * met the query criteria.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] type Hardware object type.
 *
 * @param[in] in_type_index type's ID (base 0).
 *
 * @param[out] out_nqids Total number of node (job) processes that satisfy the
 *             query criteria.
 *
 * @param[out] out_qids An array of "SMP_COMM_WORLD ranks" that met the query
 *             criteria. *out_qids must be freed by a call to free(3).
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * int nqids_enclosed_in_socket0 = 0;
 * int *qids_enclosed_in_socket0 = NULL;
 * if (QUO_SUCCESS != QUO_qids_in_type(q, QUO_OBJ_SOCKET, 0
 *                                     &nqids_enclosed_in_socket0,
 *                                     &qids_enclosed_in_socket0)) {
 *     // error handling //
 * }
 * free(qids_enclosed_in_socket0);
 * \endcode
 */
int
QUO_qids_in_type(QUO_context q,
                 QUO_obj_type_t type,
                 int in_type_index,
                 int *out_nqids,
                 int **out_qids);

/**
 * Query routine that returns the total number of NUMA nodes that are
 * present on the caller's system.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[out] out_nnumanodes Total number of NUMA nodes on the system.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * int nnumanodes = 0;
 * if (QUO_SUCCESS != QUO_nnumanodes(q, &nnumanodes)) {_
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_nnumanodes(QUO_context q,
               int *out_nnumanodes);

/**
 * Similar to QUO_nnumanodes, but returns the total number of sockets present on
 * the caller's system.
 */
int
QUO_nsockets(QUO_context q,
             int *out_nsockets);

/**
 * Similar to QUO_nnumanodes, but returns the total number of cores present on
 * the caller's system.
 */
int
QUO_ncores(QUO_context q,
           int *out_ncores);

/**
 * Similar to QUO_nnumanodes, but returns the total number of processing units
 * (PUs) (e.g., hardware threads) present on the caller's system.
 */
int
QUO_npus(QUO_context q,
         int *out_npus);

/**
 * Similar to QUO_nnumanodes, but returns the total number of compute nodes
 * (i.e., servers) in the current job.
 */
int
QUO_nnodes(QUO_context q,
           int *out_nodes);

/**
 * Similar to QUO_nnumanodes, but returns the total number of job processes that
 * are on the caller's compute node.
 *
 * \note
 * *out_nqids includes the caller. For example, if there are 3 MPI processes on
 * rank 0's (MPI_COMM_WORLD) node, then rank 0's call to this routine will
 * result in *out_nqids being set to 3.
 */
int
QUO_nqids(QUO_context q,
          int *out_nqids);

/**
 * Query routine that returns the caller's compute node QUO node ID.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[out] out_qid The caller's node ID, as assigned by libquo.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * NOTES: QIDs start at 0 and go to NNODERANKS - 1.
 *
 * \code{.c}
 * int mynodeqid = 0;
 * if (QUO_SUCCESS != QUO_id(q, &mynodeqid)) {_
 *     // error handling //
 * }
 * if (0 == mynodeqid) {
 *     // node id 0 do stuff //
 * }
 * \endcode
 */
int
QUO_id(QUO_context q,
       int *out_qid);

/**
 * Query routine that returns whether or not the caller is currently
 * "bound" to a CPU resource.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[out] bound Flag indicating whether or not the caller is currently bound.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \note
 * If the caller's current cpuset is equal to the widest available
 * cpuset, then the caller is not bound as far as libquo is concerned. For
 * example, if your system has only one core and the calling process is "bound"
 * to that one core, then as far as we are concerned, the caller is not bound.
 *
 * \code{.c}
 * int bound = 0;
 * if (QUO_SUCCESS != QUO_bound(q, &bound)) {_
 *     // error handling //
 * }
 * if (!bound) {
 *     // take action //
 * }
 * \endcode
 */

int
QUO_bound(QUO_context q,
          int *bound);

/**
 * Query routine that returns a string representation of the caller's
 * current binding policy (cpuset) in a hexadecimal format. @see CPUSET(7).
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[out] cbind_str The caller's current CPU binding policy in string form.
 *                       *cbind_str must be freed by call to free(3). (OUT)
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * char *cbindstr = NULL;
 * if (QUO_SUCCESS != QUO_stringify_cbind(q, &cbindstr)) {
 *     // error handling //
 * }
 * printf("%s\n", cbindstr);
 * free(cbindstr);
 * \endcode
 */
int
QUO_stringify_cbind(QUO_context q,
                    char **cbind_str);


/**
 * Routine that changes the caller's process binding policy. The policy
 * is maintained in the current context's stack.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] policy Policy that influence the behavior of this routine. If
 *                   QUO_BIND_PUSH_PROVIDED is provided, then the type and
 *                   obj_index are used as the new policy.  If QUO_BIND_PUSH_OBJ
 *                   is provided, then obj_index is ignored and the "closest"
 *                   type is used.
 *
 * @param[in] type The hardware resource to bind to.
 *
 * @param[in] obj_index When not ignored, type's index (base 0).
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \note
 * To revert to the previous binding policy call QUO_bind_pop.
 *
 * \code{.c}
 * // in this example we will bind to socket 0 //
 * if (QUO_SUCCESS != QUO_bind_push(q, QUO_BIND_PUSH_PROVIDED,
 *                                  QUO_OBJ_SOCKET, 0)) {
 *     // error handling //
 * }
 * // revert to previous process binding policy //
 * if (QUO_SUCCESS != QUO_bind_pop(q)) {
 *     // error handling //
 * }
 * // EXAMPLE 2
 * // in this example we will bind to the "closest" socket //
 * if (QUO_SUCCESS != QUO_bind_push(q, QUO_BIND_PUSH_OBJ,
 *                                  QUO_OBJ_SOCKET, -1)) {
 *     // error handling //
 * }
 * // revert to previous process binding policy //
 * if (QUO_SUCCESS != QUO_bind_pop(q)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_bind_push(QUO_context q,
              QUO_bind_push_policy_t policy,
              QUO_obj_type_t type,
              int obj_index);

/**
 * Routine that changes the caller's process binding policy by replacing
 * it with the policy at the top of the provided context's process bind stack.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * // in this example we will bind to socket 0 //
 * if (QUO_SUCCESS != QUO_bind_push(q, QUO_BIND_PUSH_PROVIDED,
 *                                  QUO_OBJ_SOCKET, 0)) {
 *     // error handling //
 * }
 * // revert to previous process binding policy //
 * if (QUO_SUCCESS != QUO_bind_pop(q)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_bind_pop(QUO_context q);

/**
 * Routine that acts as a compute node barrier. All context-initializing
 * processes on a node MUST call this in order for everyone to proceed past the
 * barrier. See demos for examples.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 *  // time for p1 to do some work with some of the ranks //
 *  if (working) {
 *      // *** do work *** //
 *      // signals completion //
 *      if (QUO_SUCCESS != QUO_barrier(q)) {
 *          // error handling //
 *      }
 *  } else {
 *      // non workers wait in a barrier //
 *      if (QUO_SUCCESS != QUO_barrier(q)) {
 *          // error handling //
 *      }
 *  }
 *  \endcode
 */
int
QUO_barrier(QUO_context q);

/**
 * Routine that helps evenly distribute processes across hardware
 * resources.  The total number of processes assigned to a particular resource
 * will not exceed max_qids_per_res_type.
 *
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] distrib_over_this The target hardware resource on which processes
 *                              will be evenly distributed.
 *
 * @param[in] max_qids_per_res_type The maximum number of processes that will be
 *                                  assigned to the provided resources. For
 *                                  example, if your system has two sockets and
 *                                  max_qids_per_res_type is 2, then a max of 4
 *                                  processes will be chosen (max 2 per socket).
 *                                  this routine doesn't modify the calling
 *                                  processes' affinities, but is used as a
 *                                  helper for evenly distributing processes over
 *                                  hardware resources given a global view of all
 *                                  the affinities within a job. i'm doing a
 *                                  terrible job explaining this, so look at the
 *                                  demos. Believe me, this routine is useful...
 *
 * @param[out] out_selected Flag indicating whether or not i was chosen in the
 *                          work distribution. 1 means I was chosen, 0
 *                          otherwise.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 *
 * \code{.c}
 * int res_assigned = 0;
 * if (QUO_SUCCESS != QUO_auto_distrib(q, QUO_OBJ_SOCKET,
 *                                     2, &res_assigned)) {
 *     // error handling //
 * }
 * \endcode
 */
int
QUO_auto_distrib(QUO_context q,
                 QUO_obj_type_t distrib_over_this,
                 int max_qids_per_res_type,
                 int *out_selected);

/**
 * @param[in] q Constructed and initialized QUO_context.
 *
 * @param[in] target_type Target hardware object type.
 *
 * @param[out] out_comm MPI_Comm_dup'd communicator containing processes that
 *                      match the target request. Returned resources must be
 *                      freed with a call to MPI_Comm_free.
 *
 * @retval QUO_SUCCESS if the operation completed successfully.
 */
int
QUO_get_mpi_comm_by_type(QUO_context q,
                         QUO_obj_type_t target_type,
                         MPI_Comm *out_comm);

/**
 * \note
 * Experimental.
 */
int
QUO_bind_threads(QUO_context q,
                 QUO_obj_type_t type,
                 int index);

#ifdef __cplusplus
}
#endif

#endif
