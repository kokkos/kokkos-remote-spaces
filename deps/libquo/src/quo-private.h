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
 * @file quo-private.h
 */

#ifndef QUO_PRIVATE_H_INCLUDED
#define QUO_PRIVATE_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif
#ifdef HAVE_STDBOOL_H
#include <stdbool.h>
#endif

/** Library version. */
#define QUO_VER    QUO_VERSION_CURRENT
#define QUO_SUBVER QUO_VERSION_REVISION

/* ////////////////////////////////////////////////////////////////////////// */
/* Convenience macros                                                         */
/* ////////////////////////////////////////////////////////////////////////// */

#define QUO_STRINGIFY(x) #x
#define QUO_TOSTRING(x)  QUO_STRINGIFY(x)

#define QUO_ERR_AT       __FILE__ ":"QUO_TOSTRING(__LINE__)""
#define QUO_ERR_PREFIX   "-["PACKAGE" ERROR @ "QUO_ERR_AT"]- "
#define QUO_WARN_PREFIX  "-["PACKAGE" WARNING @ "QUO_ERR_AT"]- "

/**
 * Convenience macro used to print out OOR messages.
 */
#define QUO_OOR_COMPLAIN()                                                     \
do {                                                                           \
    fprintf(stderr, QUO_ERR_PREFIX "out of resources\n");                      \
    fflush(stderr);                                                            \
} while (0)

/**
 * Convenience macro used to print out error messages.
 *
 * @param[in] whatstr Error message string.
 */
#define QUO_ERR_MSG(whatstr)                                                   \
do {                                                                           \
    fprintf(stderr, QUO_ERR_PREFIX"%s failed: %s.\n", __func__, (whatstr));    \
} while (0)

/**
 * Convenience macro used to print out error messages.
 *
 * @param[in] what The name of the function that returned an error code.
 *
 * @param[in] rc The error code.
 */
#define QUO_ERR_MSGRC(what, rc)                                                \
do {                                                                           \
    fprintf(stderr, QUO_ERR_PREFIX"%s failure: (rc: %d). "                     \
                "Cannot continue.\n", (what), (rc));                           \
} while (0)

/**
 * Convenience macro used to silence warnings about unused variables.
 *
 * @param[in] x Unused variable.
 */
#define QUO_UNUSED(x)                                                          \
do {                                                                           \
    (void)(x);                                                                 \
} while (0)

/**
 * Convenience macro used to print out error messages.
 *
 * @param[in] func The function name that caused this error.
 */
#define QUO_NO_INIT_MSG_EMIT(func)                                             \
do {                                                                           \
    fprintf(stderr, QUO_ERR_PREFIX"%s called before %s. Cannot continue.\n",   \
            (func), "QUO_create");                                             \
} while (0)

/**
 * Convenience macro used to check and handle context non-initialization.
 *
 * @param[in] qp QUO context pointer.
 */
#define QUO_NO_INIT_ACTION(qp)                                                 \
do {                                                                           \
    if (!(qp)->initialized) {                                                  \
        QUO_NO_INIT_MSG_EMIT(__func__);                                        \
        return QUO_ERR_CALL_BEFORE_INIT;                                       \
    }                                                                          \
} while (0)

/* ////////////////////////////////////////////////////////////////////////// */
/* Forward declarations. */
struct quo_hwloc_t;
typedef struct quo_hwloc_t quo_hwloc_t;

struct quo_mpi_t;
typedef struct quo_mpi_t quo_mpi_t;

/** QUO_t type definition. */
struct QUO_t {
    /** Whether or not a context has been initialized. */
    bool initialized;
    /** PID of initializer. */
    pid_t pid;
    /** Handle to hwloc instance. */
    quo_hwloc_t *hwloc;
    /** Handle to MPI instance. */
    quo_mpi_t *mpi;
    /* Information cache. */
    /** My unique QUO ID (node-local). */
    int qid;
    /** Number of processes that share a node with me. */
    int nqid;
};

#endif
