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
 * @file quof.c Fortran interface helpers.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "quo.h"
#include "quo-private.h"
#include "quo-mpi.h"

/**
 * Simply a wrapper for our Fortran interface to C interface. No need to expose
 * in quo.h header at this point, since it is only used by our Fortran module.
 */
int
QUO_create_f2c(QUO_t **q,
               MPI_Fint comm)
{
    MPI_Comm c_comm = MPI_Comm_f2c(comm);
    //
    return QUO_create(q, c_comm);
}

/**
 * Simply a wrapper for our Fortran interface to C interface. No need to expose
 * in quo.h header at this point, since it is only used by our Fortran module.
 */
int
QUO_get_mpi_comm_by_type_f2c(QUO_t *q,
                             QUO_obj_type_t target_type,
                             MPI_Fint *out_comm)
{
    MPI_Comm c_comm;
    int rc = QUO_get_mpi_comm_by_type(q, target_type, &c_comm);
    *out_comm = MPI_Comm_c2f(c_comm);

    return rc;
}

/**
 * Used to free up allocated memory on the Fortran side - don't include in
 * quo.h.
 *
 * @param[in] p Pointer to allocated memory.
 * */
int
QUO_ptr_free(void *p)
{
    if (p) free(p);
    return QUO_SUCCESS;
}
