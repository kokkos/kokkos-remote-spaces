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
 * @file quo-mpi.h
 */

#ifndef QUO_MPI_H_INCLUDED
#define QUO_MPI_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "quo-private.h"
#include "quo.h"

#include "mpi.h"

int
quo_mpi_construct(quo_mpi_t **nmpi);

int
quo_mpi_init(quo_mpi_t *nmpi,
             MPI_Comm comm);

int
quo_mpi_destruct(quo_mpi_t *nmpi);

int
quo_mpi_nnodes(const quo_mpi_t *mpi,
               int *nnodes);

int
quo_mpi_nnoderanks(const quo_mpi_t *mpi,
                   int *nnoderanks);

int
quo_mpi_noderank(const quo_mpi_t *mpi,
                 int *noderank);

int
quo_mpi_smprank2pid(quo_mpi_t *mpi,
                    int smprank,
                    pid_t *out_pid);

int
quo_mpi_ranks_on_node(const quo_mpi_t *mpi,
                      int *out_nranks,
                      int **out_ranks);

int
quo_mpi_sm_barrier(const quo_mpi_t *mpi);

int
quo_mpi_xchange_uniq_path(quo_mpi_t *mpi,
                          const char *module_name,
                          char **result);

int
quo_mpi_get_node_comm(quo_mpi_t *mpi,
                      MPI_Comm *comm);

int
quo_mpi_bcast(void *buffer,
              int count,
              MPI_Datatype datatype,
              int root,
              MPI_Comm comm);

int
quo_mpi_allgather(const void *sendbuf,
                  int sendcount,
                  MPI_Datatype sendtype,
                  void *recvbuf,
                  int recvcount,
                  MPI_Datatype recvtype,
                  MPI_Comm comm);
int
quo_mpi_get_comm_by_type(const quo_mpi_t *mpi,
                         QUO_obj_type_t target_type,
                         MPI_Comm *out_comm);
#endif
