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
 * @file quo-hwloc.h
 */

#ifndef QUO_HWLOC_H_INCLUDED
#define QUO_HWLOC_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "quo.h"
#include "quo-private.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_STDBOOL_H
#include <stdbool.h>
#endif

#include "hwloc/include/hwloc.h"

typedef quo_internal_hwloc_const_cpuset_t quo_const_nodeset_t;

int
quo_hwloc_construct(quo_hwloc_t **nhwloc);

int
quo_hwloc_init(quo_hwloc_t *hwloc,
               quo_mpi_t *mpi);

int
quo_hwloc_destruct(quo_hwloc_t *nhwloc);

int
quo_hwloc_get_nobjs_by_type(const quo_hwloc_t *hwloc,
                            QUO_obj_type_t target_type,
                            int *out_nobjs);

int
quo_hwloc_get_nobjs_in_type_by_type(const quo_hwloc_t *hwloc,
                                    QUO_obj_type_t in_type,
                                    unsigned in_type_index,
                                    QUO_obj_type_t type,
                                    int *out_result);

int
quo_hwloc_is_in_cpuset_by_type_id(const quo_hwloc_t *hwloc,
                                  QUO_obj_type_t type,
                                  pid_t pid,
                                  unsigned type_index,
                                  int *out_result);

int
quo_hwloc_bound(const quo_hwloc_t *hwloc,
                pid_t pid,
                bool *out_bound);

int
quo_hwloc_stringify_cbind(const quo_hwloc_t *hwloc,
                          pid_t pid,
                          char **out_str);

int
quo_hwloc_rebind(const quo_hwloc_t *hwloc,
                 QUO_obj_type_t type,
                 unsigned obj_index);

int
quo_hwloc_bind_push(quo_hwloc_t *hwloc,
                    QUO_bind_push_policy_t policy,
                    QUO_obj_type_t type,
                    unsigned obj_index);

int
quo_hwloc_bind_pop(quo_hwloc_t *hwloc);

int
quo_hwloc_bind_threads(quo_hwloc_t *hwloc,
		       int qid,
		       int qids_in_type,
		       int omp_thread,
		       int num_omp_threads);

int
quo_hwloc_bind_nested_threads(quo_hwloc_t *hwloc,
			      int omp_thread,
			      int num_omp_threads);
#endif
