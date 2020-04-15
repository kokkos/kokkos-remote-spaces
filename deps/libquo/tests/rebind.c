/**
 * Copyright (c) 2013-2016 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 * National Security, LLC for the U.S.  Department of Energy. The U.S.
 * Government has rights to use, reproduce, and distribute this software.
 * NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
 * WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
 * SOFTWARE.  If software is modified to produce derivative works, such modified
 * software should be clearly marked, so as not to confuse it with the version
 * available from LANL.
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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <unistd.h>

#include <hwloc.h>

/* a test to see if we can rebind a running process */

static void
emit_binding(const hwloc_topology_t *t)
{
    char *str = NULL;
    hwloc_cpuset_t cpu_set = hwloc_bitmap_alloc();

    hwloc_get_cpubind(*t, cpu_set, HWLOC_CPUBIND_PROCESS);
    hwloc_bitmap_asprintf(&str, cpu_set);
    printf("%d's cpubind bitmap is: %s\n", (int)getpid(), str);

    free(str);
    hwloc_bitmap_free(cpu_set);
}

int
main(void)
{
    int erc = EXIT_SUCCESS;
    unsigned ncores = 0;
    hwloc_cpuset_t cpu_set = hwloc_bitmap_alloc(),
                   first_bind = hwloc_bitmap_alloc();
    hwloc_topology_t topology;
    hwloc_obj_t last_core;

    /* allocate and initialize topology object. */
    hwloc_topology_init(&topology);
    /* build the topology */
    hwloc_topology_load(topology);
    /* get some info */
    ncores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    last_core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, ncores - 1);
    /* stash current binding */
    hwloc_get_cpubind(topology, first_bind, HWLOC_CPUBIND_PROCESS);
    /* start the madness */
    printf("starting rebinding test on pid %d\n", getpid());

    emit_binding(&topology);

    printf("changing binding...\n");

    hwloc_bitmap_copy(cpu_set, last_core->cpuset);
    hwloc_bitmap_singlify(cpu_set);
    hwloc_set_cpubind(topology, cpu_set, HWLOC_CPUBIND_PROCESS);

    emit_binding(&topology);

    printf("reverting binding...\n");

    hwloc_set_cpubind(topology, first_bind, HWLOC_CPUBIND_PROCESS);

    emit_binding(&topology);

    printf("done with rebinding test\n");

    /* cleanup */
    hwloc_bitmap_free(cpu_set);
    hwloc_bitmap_free(first_bind);
    hwloc_topology_destroy(topology);

    return erc;
}
