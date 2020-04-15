! Copyright (c) 2013-2016 Los Alamos National Security, LLC
!                         All rights reserved.
!
! This software was produced under U.S. Government contract DE-AC52-06NA25396
! for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
! National Security, LLC for the U.S. Department of Energy. The U.S. Government
! has rights to use, reproduce, and distribute this software.  NEITHER THE
! GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
! OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If
! software is modified to produce derivative works, such modified software
! should be clearly marked, so as not to confuse it with the version available
! from LANL.
!
! Additionally, redistribution and use in source and binary forms, with or
! without modification, are permitted provided that the following
! conditions are met:
!
! · Redistributions of source code must retain the above copyright notice,
!   this list of conditions and the following disclaimer.
!
! · Redistributions in binary form must reproduce the above copyright
!   notice, this list of conditions and the following disclaimer in the
!   documentation and/or other materials provided with the distribution.
!
! · Neither the name of Los Alamos National Security, LLC, Los Alamos
!   National Laboratory, LANL, the U.S. Government, nor the names of its
!   contributors may be used to endorse or promote products derived from
!   this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
! CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
! BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
! FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS
! ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
! INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
! (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
! HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
! STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.

! does nothing useful. just used to exercise the fortran interface.
! better examples can be found in demos

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
program quofort

    use quo
    use, intrinsic :: iso_c_binding
    implicit none

    include "mpif.h"

    logical bound, inres, have_res
    integer(c_int) info
    integer(c_int) ver, subver
    integer(c_int) nres, qid
    integer(c_int) cwrank
    integer(c_int), allocatable, dimension(:) :: sock_qids
    type(c_ptr) quoc
    integer machine_comm

    call quo_version(ver, subver, info)

    print *, info, ver, subver

    call mpi_init(info)
    call mpi_comm_rank(MPI_COMM_WORLD, cwrank, info)

    call quo_create(quoc, MPI_COMM_WORLD, info)

    call quo_bound(quoc, bound, info)

    call quo_nobjs_by_type(quoc, QUO_OBJ_CORE, nres, info)
    print *, 'bound, nres', bound, nres

    call quo_nobjs_in_type_by_type(quoc, QUO_OBJ_MACHINE, 0, &
                                   QUO_OBJ_SOCKET, nres, info)
    print *,'sock on machine', nres

    call quo_cpuset_in_type(quoc, QUO_OBJ_SOCKET, 0, inres, info)
    print *, 'rank on sock 0', cwrank, inres

    call quo_qids_in_type(quoc, QUO_OBJ_SOCKET, 0, sock_qids, info)
    print *, 'sock_qids', sock_qids
    deallocate (sock_qids)

    call quo_nnumanodes(quoc, nres, info)
    print *, 'nnumanodes', nres

    call quo_nsockets(quoc, nres, info)
    print *, 'nsockets', nres

    call quo_ncores(quoc, nres, info)
    print *, 'ncores', nres

    call quo_npus(quoc, nres, info)
    print *, 'npus', nres

    call quo_nnodes(quoc, nres, info)
    print *, 'nnodes', nres

    call quo_nqids(quoc, nres, info)
    print *, 'nqids', nres

    call quo_id(quoc, qid, info)
    print *, 'qid', qid

    if (qid == 0) then
        print *, 'hello from qid 0!'
    endif

    call quo_bind_push(quoc, QUO_BIND_PUSH_OBJ, QUO_OBJ_SOCKET, -1, info)

    call quo_bound(quoc, bound, info)
    print *, 'bound after push', bound

    call quo_bind_pop(quoc, info)

    call quo_bound(quoc, bound, info)
    print *, 'bound after pop', bound

    call quo_auto_distrib(quoc, QUO_OBJ_SOCKET, 2, have_res, info)
    print *, 'rank, have_res', cwrank, have_res

    call quo_barrier(quoc, info)

    call quo_get_mpi_comm_by_type(quoc, QUO_OBJ_MACHINE, machine_comm, info)
    if (info /= QUO_SUCCESS) then
        print *, 'QUO_FAILURE DETECTED, info', info
        stop
    end if

    call mpi_comm_free(machine_comm, info)

    call quo_free(quoc, info)

    call mpi_finalize(info)

    call exit(0)

end program quofort
