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

module quo
      use, intrinsic :: iso_c_binding

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! return codes
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer(c_int) QUO_SUCCESS
      integer(c_int) QUO_SUCCESS_ALREADY_DONE
      integer(c_int) QUO_ERR
      integer(c_int) QUO_ERR_SYS
      integer(c_int) QUO_ERR_OOR
      integer(c_int) QUO_ERR_INVLD_ARG
      integer(c_int) QUO_ERR_CALL_BEFORE_INIT
      integer(c_int) QUO_ERR_TOPO
      integer(c_int) QUO_ERR_MPI
      integer(c_int) QUO_ERR_NOT_SUPPORTED
      integer(c_int) QUO_ERR_POP
      integer(c_int) QUO_ERR_NOT_FOUND

      parameter (QUO_SUCCESS = 0)
      parameter (QUO_SUCCESS_ALREADY_DONE = 1)
      parameter (QUO_ERR = 2)
      parameter (QUO_ERR_SYS = 3)
      parameter (QUO_ERR_OOR = 4)
      parameter (QUO_ERR_INVLD_ARG = 5)
      parameter (QUO_ERR_CALL_BEFORE_INIT = 6)
      parameter (QUO_ERR_TOPO = 7)
      parameter (QUO_ERR_MPI = 8)
      parameter (QUO_ERR_NOT_SUPPORTED = 9)
      parameter (QUO_ERR_POP = 10)
      parameter (QUO_ERR_NOT_FOUND = 11)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! quo object types
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer(c_int) QUO_OBJ_MACHINE
      integer(c_int) QUO_OBJ_NUMANODE
      integer(c_int) QUO_OBJ_SOCKET
      integer(c_int) QUO_OBJ_CORE
      integer(c_int) QUO_OBJ_PU

      parameter (QUO_OBJ_MACHINE = 0)
      parameter (QUO_OBJ_NUMANODE = 1)
      parameter (QUO_OBJ_SOCKET = 2)
      parameter (QUO_OBJ_CORE = 3)
      parameter (QUO_OBJ_PU = 4)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! push policies
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer(c_int) QUO_BIND_PUSH_PROVIDED
      integer(c_int) QUO_BIND_PUSH_OBJ

      parameter (QUO_BIND_PUSH_PROVIDED = 0)
      parameter (QUO_BIND_PUSH_OBJ = 1)

interface
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      integer(c_int) &
      function quo_ptr_free_c(cptr) &
          bind(c, name='QUO_ptr_free')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: cptr
      end function quo_ptr_free_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_version_c(version, subversion) &
          bind(c, name='QUO_version')
          use, intrinsic :: iso_c_binding, only: c_int
          implicit none
          integer(c_int), intent(out) :: version, subversion
      end function quo_version_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_create_c(q, comm) &
          bind(c, name='QUO_create_f2c')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), intent(out) :: q
          integer(c_int), value :: comm
      end function quo_create_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_free_c(q) &
          bind(c, name='QUO_free')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
      end function quo_free_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_nobjs_by_type_c(q, target_type, out_nobjs) &
          bind(c, name='QUO_nobjs_by_type')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: target_type
          integer(c_int), intent(out) :: out_nobjs
      end function quo_nobjs_by_type_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_nobjs_in_type_by_type_c(q, in_type, type_index, &
                                           obj_type, oresult) &
          bind(c, name='QUO_nobjs_in_type_by_type')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: in_type, type_index, obj_type
          integer(c_int), intent(out) :: oresult
      end function quo_nobjs_in_type_by_type_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_cpuset_in_type_c(q, obj_type, type_index, oresult) &
          bind(c, name='QUO_cpuset_in_type')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: obj_type, type_index
          integer(c_int), intent(out) :: oresult
      end function quo_cpuset_in_type_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_qids_in_type_c(q, obj_type, type_index, &
                                  onqids, qids) &
          bind(c, name='QUO_qids_in_type')
          use, intrinsic :: iso_c_binding, only: c_int, c_ptr
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: obj_type, type_index
          integer(c_int), intent(out) :: onqids
          type(c_ptr), intent(out) :: qids
      end function quo_qids_in_type_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_nnumanodes_c(q, n) &
          bind(c, name='QUO_nnumanodes')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_nnumanodes_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_nsockets_c(q, n) &
          bind(c, name='QUO_nsockets')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_nsockets_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_ncores_c(q, n) &
          bind(c, name='QUO_ncores')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_ncores_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_npus_c(q, n) &
          bind(c, name='QUO_npus')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_npus_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_nnodes_c(q, n) &
          bind(c, name='QUO_nnodes')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_nnodes_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_nqids_c(q, n) &
          bind(c, name='QUO_nqids')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_nqids_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_id_c(q, n) &
          bind(c, name='QUO_id')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n
      end function quo_id_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_bound_c(q, bound) &
          bind(c, name='QUO_bound')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent (out) :: bound
      end function quo_bound_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_bind_push_c(q, policy, obj_type, obj_index) &
          bind(c, name='QUO_bind_push')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: policy, obj_type, obj_index
      end function quo_bind_push_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_bind_pop_c(q) &
          bind(c, name='QUO_bind_pop')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
      end function quo_bind_pop_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_barrier_c(q) &
          bind(c, name='QUO_barrier')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
      end function quo_barrier_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_auto_distrib_c(q, distrib_over_this, &
                                  max_qids_per_res_type, oselected) &
          bind(c, name='QUO_auto_distrib')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: distrib_over_this
          integer(c_int), value :: max_qids_per_res_type
          integer(c_int), intent(out) :: oselected
      end function quo_auto_distrib_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
interface
      integer(c_int) &
      function quo_get_mpi_comm_by_type_c(q, target_type, comm) &
          bind(c, name='QUO_get_mpi_comm_by_type_f2c')
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: target_type
          integer(c_int), intent(out):: comm
      end function quo_get_mpi_comm_by_type_c
end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!interface
!      integer(c_int) &
!      function quo_bind_threads_c(q, type, index) &
!         bind(c, name='QUO_bind_threads')
!         use, intrinsic :: iso_c_binding, only: c_int
!         import :: c_ptr
!         implicit none
!         type(c_ptr), value :: q
!         integer(c_int), value :: type, index
!       end function quo_bind_threads_c
!end interface

contains
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_ptr_free(cptr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: cptr
          integer(c_int) :: ierr
          ierr = quo_ptr_free_c(cptr)
      end subroutine quo_ptr_free

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_version(version, subversion, ierr)
          use, intrinsic :: iso_c_binding, only: c_int
          implicit none
          integer(c_int), intent(out) :: version, subversion, ierr
          ierr = quo_version_c(version, subversion)
      end subroutine quo_version

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_create(q, comm, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), intent(out) :: q
          integer, value :: comm
          integer(c_int), intent(out) :: ierr
          ierr = quo_create_c(q, comm)
      end subroutine quo_create

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_free(q, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: ierr
          ierr = quo_free_c(q)
      end subroutine quo_free

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_nobjs_by_type(q, target_type, out_nobjs, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: target_type
          integer(c_int), intent(out) :: out_nobjs
          integer(c_int), intent(out) :: ierr
          ierr = quo_nobjs_by_type_c(q, target_type, out_nobjs)
      end subroutine quo_nobjs_by_type

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_nobjs_in_type_by_type(q, in_type, type_index, &
                                           obj_type, oresult, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: in_type, type_index, obj_type
          integer(c_int), intent(out) :: oresult, ierr
          ierr = quo_nobjs_in_type_by_type_c(q, in_type, type_index, &
                                             obj_type, oresult)
      end subroutine quo_nobjs_in_type_by_type

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_cpuset_in_type(q, obj_type, type_index, &
                                    oresult, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: obj_type, type_index
          logical, intent (out) :: oresult
          integer(c_int), intent(out) :: ierr
          integer(c_int) :: ires
          ierr = quo_cpuset_in_type_c(q, obj_type, type_index, ires)
          oresult = (ires == 1)
      end subroutine quo_cpuset_in_type

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_qids_in_type(q, obj_type, type_index, qids, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: obj_type, type_index
          integer(c_int), allocatable, intent(out) :: qids(:)
          integer(c_int), pointer :: qidsp(:)
          type(c_ptr) :: qidp
          integer(c_int), intent(out) :: ierr
          integer(c_int) :: nqids, i
          ierr = quo_qids_in_type_c(q, obj_type, type_index, &
                                    nqids, qidp)
          call c_f_pointer(qidp, qidsp, [nqids])
          allocate (qids(nqids))
          forall (i = 1 : size(qidsp)) qids(i) = qidsp(i)
          call quo_ptr_free(qidp)
      end subroutine quo_qids_in_type

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_nnumanodes(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_nnumanodes_c(q, n)
      end subroutine quo_nnumanodes

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_nsockets(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_nsockets_c(q, n)
      end subroutine quo_nsockets

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_ncores(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_ncores_c(q, n)
      end subroutine quo_ncores

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_npus(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_npus_c(q, n)
      end subroutine quo_npus

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_nnodes(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_nnodes_c(q, n)
      end subroutine quo_nnodes

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_nqids(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_nqids_c(q, n)
      end subroutine quo_nqids

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_id(q, n, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: n, ierr
          ierr = quo_id_c(q, n)
      end subroutine quo_id

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_bound(q, bound, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          logical, intent (out) :: bound
          integer(c_int), intent(out) :: ierr
          integer(c_int) :: ibound = 0
          ierr = quo_bound_c(q, ibound)
          bound = (ibound == 1)
      end subroutine quo_bound

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_bind_push(q, policy, obj_type, obj_index, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: policy, obj_type, obj_index
          integer(c_int), intent(out) :: ierr
          ierr = quo_bind_push_c(q, policy, obj_type, obj_index)
      end subroutine quo_bind_push

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_bind_pop(q, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: ierr
          ierr = quo_bind_pop_c(q)
      end subroutine quo_bind_pop

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_barrier(q, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), intent(out) :: ierr
          ierr = quo_barrier_c(q)
      end subroutine quo_barrier

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_auto_distrib(q, distrib_over_this, &
                                  max_qids_per_res_type, oselected, &
                                  ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: distrib_over_this
          integer(c_int), value :: max_qids_per_res_type
          integer(c_int) :: iselected
          logical, intent(out) :: oselected
          integer(c_int), intent(out) :: ierr
          ierr = quo_auto_distrib_c(q, distrib_over_this, &
                                    max_qids_per_res_type, iselected)
          oselected = (iselected == 1)
      end subroutine quo_auto_distrib

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine quo_get_mpi_comm_by_type(q, target_type, comm, ierr)
          use, intrinsic :: iso_c_binding, only: c_ptr, c_int
          implicit none
          type(c_ptr), value :: q
          integer(c_int), value :: target_type
          integer, intent(out) :: comm
          integer(c_int), intent(out) :: ierr
          ierr = quo_get_mpi_comm_by_type_c(q, target_type, comm)
      end subroutine quo_get_mpi_comm_by_type

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !subroutine quo_bind_threads(q, type, index, ierr)
      !    use, intrinsic :: iso_c_binding, only: c_int
      !    implicit none
      !    type(c_ptr), value :: q
      !    integer(c_int), value :: type, index
      !    integer(c_int), intent(out) :: ierr
      !    ierr = quo_bind_threads_c(q, type, index)
      !end subroutine quo_bind_threads
end module quo
