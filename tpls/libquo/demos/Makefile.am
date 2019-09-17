###############################################################################
# Copyright (c) 2013-2016 Los Alamos National Security, LLC
#                         All rights reserved.
#
# This software was produced under U.S. Government contract DE-AC52-06NA25396
# for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
# National Security, LLC for the U.S. Department of Energy. The U.S. Government
# has rights to use, reproduce, and distribute this software.  NEITHER THE
# GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
# OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If
# software is modified to produce derivative works, such modified software
# should be clearly marked, so as not to confuse it with the version available
# from LANL.
#
# Additionally, redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following conditions are
# met:
#
# · Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# · Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# · Neither the name of Los Alamos National Security, LLC, Los Alamos National
#   Laboratory, LANL, the U.S. Government, nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
# SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
################################################################################

SUBDIRS = \
mpi-openmp

noinst_PROGRAMS = \
basic-info \
even-distrib \
quodemo-0 \
quodemo-1 \
caller-driven-multilib \
callee-driven-multilib \
mpi-omp \
quo-mpi-sub-comm

if QUO_WITH_MPIFC
noinst_PROGRAMS += \
quofort
endif

noinst_LTLIBRARIES = \
libcaller_driven_callee_p1.la \
libcallee_driven_callee_p1.la

###
basic_info_SOURCES = basic-info.c 
basic_info_CPPFLAGS = -I$(top_srcdir)/src
basic_info_LDADD = $(top_builddir)/src/libquo.la

###
even_distrib_SOURCES = even-distrib.c
even_distrib_CPPFLAGS = -I$(top_srcdir)/src
even_distrib_LDADD = $(top_builddir)/src/libquo.la

###
quodemo_0_SOURCES = quodemo-0.c
quodemo_0_CPPFLAGS = -I$(top_srcdir)/src
quodemo_0_LDADD = $(top_builddir)/src/libquo.la

###
quodemo_1_SOURCES = quodemo-1.c
quodemo_1_CPPFLAGS = -I$(top_srcdir)/src
quodemo_1_LDADD = $(top_builddir)/src/libquo.la

### caller-driven multi-library example
caller_driven_multilib_SOURCES = caller-driven-ex-common.h caller-driven-ex-p0.c
caller_driven_multilib_CPPFLAGS = -I$(top_srcdir)/src
caller_driven_multilib_LDADD = \
$(top_builddir)/src/libquo.la libcaller_driven_callee_p1.la

libcaller_driven_callee_p1_la_SOURCES = \
caller-driven-ex-common.h caller-driven-ex-p1.h caller-driven-ex-p1.c
libcaller_driven_callee_p1_la_CPPFLAGS = -I$(top_srcdir)/src
libcaller_driven_callee_p1_la_LIBADD = $(top_builddir)/src/libquo.la

### callee-driven multi-library example
callee_driven_multilib_SOURCES = callee-driven-ex-p0.c
callee_driven_multilib_CPPFLAGS = -I$(top_srcdir)/src
callee_driven_multilib_LDADD = \
$(top_builddir)/src/libquo.la libcallee_driven_callee_p1.la

libcallee_driven_callee_p1_la_SOURCES = \
callee-driven-ex-p1.h callee-driven-ex-p1.c
libcallee_driven_callee_p1_la_CPPFLAGS = -I$(top_srcdir)/src
libcallee_driven_callee_p1_la_LIBADD = $(top_builddir)/src/libquo.la

###
mpi_omp_SOURCES = mpi-omp.c
mpi_omp_CPPFLAGS = -I$(top_srcdir)/src
mpi_omp_LDADD = $(top_builddir)/src/libquo.la

###
quo_mpi_sub_comm_SOURCES = quo-mpi-sub-comm.c
quo_mpi_sub_comm_CPPFLAGS = -I$(top_srcdir)/src
quo_mpi_sub_comm_LDADD = $(top_builddir)/src/libquo.la

### fortran interface test
if QUO_WITH_MPIFC
quofort_SOURCES = quofort.f90
quofort_FCFLAGS  = -I$(top_builddir)/src/fortran
quofort_LDADD = \
$(top_builddir)/src/fortran/libquo-usequo.la \
$(top_builddir)/src/libquo.la
endif
