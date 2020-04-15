#
# Copyright (c) 2017      Los Alamos National Security, LLC
#                         All rights reserved.
#
# This file is part of the libquo project. See the LICENSE file at the
# top-level directory of this distribution.
#

noinst_PROGRAMS =
noinst_LTLIBRARIES =

if QUO_HAVE_OPENMP_SUPPORT
noinst_PROGRAMS += \
dgemv coupled-eta-zeta

###
dgemv_SOURCES = dgemv.c
dgemv_CFLAGS = @OPENMP_CFLAGS@
dgemv_CPPFLAGS = -I$(top_srcdir)/src
dgemv_LDADD = $(top_builddir)/src/libquo.la
endif # QUO_HAVE_OPENMP_SUPPORT

noinst_PROGRAMS += \
coupled-eta-zeta
###
coupled_eta_zeta_SOURCES = coupled-eta-zeta.c
coupled_eta_zeta_CFLAGS =
coupled_eta_zeta_CPPFLAGS =
coupled_eta_zeta_LDADD = libeta.la libzeta.la

noinst_LTLIBRARIES += \
libzeta.la
###
libzeta_la_SOURCES = zeta.h zeta.c
libzeta_la_CFLAGS =
libzeta_la_CPPFLAGS =
libzeta_la_LIBADD =

noinst_LTLIBRARIES += \
libeta.la
###
libeta_la_SOURCES = eta.h eta.c
libeta_la_CFLAGS =
libeta_la_CPPFLAGS =
libeta_la_LIBADD =
