###############################################################################
# Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
#
# Copyright (c) 2012      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Copyright (c) 2012      Inria.  All rights reserved.
#
# Copyright (c) 2013-2017 Los Alamos National Security, LLC.
#                         All rights reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
################################################################################

SUBDIRS = \
hwloc

lib_LTLIBRARIES = \
libquo.la

include_HEADERS = \
quo.h

libquo_la_SOURCES = \
quo-private.h \
quo-utils.h quo-utils.c \
quo-sm.h quo-sm.c \
quo-set.h quo-set.c \
quo-hwloc.h quo-hwloc.c \
quo-mpi.h quo-mpi.c \
quo-auto-distrib.c \
quo.h quo.c \
quof.c

libquo_la_CFLAGS =
libquo_la_LDFLAGS = -version-info @QUO_LIBVINFO@
libquo_la_LIBADD = $(HWLOC_EMBEDDED_LDADD) $(HWLOC_EMBEDDED_LIBS)
