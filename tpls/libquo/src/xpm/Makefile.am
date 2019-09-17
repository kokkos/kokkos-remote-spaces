#
# Copyright (c) 2017      Los Alamos National Security, LLC
#                         All rights reserved.
#
# This file is part of the libquo project. See the LICENSE file at the
# top-level directory of this distribution.
#

lib_LTLIBRARIES =
include_HEADERS =

if QUO_WITH_XPM
lib_LTLIBRARIES += libquo-xpm.la

include_HEADERS += quo-xpm.h

libquo_xpm_la_SOURCES = \
quo-xpm.c

libquo_xpm_la_CFLAGS = -I$(top_srcdir)/src
libquo_xpm_la_LDFLAGS = -version-info @QUO_LIBVINFO@
libquo_xpm_la_LIBADD = $(top_builddir)/src/libquo.la
endif
