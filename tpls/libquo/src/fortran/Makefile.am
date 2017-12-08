###############################################################################
# Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2012      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2012      Inria.  All rights reserved.
# Copyright (c) 2013-2016 Los Alamos National Security, LLC.
#                         All rights reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
################################################################################

lib_LTLIBRARIES =

if QUO_WITH_MPIFC
lib_LTLIBRARIES += libquo-usequo.la
endif

if QUO_WITH_MPIFC
libquo_usequo_la_SOURCES = quof.f90

libquo_usequo_la_FCFLAGS = -I$(top_srcdir)/src
libquo_usequo_la_LDFLAGS = -version-info @QUO_LIBVINFO@
libquo_usequo_la_LIBADD = $(top_builddir)/src/libquo.la
endif

# these are generated
MOSTLYCLEANFILES = `ls *.mod`

# install the generated .mod files.  unfortunately, each f90 compiler
# may generate different filenames, so we have to use a glob.  :-(
install-exec-hook:
	@ for file in `ls *.mod`; do \
	  echo $(INSTALL) $$file $(DESTDIR)$(libdir); \
	  $(INSTALL) $$file $(DESTDIR)$(libdir); \
	done

uninstall-local:
	@ for file in `ls *.mod`; do \
	  echo rm -f $(DESTDIR)$(libdir)/$$file; \
	  rm -f $(DESTDIR)$(libdir)/$$file; \
	done
