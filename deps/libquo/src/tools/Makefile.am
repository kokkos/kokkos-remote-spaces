###############################################################################
# Copyright (c)      2016 Los Alamos National Security, LLC.
#                         All rights reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
################################################################################

bin_PROGRAMS = \
quo-info

quo_info_SOURCES = \
quo-info.c

quo_info_CPPFLAGS = \
-I${top_srcdir}/src \
-DQUO_BUILD_USER="\"@USER@\"" \
-DQUO_BUILD_HOST="\"@HOSTNAME@\"" \
-DQUO_BUILD_DATE="\"@DATE@\"" \
-DQUO_BUILD_PREFIX="\"@prefix@\"" \
-DQUO_BUILD_INCLUDEDIR="\"@includedir@\"" \
-DQUO_BUILD_LIBDIR="\"@libdir@\"" \
-DQUO_BUILD_CC="\"@CC@\"" \
-DQUO_BUILD_WHICH_CC="\"@QUO_WHICH_CC@\"" \
-DQUO_BUILD_CFLAGS="\"@CFLAGS@\"" \
-DQUO_BUILD_CPPFLAGS="\"@CPPFLAGS@\"" \
-DQUO_BUILD_CXXFLAGS="\"@CXXFLAGS@\"" \
-DQUO_BUILD_CXXCPPFLAGS="\"@CXXCPPFLAGS@\""

if QUO_WITH_MPIFC
quo_info_CPPFLAGS += \
-DQUO_WITH_MPIFC \
-DQUO_BUILD_FC="\"@FC@\"" \
-DQUO_BUILD_WHICH_FC="\"@QUO_WHICH_FC@\"" \
-DQUO_BUILD_FFLAGS="\"@FFLAGS@\"" \
-DQUO_BUILD_FCFLAGS="\"@FCFLAGS@\""
endif

quo_info_CPPFLAGS += \
-DQUO_BUILD_LDFLAGS="\"@LDFLAGS@\"" \
-DQUO_BUILD_LIBS="\"@LIBS@\""
