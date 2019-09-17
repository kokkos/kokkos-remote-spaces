#
# Copyright (c) 2013-2017 Los Alamos National Security, LLC
#                         All rights reserved.
#
# This file is part of the libquo project. See the LICENSE file at the
# top-level directory of this distribution.
#

AC_PREREQ([2.60])

AC_INIT([libquo],
        [1.3.1-alpha],
        [samuel@lanl.gov],
        [libquo],
        [http://losalamos.github.io/libquo/])

AC_CONFIG_MACRO_DIR([config])
AC_CONFIG_SRCDIR([src])

AX_QUO_LIBTOOL_VERSION

################################################################################
# AC_CANONICAL_*
################################################################################
# for hwloc - must be in between AC_INIT and AM_INIT_AUTOMAKE
AC_CANONICAL_BUILD
AC_CANONICAL_HOST
AC_CANONICAL_TARGET

################################################################################
# Automake things
################################################################################
# note: -Wall -Werror are automake options.  they are NOT compiler flags
AM_INIT_AUTOMAKE([1.11 -Wall -Werror foreign])
# Build using AM_SILENT_RULES
AM_SILENT_RULES([yes])
AM_CONFIG_HEADER([config.h])

################################################################################
# C Language
################################################################################
AC_USE_SYSTEM_EXTENSIONS

AC_PROG_CC_C99
AC_PROG_CC_C_O
AC_PROG_CPP

AX_PROG_MPICC

AS_IF([test "x$AX_PROG_MPICC_HAVE_MPICC" != "x1"],dnl
      [AC_MSG_ERROR(
       [*** CC cannot compile MPI apps. set CC to something that can. ***])])

################################################################################
# Fortran Language Support
################################################################################
AC_PROG_FC
AC_PROG_FC_C_O

AX_PROG_MPIFC
# determine whether or not we have a fortran wrapper compiler that can deal with
# mpi apps.  if not disable fortran support.
AS_IF([test "x$AX_PROG_MPIFC_HAVE_MPIFC" != "x1"],dnl
      [AC_MSG_WARN(
       [*** FC cannot compile MPI apps. set FC to something that can. ***])])
AM_CONDITIONAL(QUO_WITH_MPIFC, test "x$AX_PROG_MPIFC_HAVE_MPIFC" = "x1")

################################################################################
# Libtool things
################################################################################
AM_PROG_AR

LT_PREREQ([2.4.2])
LT_INIT

################################################################################
# C++ Language Support
################################################################################
# Delay this bit until now. There is some weird libtool linker interaction that
# I don't understand that is causing some link-time errors in our distro. This
# ordering seems to "fix" things, but I still don't understand why...
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_CXX_C_O

AX_PROG_MPICXX
# determine whether or not we have a C++ wrapper compiler that can deal with
# mpi apps.  if not disable C++ support.
AS_IF([test "x$AX_PROG_MPICXX_HAVE_MPICXX" != "x1"],dnl
      [AC_MSG_WARN(
       [*** CXX cannot compile MPI apps. set CXX to something that can. ***])])
AM_CONDITIONAL(QUO_WITH_MPICXX, test "x$AX_PROG_MPICXX_HAVE_MPICXX" = "x1")

################################################################################
# Cross-process memory (xpm) support
################################################################################

AX_PKG_QUO_XPM
AM_CONDITIONAL(QUO_WITH_XPM, test "x$AX_PKG_QUO_XPM_HAVE_XPM" = "x1")

# checks for header files.
AC_CHECK_HEADERS([\
inttypes.h limits.h stdint.h stdlib.h string.h unistd.h stdbool.h time.h \
getopt.h ctype.h netdb.h sys/socket.h netinet/in.h arpa/inet.h sys/types.h \
stddef.h assert.h pthread.h sys/mman.h sys/stat.h fcntl.h syscall.h omp.h \
sched.h strings.h stdio.h errno.h math.h
])

# checks for typedefs, structures, and compiler characteristics.
AC_TYPE_PID_T
AC_TYPE_SIZE_T

# checks for library functions.
AC_CHECK_FUNCS([memset strerror strtoul mmap])

dnl check for sizeof(uintptr_t) for the Fortran interface.
dnl This sets the size of QUO_IKIND in the quof.h header that is generated.
dnl AC_COMPUTE_INT(quo_sizeof_voidptr, [sizeof(void *)])
dnl AC_SUBST(QUO_IKIND_VAL, $quo_sizeof_voidptr)

################################################################################
# pthreads stuff
################################################################################
AX_PTHREAD([], AC_MSG_ERROR([*** pthread configure failure. ***]))
# TODO
#AC_SUBST(QUO_PTHREAD_CFLAGS, $PTHREAD_CFLAGS)
#AC_SUBST(QUO_PTHREAD_LIBS, $PTHREAD_LIBS)

# Check for pthread_barrier_init. Error out if not found.
AC_SEARCH_LIBS([pthread_barrier_init], [pthread],
               [],
               [AC_MSG_ERROR([pthread_barrier_init() required but not found.])])

################################################################################
# OpenMP configury
################################################################################
AX_OPENMP([HAVE_OPENMP=1], [])

if test "x$HAVE_OPENMP" = "x"; then
    AC_MSG_WARN([No OpenMP support. Continuing without OpenMP examples.])
else
    AC_MSG_NOTICE([OpenMP support detected... Building OpenMP examples.])
    AC_SUBST([OPENMP_CFLAGS])
    AC_SUBST([OPENMP_CXXFLAGS])
    AC_SUBST([OPENMP_FFLAGS])
fi
AM_CONDITIONAL([QUO_HAVE_OPENMP_SUPPORT],
               [test "x$HAVE_OPENMP" != "x"])

################################################################################
# hwloc stuff
################################################################################
# set some hwloc configure options
enable_embedded_mode=yes
enable_libxml2=no
enable_xml=yes
# TODO - allow for this someday (busted on the Crays -- no static libs)
enable_pci=no
enable_libpci=no
enable_libudev=no
#enable_libnuma=no

HWLOC_SET_SYMBOL_PREFIX(quo_internal_)
HWLOC_SETUP_CORE([src/hwloc],
                 [],
                 [AC_MSG_ERROR([*** hwloc configure failure. ***])],
                 [1])
CPPFLAGS="$HWLOC_EMBEDDED_CPPFLAGS $CPPFLAGS"
LIBS="$HWLOC_EMBEDDED_LIBS $LIBS"
HWLOC_DO_AM_CONDITIONALS

################################################################################
# For docs target support
################################################################################
AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test "x$DOXYGEN" = "x"; then
    AC_MSG_WARN([Doxygen not found. Continuing without Doxygen support])
fi
AM_CONDITIONAL([QUO_HAVE_DOXYGEN_DOCS_SUPPORT],
               [test "x$DOXYGEN" != "x"])

################################################################################
# For package info (quo-info)
################################################################################
QUO_WHICH_CC=`which \`echo "$CC" | tr -s ' ' | cut -f1 -d' '\``
QUO_WHICH_FC=`which \`echo "$FC" | tr -s ' ' | cut -f1 -d' '\``
if test "x$QUO_WHICH_FC" = x; then
    QUO_WHICH_FC="none"
fi

AC_ARG_VAR(USER,
           [ Username used in build info (Default: `id -u -n` output)])
if test "x$USER" = x; then
    USER=`id -u -n`
fi
AC_ARG_VAR(HOSTNAME,
           [ Hostname used in build info (Default: `hostname` output)])
if test "x$HOSTNAME" = x; then
    HOSTNAME=`hostname`
fi
AC_ARG_VAR(DATE,
           [ Date used in build info (Default: `date` output)])
if test "x$DATE" = x; then
    DATE=`date`
fi

AC_SUBST([CC])
AC_SUBST([QUO_WHICH_CC])
AC_SUBST([CFLAGS])
AC_SUBST([CPPFLAGS])
AC_SUBST([CXXFLAGS])
AC_SUBST([CXXCPPFLAGS])
AC_SUBST([FC])
AC_SUBST([QUO_WHICH_FC])
AC_SUBST([USER])
AC_SUBST([HOSTNAME])
AC_SUBST([DATE])
AC_SUBST([FFLAGS])
AC_SUBST([FCFLAGS])
AC_SUBST([LDFLAGS])
AC_SUBST([LIBS])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/fortran/Makefile
                 src/tools/Makefile
                 src/xpm/Makefile
                 tests/Makefile
                 demos/Makefile
                 demos/mpi-openmp/Makefile
                 pkgconfig/Makefile
                 pkgconfig/libquo.pc
                 docs/Makefile
                 docs/doxyfile])

AC_OUTPUT

# let the user know the configuration
cat << EOF

+-------------------------------------------------------------------------------
| libquo Configuration :::::::::::::::::::::::::::::::::::::::::::::::::::::::
+-------------------------------------------------------------------------------
| version     : ${VERSION}
| target      : ${target}
| CC          : ${CC}
| CFLAGS      : ${CFLAGS}
| CPP         : ${CPP}
| CPPFLAGS    : ${CPPFLAGS}
| LDFLAGS     : ${LDFLAGS}
| LIBS        : ${LIBS}
+-------------------------------------------------------------------------------
| C++ Support Configuration ::::::::::::::::::::::::::::::::::::::::::::::::::
+-------------------------------------------------------------------------------
EOF

if test "x$AX_PROG_MPICXX_HAVE_MPICXX" = "x1"; then
cat << EOF
| CXX         : ${CXX}
| CXXFLAGS    : ${CXXFLAGS}
EOF
else
cat << EOF
| Not Enabled
EOF
fi

cat << EOF
+-------------------------------------------------------------------------------
| Fortran Support Configuration ::::::::::::::::::::::::::::::::::::::::::::::
+-------------------------------------------------------------------------------
EOF

if test "x$AX_PROG_MPIFC_HAVE_MPIFC" = "x1"; then
cat << EOF
| FC          : ${FC}
| FCFLAGS     : ${FCFLAGS}
| FCLIBS      : ${FCLIBS}
EOF
else
cat << EOF
| Not Enabled
EOF
fi
cat << EOF
+-------------------------------------------------------------------------------
| Other Configury ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
+-------------------------------------------------------------------------------
EOF

# Doxygen configury.
printf "| Doxygen docs Target Available: %s\n" \
    $( test "x$DOXYGEN" = "x" && printf "no" || printf "yes" )
printf "|\n"
# OpenMP configury.
printf "| Build OpenMP Examples: "
if test "x$HAVE_OPENMP" = "x"; then
    printf "no\n"
else
    printf "yes\n"
cat << EOF
| OPENMP_CFLAGS  : $OPENMP_CFLAGS
| OPENMP_CXXFLAGS: $OPENMP_CXXFLAGS
| OPENMP_FFLAGS  : $OPENMP_FFLAGS
EOF
fi

cat << EOF
+-------------------------------------------------------------------------------
EOF
