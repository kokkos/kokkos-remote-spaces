#
# SYNOPSIS
#
#   AX_PKG_HWLOC()
#
# DESCRIPTION
# checks for hwloc headers and library.
#
# COPYRIGHT
# Copyright (c) 2013      Los Alamos National Security, LLC.
#                         All rights reserved.
#

m4_define([AX_PKG_HWLOC_testbody], [
    #include <hwloc.h>
])

AC_DEFUN([AX_PKG_HWLOC], [dnl
    AX_PKG_HWLOC_HAVE_HWLOC=0

    AC_SEARCH_LIBS([hwloc_topology_init], [hwloc],dnl
                   [ax_pkg_hwloc_libs_happy=1],dnl
                   [ax_pkg_hwloc_libs_happy=0])

    AC_COMPILE_IFELSE([AC_LANG_SOURCE([AX_PKG_HWLOC_testbody])],dnl
                      [ax_pkg_hwloc_includes_happy=1],dnl
                      [ax_pkg_hwloc_includes_happy=0])

    if test "x$ax_pkg_hwloc_includes_happy" = "x1"; then
        if test "x$ax_pkg_hwloc_libs_happy" = "x1"; then
            AX_PKG_HWLOC_HAVE_HWLOC=1
        else
            AC_MSG_WARN([could not find hwloc devel libs. please set LDFLAGS.])
        fi
    else
        AC_MSG_WARN([could not find hwloc devel headers. please set CPPFLAGS.])
    fi
])
