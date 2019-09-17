#
# SYNOPSIS
#
#   AX_PKG_QUO_XPM([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
# Enables cross-process memory support in QUO and checks for support.
#
# COPYRIGHT
# Copyright (c) 2017      Los Alamos National Security, LLC.
#                         All rights reserved.
#

AC_DEFUN([AX_PKG_QUO_XPM], [
dnl
AX_PKG_QUO_XPM_HAVE_XPM=0

AC_ARG_ENABLE([xpm],
    AS_HELP_STRING([--enable-xpm],
                   [Enable xpm support (disabled by default).]))

AS_IF([test "x$enable_xpm" = "xyes"], [
    AC_MSG_NOTICE([xpm support requested... checking for support.])
    AX_PKG_QUO_XPM_HAVE_XPM=1
])
dnl
]) dnl AX_PKG_QUO_XPM
