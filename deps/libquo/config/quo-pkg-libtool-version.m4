#
# SYNOPSIS
#
#   AX_QUO_LIBTOOL_VERSION()
#
# DESCRIPTION
# sets (AC_SUBSTs) release libtool version strings:
# QUO_VERSION_CURRENT
# QUO_VERSION_REVISION
# QUO_VERSION_AGE
#
# COPYRIGHT
# Copyright (c)      2016 Los Alamos National Security, LLC.
#                         All rights reserved.
#

# From: http://www.sourceware.org/autobook/autobook/autobook_91.html
#
# The version scheme used by Libtool tracks interfaces, where an interface is the
# set of exported entry points into the library. All Libtool libraries start with
# `-version-info' set to `0:0:0' -- this will be the default version number if
# you don't explicitly set it on the Libtool link command line. The meaning of
# these numbers (from left to right) is as follows:
#
# CURRENT -- The number of the current interface exported by the library. A
# current value of `0', means that you are calling the interface exported by
# this library interface 0.
#
# REVISION -- The implementation number of the most recent interface exported by
# this library. In this case, a revision value of `0' means that this is the
# first implementation of the interface.  If the next release of this library
# exports the same interface, but has a different implementation (perhaps some
# bugs have been fixed), the revision number will be higher, but current number
# will be the same. In that case, when given a choice, the library with the
# highest revision will always be used by the runtime loader.
#
# AGE -- The number of previous additional interfaces supported by this library.
# If age were `2', then this library can be linked into executables which were
# built with a release of this library that exported the current interface
# number, current, or any of the previous two interfaces.  By definition age
# must be less than or equal to current. At the outset, only the first ever
# interface is implemented, so age can only be `0'.
#
# For later releases of a library, the `-version-info' argument needs to be set
# correctly depending on any interface changes you have made. This is quite
# straightforward when you understand what the three numbers mean:

# If you have changed any of the sources for this library, the revision number
# must be incremented. This is a new revision of the current interface.  If the
# interface has changed, then current must be incremented, and revision reset to
# `0'. This is the first revision of a new interface.  If the new interface is a
# superset of the previous interface (that is, if the previous interface has not
# been broken by the changes in this new release), then age must be incremented.
# This release is backwards compatible with the previous release.  If the new
# interface has removed elements with respect to the previous interface, then
# you have broken backward compatibility and age must be reset to `0'. This
# release has a new, but backwards incompatible interface.  For example, if the
# next release of the library included some new commands for an existing socket
# protocol, you would use -version-info 1:0:1. This is the first revision of a
# new interface. This release is backwards compatible with the previous release.
# Later, you implement a faster way of handling part of the algorithm at the
# core of the library, and release it with -version-info 1:1:1. This is a new
# revision of the current interface.
#
# Unfortunately the speed of your new implementation can only be fully exploited
# by changing the API to access the structures at a lower level, which breaks
# compatibility with the previous interface, so you release it as -version-info
# 2:0:0. This release has a new, but backwards incompatible interface.

AC_DEFUN([AX_QUO_LIBTOOL_VERSION], [dnl
    QUO_VERSION_CURRENT=6
    QUO_VERSION_REVISION=0
    QUO_VERSION_AGE=0

    QUO_LIBVINFO="$QUO_VERSION_CURRENT:$QUO_VERSION_REVISION:$QUO_VERSION_AGE"

    AC_SUBST([QUO_VERSION_CURRENT])
    AC_SUBST([QUO_VERSION_REVISION])
    AC_SUBST([QUO_VERSION_AGE])
    AC_SUBST([QUO_LIBVINFO])

    AC_DEFINE_UNQUOTED([QUO_VERSION_CURRENT],
                       [$QUO_VERSION_CURRENT],
                       [Current version.])

    AC_DEFINE_UNQUOTED([QUO_VERSION_REVISION],
                       [$QUO_VERSION_REVISION],
                       [Current revision.])

    AC_DEFINE_UNQUOTED([QUO_VERSION_AGE],
                       [$QUO_VERSION_AGE],
                       [Current age.])
])
