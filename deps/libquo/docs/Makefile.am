###############################################################################
# Copyright (c)      2016 Los Alamos National Security, LLC.
#                         All rights reserved.
################################################################################

EXTRA_DIST = \
builds

if QUO_HAVE_DOXYGEN_DOCS_SUPPORT
all-local:
	$(DOXYGEN) doxyfile
else
# Nothing to do
all-local:
endif

clean-local:
	rm -rf html latex
