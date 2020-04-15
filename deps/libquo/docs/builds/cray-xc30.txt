################################################################################

How to Build QUO on a Cray XC30

################################################################################

o Load appropriate programming environment (Intel, PGI, etc.).

o export QUO_PREFIX=[prefix]

o Build static libnuma and install to QUO's install prefix.
make prefix=$QUO_PREFIX libdir=$QUO_PREFIX/lib CC=gcc all install
o Delete $QUO_PREFIX/bin directory
  (we don't need those utilities -- only need libs and includes).

o Build QUO and link against the installed version of libnuma.
./configure CC=cc FC=ftn \
LDFLAGS="-dynamic -L${QUO_PREFIX}/lib" \
--prefix=$QUO_PREFIX --enable-static

o Install
make && make install
