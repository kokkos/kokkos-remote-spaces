KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
QUO_PATH = ${HOME}/Kokkos/TPLS/libquo/build
SRC = main.cpp
ifeq ($(KOKKOS_INTERNAL_USE_QUOSPACE), 1)
  SRC += ../../src/QUO/Kokkos_QUOSpace.cpp
endif
ifeq ($(KOKKOS_INTERNAL_USE_SHMEMSPACE), 1)
  SRC += ../../src/SHMEM/Kokkos_SHMEMSpace.cpp
endif

vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

CXX = mpicxx
CXXFLAGS = -O0 -g 
LINK = ${CXX}
LDFLAGS =  
EXE = symmetric_view.host
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "SNB"

CXXFLAGS += -I../../src
ifeq ($(KOKKOS_INTERNAL_USE_QUOSPACE), 1)
CXXFLAGS += -I${QUO_PATH}/include -I../../src/QUO
LDFLAGS += -L${QUO_PATH}/lib -lquo-xpm -lquo
CXXFLAGS += -DKOKKOS_ENABLE_QUOSPACE
endif
ifeq ($(KOKKOS_INTERNAL_USE_SHMEMSPACE), 1)
CXXFLAGS += -I../../src/SHMEM
LDFLAGS += -loshmem
CXXFLAGS += -DKOKKOS_ENABLE_SHMEMSPACE
endif


DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
