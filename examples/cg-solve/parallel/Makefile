KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
KOKKOS_REMOTE_SPACES_PATH = ${HOME}/Kokkos/kokkos-remote-spaces
KOKKOS_CUDA_OPTIONS=enable_lambda,rdc
NVSHMEM_PATH=${NVSHMEM_HOME}
#CUDA_PATH=/usr/local/cuda

KOKKOS_REMOTE_SPACES = NVSHMEM
KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH = "Volta70"

QUO_PATH = ${HOME}/Kokkos/TPLS/libquo/build

SRC = cgsolve.cpp


vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

CXX = mpicxx
CXXFLAGS = -O3 -g -I${CUDA_PATH} -I./
CXXFLAGS += -DKOKKOS_ENABLE_NVSHMEM_PTR
LINK = ${CXX}
LDFLAGS =  
EXE = symmetric_view.host

DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos
include $(KOKKOS_REMOTE_SPACES_PATH)/Makefile.kokkos-remote-spaces

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
