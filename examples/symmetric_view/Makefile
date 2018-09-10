KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
KOKKOS_REMOTE_SPACES_PATH = ${HOME}/Kokkos/kokkos-remote-spaces
KOKKOS_CUDA_OPTIONS=enable_lambda

KOKKOS_REMOTE_SPACES = MPI
QUO_PATH = ${HOME}/Kokkos/TPLS/libquo/build

SRC = main.cpp


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
