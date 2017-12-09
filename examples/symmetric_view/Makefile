KOKKOS_PATH = ${HOME}/Shepard/Kokkos/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
QUO_PATH = /home/crtrott/Shepard/Kokkos/TPLS/libquo/build
SRC = main.cpp ../../src/QUO/Kokkos_QUOSpace.cpp
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

CXX = mpicxx
CXXFLAGS = -O0 -g 
LINK = ${CXX}
LDFLAGS =  
EXE = 01_hello_world.host
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "SNB"


CXXFLAGS += -I${QUO_PATH}/include -I../../src -I../../src/QUO
LDFLAGS += -L${QUO_PATH}/lib -lquo-xpm -lquo

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
