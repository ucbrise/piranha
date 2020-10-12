CONDA_BASE=/data/jlwatson/anaconda3
BUILDDIR=build

USING_GPU := ENABLED

ifdef USING_GPU
CXX=nvcc
FLAGS := -Xcompiler="-O3,-w,-std=c++11,-pthread,-msse4.1,-maes,-msse2,-mpclmul,-fpermissive,-fpic,-DUSING_GPU" -Xcudafe "--diag_suppress=declared_but_not_referenced"
else
CXX=g++
FLAGS := -O3 -w -std=c++11 -pthread -msse4.1 -maes -msse2 -mpclmul -fpermissive -fpic -DUSING_EIGEN
endif

VPATH             := src/ util/
ifdef USING_GPU
VPATH             += gpumatmul/
endif

SRC_CPP_FILES     := $(wildcard src/*.cpp) $(wildcard util/*.cpp) $(wildcard gpumatmul/*.cpp)
ifdef USING_GPU
SRC_CPP_FILES     += $(wildcard gpumatmul/*.cpp)
SRC_CU_FILES      := $(wildcard src/*.cu) $(wildcard gpumatmul/*.cu)
endif

OBJ_FILES         := $(addprefix $(BUILDDIR)/, $(notdir $(SRC_CPP_FILES:.cpp=.o)))
ifdef USING_GPU
OBJ_FILES         += $(addprefix $(BUILDDIR)/, $(notdir $(SRC_CU_FILES:.cu=.o)))
endif

HEADER_FILES      := $(wildcard src/*.h)
ifdef USING_GPU
HEADER_FILES      += $(wildcard gpumatmul/*.h) $(wildcard gpumatmul/*.cuh)
endif

LIBS := -lcrypto -lssl -lcudart -lcuda
OBJ_INCLUDES := -I 'lib_eigen/' -I 'util/Miracl/' -I 'util/' -I 'gpumatmul/'
OBJ_INCLUDES += -I '$(CONDA_BASE)/include' -I '/usr/local/cuda-10.2/include'
BMR_INCLUDES := $(OBJ_INCLUDES), -L./ -L$(CONDA_BASE)/lib -L/usr/local/cuda-10.2/lib64
#########################################################################################
RUN_TYPE := localhost # RUN_TYPE {localhost, LAN or WAN} 
NETWORK := AlexNet # NETWORK {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
DATASET	:= ImageNet # Dataset {MNIST, CIFAR10, and ImageNet}
SECURITY:= Semi-honest # Security {Semi-honest or Malicious} 
#########################################################################################

all: $(BUILDDIR) BMRPassive.out

$(BUILDDIR):
	mkdir -p $@
	echo $(SRC_CPP_FILES)
	echo $(SRC_CU_FILES)
	echo $(OBJ_FILES)

BMRPassive.out: $(OBJ_FILES)
	$(CXX) $(FLAGS) -o $@ $(OBJ_FILES) $(BMR_INCLUDES) $(LIBS)

$(BUILDDIR)/%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(FLAGS) -c $< -o $@ $(OBJ_INCLUDES)

$(BUILDDIR)/%.o: %.cu $(HEADER_FILES)
	$(CXX) $(FLAGS) -c $< -o $@ $(OBJ_INCLUDES)

clean:
	rm -rf BMRPassive.out
	rm -rf $(BUILDDIR)

################################# Remote runs ##########################################
terminal: BMRPassive.out
	./BMRPassive.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC >/dev/null &
	./BMRPassive.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB >/dev/null &
	./BMRPassive.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC 
	@echo "Execution completed"

file: BMRPassive.out
	./BMRPassive.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC >/dev/null &
	./BMRPassive.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB >/dev/null &
	./BMRPassive.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC >output/3PC.txt
	@echo "Execution completed"

valg: BMRPassive.out 
	./BMRPassive.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC >/dev/null &
	./BMRPassive.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB >/dev/null &
	valgrind --tool=memcheck --leak-check=full --track-origins=yes --dsymutil=yes ./BMRPassive.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC

command: BMRPassive.out
	./BMRPassive.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC $(NETWORK) $(DATASET) $(SECURITY) >/dev/null &
	./BMRPassive.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB $(NETWORK) $(DATASET) $(SECURITY) >/dev/null &
	./BMRPassive.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC $(NETWORK) $(DATASET) $(SECURITY) 
	@echo "Execution completed"
#########################################################################################

zero: BMRPassive.out
	./BMRPassive.out 0 files/IP_$(RUN_TYPE) files/keyA files/keyAB files/keyAC

one: BMRPassive.out
	./BMRPassive.out 1 files/IP_$(RUN_TYPE) files/keyB files/keyBC files/keyAB

two: BMRPassive.out
	./BMRPassive.out 2 files/IP_$(RUN_TYPE) files/keyC files/keyAC files/keyBC

