
CXX=g++
SRC_CPP_FILES     := $(wildcard src/*.cpp)
OBJ_CPP_FILES     := $(wildcard util/*.cpp)
OBJ_FILES    	  := $(patsubst src/%.cpp, src/%.o,$(SRC_CPP_FILES))
OBJ_FILES    	  += $(patsubst util/%.cpp, util/%.o,$(OBJ_CPP_FILES))
HEADER_FILES       = $(wildcard src/*.h)

# FLAGS := -static -g -O0 -w -std=c++11 -pthread -msse4.1 -maes -msse2 -mpclmul -fpermissive -fpic
FLAGS := -O3 -w -std=c++11 -pthread -msse4.1 -maes -msse2 -mpclmul -fpermissive -fpic
LIBS := -lcrypto -lssl
OBJ_INCLUDES := -I 'lib_eigen/' -I 'util/Miracl/' -I 'util/'
BMR_INCLUDES := $($(OBJ_INCLUDES), -L./)
#########################################################################################
RUN_TYPE := localhost # RUN_TYPE {localhost, LAN or WAN} 
NETWORK := SecureML # NETWORK {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
DATASET	:= MNIST # Dataset {MNIST, CIFAR10, and ImageNet}
SECURITY:= Semi-honest # Security {Semi-honest or Malicious} 
#########################################################################################


all: BMRPassive.out

BMRPassive.out: $(OBJ_FILES)
	g++ $(FLAGS) -o $@ $(OBJ_FILES) $(BMR_INCLUDES) $(LIBS)
%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(FLAGS) -c $< -o $@ $(OBJ_INCLUDES)
clean:
	rm -rf BMRPassive.out
	rm -rf src/*.o util/*.o


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