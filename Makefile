BINARY=piranha
DEBUG_BINARY=piranha-debug
BUILD_DIR=build
DEBUG_DIR=debug

CUDA_VERSION=10.0
CUTLASS_PATH=ext/cutlass

CXX=nvcc
FLAGS := -Xcompiler="-O3,-w,-std=c++14,-pthread,-msse4.1,-maes,-msse2,-mpclmul,-fpermissive,-fpic,-pthread" -Xcudafe "--diag_suppress=declared_but_not_referenced"
DEBUG_FLAGS := -Xcompiler="-O0,-g,-w,-std=c++14,-pthread,-msse4.1,-maes,-msse2,-mpclmul,-fpermissive,-fpic,-pthread" -Xcudafe "--diag_suppress=declared_but_not_referenced"

PIRANHA_FLAGS :=

VPATH             := src/:src/gpu:src/nn:src/mpc:src/util:src/test
SRC_CPP_FILES     := $(wildcard src/*.cpp src/**/*.cpp)
SRC_CU_FILES      := $(wildcard src/*.cu src/**/*.cu)
OBJ_FILES         := $(addprefix $(BUILD_DIR)/, $(notdir $(SRC_CPP_FILES:.cpp=.o)))
OBJ_FILES         += $(addprefix $(BUILD_DIR)/, $(notdir $(SRC_CU_FILES:.cu=.o)))
DEBUG_OBJ_FILES   := $(addprefix $(DEBUG_DIR)/, $(notdir $(SRC_CPP_FILES:.cpp=.o)))
DEBUG_OBJ_FILES   += $(addprefix $(DEBUG_DIR)/, $(notdir $(SRC_CU_FILES:.cu=.o)))
HEADER_FILES      := $(wildcard src/*.h src/**/*.h src/*.cuh src/**/*.cuh src/*.inl src/**/*.inl)

LIBS := -lcrypto -lssl -lcudart -lcuda -lgtest -lcublas
OBJ_INCLUDES := -I '/usr/local/cuda-$(CUDA_VERSION)/include' -I '$(CUTLASS_PATH)/include' -I '$(CUTLASS_PATH)/tools/util/include' -I 'include'
INCLUDES := $(OBJ_INCLUDES), -L./ -L/usr/local/cuda-$(CUDA_VERSION)/lib64 -L$(CUTLASS_PATH)/build/tools/library

TEST :=

################################# OPTIONS ###############################################
CONFIG_FILE := config.json
#########################################################################################

all: $(BINARY)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BINARY): $(BUILD_DIR) $(OBJ_FILES) 
	$(CXX) $(FLAGS) $(PIRANHA_FLAGS) -o $@ $(OBJ_FILES) $(INCLUDES) $(LIBS)

$(BUILD_DIR)/%.o: %.cpp $(HEADER_FILES)
	$(CXX) -dc $(FLAGS) $(PIRANHA_FLAGS) -c $< -o $@ $(OBJ_INCLUDES)

$(BUILD_DIR)/%.o: %.cu $(HEADER_FILES)
	$(CXX) -dc $(FLAGS) -Xcompiler="$(PIRANHA_FLAGS)" -c $< -o $@ $(OBJ_INCLUDES)

$(DEBUG_DIR):
	mkdir -p $(DEBUG_DIR)

$(DEBUG_BINARY): $(DEBUG_DIR) $(DEBUG_OBJ_FILES)
	$(CXX) $(DEBUG_FLAGS) $(PIRANHA_FLAGS) -o $@ $(DEBUG_OBJ_FILES) $(INCLUDES) $(LIBS)

$(DEBUG_DIR)/%.o: %.cpp $(HEADER_FILES)
	$(CXX) -dc $(DEBUG_FLAGS) $(PIRANHA_FLAGS) -c $< -o $@ $(OBJ_INCLUDES)

$(DEBUG_DIR)/%.o: %.cu $(HEADER_FILES)
	$(CXX) -dc $(DEBUG_FLAGS) -Xcompiler="$(PIRANHA_FLAGS)" -c $< -o $@ $(OBJ_INCLUDES)

clean:
	rm -rf $(BINARY)
	rm -rf $(BUILD_DIR)
	rm -rf $(DEBUG_DIR)

################################# Remote runs ##########################################

run: $(BINARY)
	#@./$(BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#@./$(BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#@./$(BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#@./$(BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST)
	@echo "Execution completed"

gdb: $(DEBUG_BINARY)
	#./$(DEBUG_BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#cuda-gdb --args ./$(DEBUG_BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(DEBUG_BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	cuda-gdb --args ./$(DEBUG_BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST)
	@echo "Execution completed"

gdb-one: $(DEBUG_BINARY)
	#./$(DEBUG_BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#cuda-gdb --args ./$(DEBUG_BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(DEBUG_BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	cuda-gdb --args ./$(DEBUG_BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST)
	@echo "Execution completed"

gdb-two: $(DEBUG_BINARY)
	#./$(DEBUG_BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#cuda-gdb --args ./$(DEBUG_BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(DEBUG_BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	cuda-gdb --args ./$(DEBUG_BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST)
	@echo "Execution completed"

gdb-three: $(DEBUG_BINARY)
	#./$(DEBUG_BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#cuda-gdb --args ./$(DEBUG_BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(DEBUG_BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	cuda-gdb --args ./$(DEBUG_BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST)
	@echo "Execution completed"

memcheck: $(DEBUG_BINARY)
	#./$(DEBUG_BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#./$(DEBUG_BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	#cuda-memcheck ./$(DEBUG_BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(DEBUG_BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	@./$(DEBUG_BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST) >/dev/null 2>&1 &
	cuda-memcheck ./$(DEBUG_BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST)	
	@echo "Execution completed"

#########################################################################################

party: $(BINARY)
	@./$(BINARY) $(PARTY_NUM) $(CONFIG_FILE) --gtest_filter=$(TEST)

zero: $(BINARY)
	#@./$(BINARY) 0 files/IP_$(RUN_TYPE) files/keys/key0 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(BINARY) 0 $(CONFIG_FILE) --gtest_filter=$(TEST)
	
one: $(BINARY)
	#@./$(BINARY) 1 files/IP_$(RUN_TYPE) files/keys/key1 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(BINARY) 1 $(CONFIG_FILE) --gtest_filter=$(TEST)

two: $(BINARY)
	#@./$(BINARY) 2 files/IP_$(RUN_TYPE) files/keys/key2 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(BINARY) 2 $(CONFIG_FILE) --gtest_filter=$(TEST)

three: $(BINARY)
	#@./$(BINARY) 3 files/IP_$(RUN_TYPE) files/keys/key3 $(NETWORK) $(LR_FILE) $(SEED) $(RUN_NAME) $(PRELOAD) --gtest_filter=$(TEST)
	@./$(BINARY) 3 $(CONFIG_FILE) --gtest_filter=$(TEST)


