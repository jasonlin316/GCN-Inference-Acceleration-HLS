#######################################################################################
.PHONY: help
help:
	@echo "Makefile Usage:"
	@echo "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command to generate the design for specified Target and Device."
	@echo ""
	@echo "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command to run for specified Target."
	@echo ""
	@echo "  make exe "
	@echo "      Command to generate host."
	@echo ""
	@echo "  make xclbin "
	@echo "      Command to generate hardware platform files(xo,xclbin)."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""
#######################################################################################
TARGET := sw_emu
PLATFORM := /opt/xilinx/platforms/xilinx_u200_xdma_201830_2/xilinx_u200_xdma_201830_2.xpfm
HOST_EXE := host
XO := spdmm.$(TARGET).$(PLATFORM).xo
XO2:= mmult.$(TARGET).$(PLATFORM).xo
XCLBIN := GCN.$(TARGET).$(PLATFORM).xclbin

# Host building global settings
CXXFLAGS := -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -Wall -O0 -g -std=c++11 -L$(XILINX_XRT)/lib/ -lpthread -lrt -lstdc++
CXXFLAGS2 := -lOpenCL

# Kernel compiler & linker global settings
KRNL_COMPILE_OPTS := -t $(TARGET) --config gcn.cfg
MM_KRNL_COMPILE_OPTS := -t $(TARGET) --config mmult.cfg
KRNL_LINK_OPTS := -t $(TARGET) --config design_2.cfg

BINARY_CONTAINERS += GCN.$(TARGET).$(PLATFORM).xclbin
BINARY_CONTAINER_matrix_ops_OBJS += spdmm.$(TARGET).$(PLATFORM).xo
BINARY_CONTAINER_matrix_ops_OBJS += mmult.$(TARGET).$(PLATFORM).xo

build:  $(XO) $(XO2)  $(XCLBIN) $(HOST_EXE) emconfig 

exe: $(HOST_EXE)

xclbin: $(XO) $(XO2) $(XCLBIN) 

SRC_REPO := ../src

HOST_SRC_CPP := $(SRC_REPO)/host.cpp

KERNEL_SRC_CPP := $(SRC_REPO)/spdmm.cpp
KERNEL_SRC_CPP += $(SRC_REPO)/mmult.cpp

# Building kernel
spdmm.$(TARGET).$(PLATFORM).xo: $(SRC_REPO)/spdmm.cpp
	v++ $(KRNL_COMPILE_OPTS) -c -k spdmm -I'$(<D)' -o'$@' '$<'

mmult.$(TARGET).$(PLATFORM).xo: $(SRC_REPO)/mmult.cpp
	v++ $(MM_KRNL_COMPILE_OPTS) -c -k mmult -I'$(<D)' -o'$@' '$<'

GCN.$(TARGET).$(PLATFORM).xclbin: $(BINARY_CONTAINER_matrix_ops_OBJS)
	v++ $(KRNL_LINK_OPTS) -l -g -R2 -o'$@' $(+)

# Building Host
$(HOST_EXE): $(HOST_SRC_CPP)
	g++ $(CXXFLAGS) $(HOST_SRC_CPP) -o $@ $(CXXFLAGS2)


.PHONY: emconfig
emconfig:
	emconfigutil --platform $(PLATFORM)

.PHONY: run
run: build
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	XCL_EMULATION_MODE=$(TARGET) ./$(HOST_EXE) $(XCLBIN)
else
	./$(HOST_EXE) $(XCLBIN)
endif

# Cleaning stuff
.PHONY: clean
clean:
	rm -f $(HOST_EXE) *spdmm.$(TARGET).$(PLATFORM).* *.log *.json *.xo
