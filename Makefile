OS=$(shell uname)
CUDA_HOME=/usr/local/cuda
ifeq ($(OS), Darwin)
	NO_NVML=1
else
	NO_NVML=0
endif

CXXFLAGS+=-I$(CUDA_HOME)/include
LFLAGS=-lcudart_static -lpthread -ldl

ifeq ($(OS), Darwin)
	CXXFLAGS+=-L$(CUDA_HOME)/lib/
else
	CXXFLAGS+=-L$(CUDA_HOME)/lib64/ -L/usr/lib/nvidia-$(NVIDIA_DRIVER_VERSION)/
endif

ifneq ($(NO_NVML), 1)
	NVIDIA_DRIVER_VERSION=$(shell cat /proc/driver/nvidia/version | sed -e 2d | sed -E 's,.* ([0-9]*)\.([0-9]*) .*,\1,')
	CXXFLAGS+=-L/usr/lib/nvidia-$(NVIDIA_DRIVER_VERSION)/
	LFLAGS+=-lrt -lnvidia-ml
endif

cuda-smi: cuda-smi.cpp nvml.h
	$(CXX) $(CXXFLAGS) $< $(LFLAGS) -o $@ -DNO_NVML=$(NO_NVML)

clean:
	$(RM) cuda-smi
