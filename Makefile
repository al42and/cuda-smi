CUDA_HOME=/usr/local/cuda/

NVIDIA_DRIVER_VERSION=$(shell cat /proc/driver/nvidia/version | sed -e 2d | sed -E 's,.* ([0-9]*)\.([0-9]*) .*,\1,')

CXXFLAGS+=-I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64/ -L/usr/lib/nvidia-$(NVIDIA_DRIVER_VERSION)/

cuda-smi: cuda-smi.cpp nvml.h
	$(CXX) $(CXXFLAGS) $< -lnvidia-ml -lcudart_static -lpthread -ldl -lrt -o $@

clean:
	$(RM) cuda-smi
