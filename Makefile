CUDA_HOME=/usr/local/cuda/

CXXFLAGS+=-I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64/

cuda-smi: cuda-smi.cpp nvml.h
	$(CXX) $(CXXFLAGS) $< -lnvidia-ml -lcudart_static -lpthread -ldl -lrt -o $@

clean:
	$(RM) cuda-smi
