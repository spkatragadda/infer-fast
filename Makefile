CXX      = g++
CXXFLAGS = -std=c++17 -Wall -O2
COMMON   = model.o bpe.o json.o dataset.o

# GPU trainer. Turing (GTX 16xx / RTX 20xx) is sm_75.
NVCC       = /usr/local/cuda/bin/nvcc
NVCCFLAGS  = -std=c++17 -O3 -arch=sm_75
CUDA_LIBS  = -lcublas

# The CPU tools are the default build: inference never needs CUDA.
all: train infer finetune

train: train.o $(COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

infer: infer.o $(COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

finetune: finetune.o $(COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Built only on request, so a machine without nvcc still builds everything else
gpu: train_cuda

train_cuda: train_cuda.cu $(COMMON)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LIBS)

train.o: train.cpp model.h bpe.h dataset.h
infer.o: infer.cpp model.h bpe.h
finetune.o: finetune.cpp model.h bpe.h json.h dataset.h
model.o: model.cpp model.h bpe.h io.h
bpe.o: bpe.cpp bpe.h io.h
json.o: json.cpp json.h
dataset.o: dataset.cpp dataset.h bpe.h json.h

clean:
	rm -f train infer finetune train_cuda *.o

.PHONY: all gpu clean
