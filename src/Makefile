CC=gcc -std=c99 
CUDACC=nvcc
Objs=main.o gpudistancegrid.o gpurefwavecalc.o gpufouriertransform.o writetoimage.o gpupropagate.o
BinaryName=gpuholography

CFLAGS=-O2 -g -O0 -Wall -L/usr/local/cuda-5.5/lib64 -I. -I/usr/local/cuda-5.5/lib64 -I/usr/local/cuda-5.5/include -I/usr/local/lib -I/usr/lib -lcuda -lcudart -lm -ltiff -lcufft -lglfw -lGL -Wextra
LDFLAGS=-O2 -L/usr/local/lib  -L/usr/local/cuda-5.5/lib64 -lcudart -lcuda -ltiff -lm -lcufft -lglfw -lGL 
CUDACCFLAGS=-use_fast_math -arch=sm_20 -I. -I/usr/local/cuda-5.5/include -I/usr/local/cuda-5.5/lib64 -lineinfo

all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) -c $(CUDACCFLAGS) $<

$(BinaryName):
	$(CUDACC) -o $(BinaryName) $(Objs) $(CUDACCFLAGS) $(LDFLAGS)

x86_64clean:
	rm -f $(Objs) $(BinaryName)
