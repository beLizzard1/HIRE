CC=gcc -ansi
CUDACC=nvcc

Objs=main.o matrix.o gpu.o tiffman.o

BinaryName=gpuholography

CFLAGS=-O2 -Wall -Werror -L/usr/local/cuda/lib64 -I. -I/usr/local/cuda/include -I/usr/local/lib -lcuda -lcudart -lm -ltiff
LDFLAGS=-O2 -L/usr/local/lib  -L/usr/local/cuda/lib64 -lcudart -lcuda -ltiff -lm
CUDACCFLAGS=-use_fast_math -arch=sm_20 --ptxas-options=-v -I. -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -G -pg -g -lineinfo


all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) -c $(CUDACCFLAGS) $<

$(BinaryName):
	$(CUDACC) -o $(BinaryName) $(Objs) $(LDFLAGS) $(CUDACCFLAGS)

clean:
	rm -f $(Objs) $(BinaryName)
