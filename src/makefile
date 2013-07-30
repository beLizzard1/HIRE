CC=gcc -std=c99 
CUDACC=nvcc

Objs=main.o gpudistancegrid.o gpurefwavecalc.o gpufouriertransform.o writetoimage.o gpupropagate.o
BinaryName=gpuholography

CFLAGS=-O2 -Wall -L/usr/local/cuda/lib64 -I. -I/usr/local/cuda/lib64 -I/usr/local/cuda/include -I/usr/local/lib -lcuda -lcudart -lm -ltiff -lcufft -Wextra
LDFLAGS=-O2 -L/usr/local/lib  -L/usr/local/cuda/lib64 -lcudart -lcuda -ltiff -lm -lcufft
CUDACCFLAGS=-use_fast_math -arch=sm_20 -I. -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lineinfo

all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) -c $(CUDACCFLAGS) $<

$(BinaryName):
	$(CUDACC) -o $(BinaryName) $(Objs) $(CUDACCFLAGS) $(LDFLAGS)

clean:
	rm -f $(Objs) $(BinaryName) *.tiff
