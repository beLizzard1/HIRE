CC=gcc -ansi
CUDACC=nvcc

Objs=main.o subtractimages.o gpudistancegrid.o gpurefwavecalc.o gpusubdivref.o gpufouriertransform.o writetoimage.o
BinaryName=gpuholography

CFLAGS=-O2 -Wall -L/usr/local/cuda/lib64 -I. -I/usr/local/cuda/include -I/usr/local/lib -lcuda -lcudart -lm -ltiff -lcufft
LDFLAGS=-O2 -L/usr/local/lib  -L/usr/local/cuda/lib64 -lcudart -lcuda -ltiff -lm -lcufft
CUDACCFLAGS=-use_fast_math -arch=sm_20 -I. -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -G -g -lineinfo


all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) -c $(CUDACCFLAGS) $<

$(BinaryName):
	$(CUDACC) -o $(BinaryName) $(Objs) $(LDFLAGS) $(CUDACCFLAGS)

clean:
	rm -f $(Objs) $(BinaryName)
