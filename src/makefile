CC=gcc -ansi
CUDACC=nvcc

Objs=main.o fftcuda.o datamanager.o

BinaryName=gpuholography

CFLAGS=-O2 -g -Wall -Werror -Wextra -I. -I/usr/local/cuda/include -lcudart -ltiff -lm
LDFLAGS=-Wall -O2 -g -L/usr/local/lib -L/usr/local/cuda/lib64 -lcudart -lcudart -lcuda -lm -ltiff
CUDACCFLAGS=-O2 -g -I. -I/usr/local/cuda/include


all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) $(CUDACCFLAGS) -c $<

$(BinaryName):
	g++ -o $(BinaryName) $(Objs) $(LDFLAGS)

clean:
	rm -f $(Objs) $(BinaryName)
