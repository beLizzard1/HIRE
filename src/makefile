CC=gcc -ansi
CUDACC=nvcc

Objs=main.o rasterarray.o

BinaryName=gpuholography

CFLAGS=-O2 -g -Wall -Werror -Wextra -std=c99 -I. -I/usr/local/cuda/include -I/usr/local/lib -lcudart -lm -ltiff
LDFLAGS=-Wall -O2 -g -L/usr/local/lib -std=c99 -L/usr/local/cuda/lib64 -lcudart -lcuda -ltiff
CUDACCFLAGS=-O2 -g -I. -I/usr/local/cuda/include


all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) $(CUDACCFLAGS) -c $<

$(BinaryName):
	$(CC) -o $(BinaryName) $(Objs) $(LDFLAGS)

clean:
	rm -f $(Objs) $(BinaryName)
