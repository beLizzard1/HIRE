CC=gcc
CUDACC=nvcc

Objs=main.o fftcuda.o tiffmanager.o

BinaryName=gpuholography

CFLAGS=-O2 -g -Wall -Werror -pedantic -Wextra
LDFLAGS=-Wall -Werror -pedantic -Wextra



all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) $(CUDACCFLAGS) -c $<

$(BinaryName):
	$(CC) -o $(BinaryName) $(LDFLAGS) $(Objs)

clean:
	rm -f $(Objs) $(BinaryName)
