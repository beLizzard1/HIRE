CC=gcc -std=c99 
CUDACC=nvcc

Objs=main.o gpudistancegrid.o gpurefwavecalc.o gpufouriertransform.o writetoimage.o gpupropagate.o
BinaryName=gpuholography

CFLAGS=-O2 -Wall -L/usr/local/cuda/lib64 -I. -I/usr/local/cuda/lib64 -I/usr/local/cuda/include -I/usr/local/lib -I/usr/lib -lcuda -lcudart -lm -ltiff -lcufft -lglfw -lGL -Wextra
LDFLAGS=-O2 -L/usr/local/lib  -L/usr/local/cuda/lib64 -lcudart -lcuda -ltiff -lm -lcufft -lglfw -lGL 
CUDACCFLAGS=-use_fast_math -arch=sm_20 -I. -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lineinfo

SDL_CFLAGS := $(shell sdl-config --cflags)
SDL_LDFLAGS := $(shell sdl-config --libs)

all: $(Objs) $(BinaryName)

%.o:	%.c
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $<

%.o: %.cu
	$(CUDACC) -c $(CUDACCFLAGS) $(SDL_CFLAGS) $<

$(BinaryName):
	$(CUDACC) -o $(BinaryName) $(Objs) $(CUDACCFLAGS) $(SDL_CFLAGS) $(SDL_LDFLAGS) $(LDFLAGS)

clean:
	rm -f $(Objs) $(BinaryName) *.tiff
