#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>

#include "writetoimage.h"

__global__ void gpupropagatekernel(cuComplex *gpuhologram, cuComplex *gpupropagated, unsigned int width, unsigned int height, float k, float pixelsize, float propdist){

int xcoord = blockIdx.x * blockDim.x + threadIdx.x;
int ycoord = blockIdx.y * blockDim.y + threadIdx.y;
int offset = ycoord * width + xcoord;

	if(offset > (width * height)){
		printf("Something went wrong with the kernel sizing \n");
		asm("trap;");
	}

	pixelsize = 6.45;
	
	float kpixelsizex, kpixelsizey;
	kpixelsizex = (2 * M_PI) / (pixelsize * width);
	kpixelsizey = (2 * M_PI) / (pixelsize * height);
	
	float kxij, kyij;
	kxij =  ((float)xcoord - ((float)width / 2)) * kpixelsizex;
	kyij =  ((float)ycoord - ((float)height /2)) * kpixelsizey;

	float realcomp, imagcomp, exponent;
	float kfactor;
	kfactor = -1 * sqrtf( ((k) * (k)) - ((kxij) * (kxij)) - ((kyij) * (kyij)));
	exponent = kfactor * propdist;
	sincosf(exponent, &imagcomp, &realcomp);

	gpupropagated[offset].x = (gpuhologram[offset].x * realcomp) - (gpuhologram[offset].y * imagcomp);
	gpupropagated[offset].y = (gpuhologram[offset].y * realcomp) + (gpuhologram[offset].x * imagcomp);

	__syncthreads();

}


extern "C" int gpupropagate(cuComplex *hologram, cuComplex *propagated, unsigned int width, unsigned int height, float k, float pixelsize, float propdist, tsize_t scanlinesize, int scalefactor){

	dim3 gridinblocks, blockinthreads;
	unsigned int totalthreadsperblock;
	gridinblocks.x = 87 * scalefactor;
	gridinblocks.y = 65 * scalefactor;
	gridinblocks.z = 1;
	blockinthreads.x = width/gridinblocks.x;
	blockinthreads.y = height/gridinblocks.y;
	blockinthreads.z = 1;
	totalthreadsperblock = blockinthreads.x * blockinthreads.y * blockinthreads.z;
	if(totalthreadsperblock > 1024){
		printf("Error in your thread config.\n");
	}
	cuComplex *gpuhologram, *gpuprophologram;

	cudaMalloc(&gpuhologram, sizeof(cuComplex) * (width * height));
	cudaMemcpy(gpuhologram, hologram, sizeof(cuComplex) * (width * height), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuprophologram, sizeof(cuComplex) * (width * height));

        cudaDeviceSynchronize();
//        printf("Allocating and Copying Memory onto device Errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	/* Need to be sensible about the use of our kernel it will do threadsperblock calculations per launch we then need to cycle the inital start point till we have a complete set */

	gpupropagatekernel<<<gridinblocks, blockinthreads>>>(gpuhologram, gpuprophologram, width, height, k, pixelsize, propdist);

	cudaDeviceSynchronize();
        //printf("Kernel Operation Errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(propagated, gpuprophologram, sizeof(cuComplex) * (width * height), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //printf("Result return Errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(gpuprophologram);
	cudaFree(gpuhologram);
/*	
	for(unsigned int offset = 0; offset < (width * height); offset++){
		printf("Propagated real: %f, imag: %fi\n", propagated[offset].x, propagated[offset].y);
	}
*/

	return(0);
}
