#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>

__device__ float distcalc(unsigned int bidx, unsigned int bidy, unsigned int width, unsigned int height, float pinholedist, float pixelsize){

	float xcon, ycon, Rxy;
	xcon = (((float)bidx - (float)width/2 - 80) * pixelsize);
	ycon = (((float)bidy - (float)height/2) * pixelsize);

	Rxy = sqrtf((xcon * xcon) + (ycon * ycon) + (pinholedist * pinholedist));
	return(Rxy);
}

__global__ void gpurefwavecalckernel(cuComplex *gpureferencewave, unsigned int width, unsigned int height, float pinholedist, float k, float pixelsize){

	int bidx = blockIdx.x * blockDim.x + threadIdx.x;
	int bidy = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = (bidy * width) + bidx;
	float sin, cos;
	float Rxy;
	Rxy = distcalc(bidx,bidy, width, height, pinholedist, pixelsize);

	sincosf(k * Rxy, &sin, &cos);

	gpureferencewave[offset].x = cos;
	gpureferencewave[offset].y = sin;

}

extern "C" int gpurefwavecalc(cuComplex *refwave, unsigned int width, unsigned int height, float pinholedist, float k, float pixelsize){

	dim3 threadsPerBlock(16,16);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);

	cuComplex *gpureferencewave;
	cudaMalloc(&gpureferencewave, sizeof(cuComplex) * width * height);

	cudaDeviceSynchronize();
	printf("Allocating Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	gpurefwavecalckernel<<<numBlock, threadsPerBlock>>>(gpureferencewave, width, height, pinholedist, k, pixelsize);

	cudaMemcpy(refwave, gpureferencewave, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
        printf("Copying results from GPU errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(gpureferencewave);
	return(0);
}
