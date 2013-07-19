#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>

__global__ void gpurefwavecalc(cuComplex *gpureferencewave, float *gpudistancegrid, float *gpuimage2f, float k, float width){

	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int offset;
	float sincomp, coscomp;

	offset = (bidy * width) + bidx;

	sincosf(k * gpudistancegrid[offset], &sincomp, &coscomp);

	gpureferencewave[offset].x = (gpuimage2f[offset] * coscomp);
	gpureferencewave[offset].y = (gpuimage2f[offset] * sincomp);


}

extern "C" int gpurefwavecalc(cuComplex *refwave, float *image2f, float *distancegrid, float k, unsigned int width, unsigned int height){

	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);

	cuComplex *gpureferencewave;
	float *gpudistancegrid, *gpuimage2f;

	cudaMalloc(&gpureferencewave, sizeof(cuComplex) * width * height);
	cudaMalloc(&gpudistancegrid, sizeof(float) * width * height);
	cudaMalloc(&gpuimage2f, sizeof(float) * width * height);

	cudaMemcpy(gpudistancegrid, distancegrid, sizeof(float) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuimage2f, image2f, sizeof(float) * width * height, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	printf("Allocating Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	gpurefwavecalc<<<numBlock, threadsPerBlock>>>(gpureferencewave, gpudistancegrid, gpuimage2f, k, width);
	cudaDeviceSynchronize();
	printf("ReferenceWaveCalcKernel errors(?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(refwave, gpureferencewave, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("ReferenceWaveCalcKernel errors(?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(gpureferencewave);
	cudaFree(gpudistancegrid);
	cudaFree(gpuimage2f);
	cudaDeviceReset();
	return(0);
}
