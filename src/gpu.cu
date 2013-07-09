#include <complex.h>
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void sqrtkernel(float* gpudata, float* gpuresult, unsigned int width){

	int tidx = blockIdx.x;
	int tidy = blockIdx.y;
	int offset;

	offset = (tidy * width)+ tidx;
	gpuresult[offset] = sqrtf(gpudata[offset]);

}

__global__ void distancekernel(float* gpudistance, unsigned int width, unsigned int height, float realz){
	int xcoord = blockIdx.x;
	int ycoord = blockIdx.y;
	int offset;
	
	offset = (ycoord * width) + xcoord;
		
	gpudistance[offset] = sqrtf( pow(((xcoord - 696)*6.45),2) + pow(((ycoord - 520)*6.45),2) + pow(realz,2) );

}

__global__ void wavecalckernel(cuComplex** refwave, float* data, float* distance, float k, unsigned int width){

	int xcoord = blockIdx.x;
	int ycoord = blockIdx.y;
	int offset;
	float sinvalue, cosvalue;

	offset = (ycoord * width) + xcoord;
	sincosf(k * distance[offset], &sinvalue, &cosvalue);
	
	refwave[offset]->x = data[offset] * cosvalue;
	refwave[offset]->y = data[offset] * sinvalue;
}

extern "C" int referencephase(float *data, unsigned int width, unsigned int height){
	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	float realz, k;
	float *distance, *gpudistance, *gpudata;
	int offset;

	k = 2 * M_PI / 0.780241;

	realz = 43048; /* Distance from pinhold in Z axis */

	distance = (float*)malloc(width * height * sizeof(float));
	cudaMalloc(&gpudistance, sizeof(cuComplex) * (width * height));
	distancekernel<<<numBlock, threadsPerBlock >>>(gpudistance, width, height, realz);	
	cudaMemcpy(distance, gpudistance, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cuComplex *gpurefwave, *refwave;
	cudaMalloc(&gpurefwave, sizeof(cuComplex) * (width * height));
	refwave = (cuComplex *)calloc(width * height, sizeof(cuComplex));
	cudaMalloc(&gpudata, sizeof(float) * width * height);
	cudaMemcpy(gpudata,data, sizeof(float) * (width * height),cudaMemcpyHostToDevice);

	wavecalckernel<<<numBlock, threadsPerBlock >>>(&gpurefwave, gpudata, gpudistance, k, width);
	cudaMemcpy(gpurefwave,refwave,sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);
	
	for(offset = 0; offset < (width * height); offset++){
		printf("%g + %gi\n", refwave[offset].x, refwave[offset].y);
	}

	cudaFree(gpurefwave);
	return(0);

}

extern "C" int gpusqrt(float *data, unsigned int width, unsigned int height){
	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	float *gpudata, *gpuresult;	

	cudaMalloc(&gpudata, sizeof(float) * (width * height));
	cudaMalloc(&gpuresult, sizeof(float) * (width * height));	
	cudaMemcpy(gpudata, data, sizeof(float) * (width * height), cudaMemcpyHostToDevice);

	sqrtkernel<<<numBlock, threadsPerBlock >>>(gpudata,gpuresult, width);
	cudaFree(gpudata);

	cudaMemcpy(data, gpuresult, sizeof(float) * (width * height), cudaMemcpyDeviceToHost);
	
	cudaFree(gpuresult);
	return(0);
}
