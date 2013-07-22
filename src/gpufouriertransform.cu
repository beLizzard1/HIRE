#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

extern "C" int gpufouriertransform(cuComplex *original, cuComplex *transform, unsigned int width, unsigned int height){

	cufftHandle plan;
	cufftPlan2d(&plan,width, height, CUFFT_C2C);

	cuComplex *gpuoriginal, *gputransform;

	printf("Starting the gpu FFT\n");

	cudaMalloc(&gpuoriginal, sizeof(cuComplex) * width * height);
	cudaMalloc(&gputransform, sizeof(cuComplex) * width * height);

	cudaDeviceSynchronize();
        printf("Allocating Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(gpuoriginal, original, sizeof(cuComplex) * width * height, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
        printf("Copying Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));
	
	cufftExecC2C(plan,gpuoriginal,gputransform, -1);

	cudaMemcpy(transform, gputransform, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);		
	cudaDeviceSynchronize();
        printf("Copying back, Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

/*
	for(offset = 0; offset < (width * height); offset ++){
		printf("%g%+g\n", transform[offset].x, transform[offset].y);
	}

*/

	return(0);
}

extern "C" int floatfft(float* original,cuComplex* transform,unsigned int width, unsigned int height){


	cufftHandle plan;
	cufftPlan2d(&plan,width, height, CUFFT_R2C);

	cufftReal *gpuoriginal;
	cuComplex  *gputransform;


	cudaMalloc(&gpuoriginal, sizeof(cuComplex) * width * height);
	cudaMalloc(&gputransform, sizeof(cufftReal) * width * height);

	cudaDeviceSynchronize();

	cudaMemcpy(gpuoriginal, original, sizeof(cufftReal) * width * height, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cufftExecR2C(plan, gpuoriginal, gputransform);

	cudaMemcpy(transform, gputransform, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);


	return(0);

}
