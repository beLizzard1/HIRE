#include <complex.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define PI 3.1415926535

__global__ void kspacedistkernel(float* kspacedist, unsigned int width, unsigned int height){

	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int offset;

	offset = (bidx * width) + bidy;

	float xkspace, ykspace;

	xkspace = ((2 * PI ) / 6.45 ) / (width);
	ykspace = ((2 * PI ) / 6.45 ) / (height);

	kspacedist[offset] = sqrtf( powf(xkspace*(bidx - 696),2) + powf(ykspace*(bidy - 520),2));

}

extern "C" int gpudftwithprop(cuComplex *reducedhologram, unsigned int width, unsigned int height, float startz, float endz){

	printf("Starting Discrete Fourier Transform and Propagation Stuff\n");
	dim3 threadsPerBlock(1,1);
        dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	cuComplex *gpureducedhologram, *gputransformed, *transformed;		
	float *kspacedist, *temp;
	int offset;
/*
	for(offset = 0; offset < width * height; offset++){
		printf("Reduced Hologram: %f%+fi\n",reducedhologram[offset].x, reducedhologram[offset].y);
	}
*/
	transformed = (cufftComplex *)malloc(sizeof(cuComplex)*width*height);

	cudaMalloc(&gpureducedhologram, sizeof(cuComplex) * width * height);
	cudaMalloc(&gputransformed, sizeof(cuComplex) * width * height);
	cudaMemcpy(gpureducedhologram, reducedhologram, sizeof(cuComplex) * width * height, cudaMemcpyHostToDevice);
	
	printf("Allocating memory on the GPU so that the cuFFT library can access it during the operation\n");
	
	cufftHandle plan;
	cufftPlan2d(&plan, width, height, CUFFT_C2C);

	cufftExecC2C(plan,gpureducedhologram,gputransformed, -1);

	cudaMemcpy(transformed, gputransformed, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);
/*
	for(offset = 0; offset < (width * height); offset++){
		printf("%f %+fi\n", transformed[offset].x, transformed[offset].y);
	} 
*/
	cudaFree(gpureducedhologram);
	cudaFree(gputransformed);	
	
	cudaMalloc(&kspacedist, sizeof(float) * width * height);
	temp = (float *)malloc(sizeof(float) * width * height);

	kspacedistkernel<<<numBlock, threadsPerBlock>>>(kspacedist,width, height);

	cudaMemcpy(temp, kspacedist, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

/*	for(offset = 0; offset < (width * height); offset ++){
		printf("k-space dist: %f\n", temp[offset]);
	
	}*/

	cufftDestroy(plan);

	return(0);
}
