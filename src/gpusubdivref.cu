#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>

#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>

__global__ void reducedhologramkernel(cuComplex *gpureducedhologram, cuComplex *gpureferencewave, float *gpusubtractedimage, unsigned int width){

	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int offset;
	offset = (bidy * width) + bidx;

	gpureducedhologram[offset].x = gpusubtractedimage[offset] * ( 1 * gpureferencewave[offset].x);
	gpureducedhologram[offset].y = gpusubtractedimage[offset] * (-1 * gpureferencewave[offset].y);

}


extern "C" int gpusubdivref(cuComplex *reducedhologram, float *subtractedimage, cuComplex *referencewave,unsigned int width, unsigned int height){

	        dim3 threadsPerBlock(1,1);
	        dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);

		cuComplex *gpureducedhologram, *gpureferencewave;
		float *gpusubtractedimage;

		cudaMalloc(&gpureducedhologram, sizeof(cuComplex) * width * height);
		cudaMalloc(&gpureferencewave, sizeof(cuComplex) * width * height);
		cudaMalloc(&gpusubtractedimage, sizeof(float) * width * height);

		cudaMemcpy(gpureferencewave, referencewave, sizeof(cuComplex) * width * height, cudaMemcpyHostToDevice);
		cudaMemcpy(gpusubtractedimage, subtractedimage, sizeof(float) * width * height, cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
		printf("ReducedHologram errors allocating and copying memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	reducedhologramkernel<<<numBlock, threadsPerBlock>>>(gpureducedhologram, gpureferencewave, gpusubtractedimage, width);

		cudaDeviceSynchronize();
		printf("Running the kernel errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

		cudaMemcpy(reducedhologram, gpureducedhologram, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);	
		cudaDeviceSynchronize();
		printf("Copying result from GPU, errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

		cudaFree(gpureducedhologram);
		cudaFree(gpureferencewave);
		cudaFree(gpusubtractedimage);

		cudaDeviceReset();
	

	return(0);

}
