#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <cuda.h>

__global__ void gpudistancekernel(float *gpudistance, unsigned int width, unsigned int height, float pinholedist){
	unsigned int xcoord, ycoord, offset;
	xcoord = blockIdx.x;
	ycoord = blockIdx.y;

	float pixelsize;
	pixelsize = 6.45;

	offset = ( ycoord * width) + xcoord;
	float xcontrib, ycontrib;
	
	xcontrib = (xcoord - (width /2)) * pixelsize;
	ycontrib = (ycoord - (height /2)) * pixelsize;

	gpudistance[offset] = sqrtf( (xcontrib * xcontrib) + (ycontrib * ycontrib) + (pinholedist * pinholedist) );
	 
}


extern "C" int gpudistancegrid(float *dist2pinhole,float pinholedist, unsigned int width, unsigned int height){

	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);

	float *gpudistance;
	cudaMalloc(&gpudistance, sizeof(float) * width * height);
	
	gpudistancekernel<<<numBlock, threadsPerBlock>>>(gpudistance, width, height, pinholedist);
	cudaDeviceSynchronize();
	printf("Errors after running kernel(?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(dist2pinhole, gpudistance, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
        printf("Errors after copying data back to host(?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(gpudistance);
	cudaDeviceReset();

	return(0);
}
