#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <cuda.h>

__global__ void gpudistancekernel(float *gpudistance, unsigned int width, unsigned int height, float pinholedist){

	int xcoord = blockIdx.x * blockDim.x + threadIdx.x;
	int ycoord = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = ycoord * width + xcoord;

	float pixelsize;
	pixelsize = 6.45;

	offset = ( ycoord * width) + xcoord;
	float xcontrib, ycontrib;
	
	xcontrib = (xcoord - (width /2)) * pixelsize;
	ycontrib = (ycoord - (height /2)) * pixelsize;

	gpudistance[offset] = sqrtf( (xcontrib * xcontrib) + (ycontrib * ycontrib) + (pinholedist * pinholedist) );
	 
}


extern "C" int gpudistancegrid(float *dist2pinhole,float pinholedist, unsigned int width, unsigned int height){

	dim3 gridinblocks, blockinthreads;
	unsigned int totalthreadsperblock;

	blockinthreads.x = width/87;
	blockinthreads.y = height/65;
	blockinthreads.z = 1;
	totalthreadsperblock = blockinthreads.x * blockinthreads.y * blockinthreads.z;
	if(totalthreadsperblock > 1024){
		printf("Error in your thread config.\n");
		return(1);
	}
	gridinblocks.x = (width / blockinthreads.x);
	gridinblocks.y = (height / blockinthreads.y);

	float *gpudistance;
	cudaMalloc(&gpudistance, sizeof(float) * width * height);
	
	gpudistancekernel<<<gridinblocks, blockinthreads>>>(gpudistance, width, height, pinholedist);
	cudaDeviceSynchronize();
	//printf("Errors after running kernel(?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(dist2pinhole, gpudistance, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
        //printf("Errors after copying data back to host(?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(gpudistance);
	cudaDeviceReset();

	return(0);
}
