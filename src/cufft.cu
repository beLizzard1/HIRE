#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime_api.h>

__global__ void FFT(short* gpumatrix, unsigned int width, unsigned int height){

	int x,y, offset;
	double sqrt1, number;

	for(x = 0; x < width; x++){
		for(y = 0; y < height; y++){
			offset = ( y * width) + x;
			number = (double)gpumatrix[offset];
			sqrt1 = sqrt(number);
			printf("%lf\n", sqrt1);
			gpumatrix[offset] = sqrt1;

		}
	}


}


extern "C" void gpucalculation(unsigned int width, unsigned int height, short int **hostmatrix){

	//Weird GPU stuff (need to figure it out) 
	dim3 blockDim(16,16);
	dim3 gridDim(width/blockDim.x, height/ blockDim.y);
	unsigned int x,y,arrayint;	
	short *gpumatrix, *temporarymatrix;
	unsigned int numberofpixels;
/*
	for(x = 0; x < width; x++){
		for(y=0; y < height; y++){
			printf("Intensity Hostmatrix: %+d \n",hostmatrix[x][y]);
		}
	}
*/
	gpumatrix = NULL;
	temporarymatrix = NULL;

	numberofpixels = width * height;	

	temporarymatrix = (short *)malloc((numberofpixels * sizeof(short)));
	cudaMalloc(&gpumatrix, (numberofpixels * sizeof(short)));

	// Just allocated memory on the GPU, now we need to put the values in
		
	// For any x,y coordinate in the host matrix we can say that an offset value is equal to y*width + x
	
	for(x = 0; x < width; x++){
		for(y = 0; y < height; y++){
		
			arrayint = ((y * width) + x);
			temporarymatrix[arrayint] = hostmatrix[x][y];
			//printf("Intensity Temp Matrix: %+d\n", temporarymatrix[arrayint]);
		}
	}


	cudaMemcpy(gpumatrix, temporarymatrix, (numberofpixels * sizeof(short)), cudaMemcpyHostToDevice);

	FFT<<<gridDim, blockDim>>>(gpumatrix,width, height);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess){
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
	
	cudaMemcpy(temporarymatrix,gpumatrix, (numberofpixels * sizeof(short)), cudaMemcpyDeviceToHost);

      for(x = 0; x < width; x++){
                for(y = 0; y < height; y++){
			printf("Intensity Temp Matrix: %+d\n", temporarymatrix[arrayint]);
                }
        }


}


