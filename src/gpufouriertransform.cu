#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

extern "C" int fftshift(cuComplex *target, unsigned int width, unsigned int height){
	unsigned int halfw, halfh;
	unsigned int x,y, offset, tmpoffset;
	cuComplex tmp13, tmp24;

	halfw = width / 2;
	halfh = height / 2;
	
	//printf("Break up the image into 4 quadrants and rearranging them\n");

	//printf("Quadrants 1 & 3\n");
	for( x = 0; x < halfw; x++){
		for(y = 0; y < halfh; y++){
			offset = y * width + x;

			tmp13.x = target[offset].x;
			tmp13.y = target[offset].y;

			tmpoffset = (y + halfh)* width + (x + halfw);

			target[offset].x = target[tmpoffset].x;
			target[offset].y = target[tmpoffset].y;

			target[tmpoffset].x = tmp13.x;
			target[tmpoffset].y = tmp13.y;
		}
	}
//	printf("Quadrants 2 & 4\n");
        for( x = 0; x < halfw; x++){
                for(y = 0; y < halfh; y++){
                        offset = (y+halfh) * width + x;
                        tmp24.x = target[offset].x;
			tmp24.y = target[offset].y;

                        tmpoffset = y * width + (x + halfw);

                        target[offset].x = target[tmpoffset].x;
			target[offset].y = target[tmpoffset].y;

                        target[tmpoffset].x = tmp24.x;
			target[tmpoffset].y = tmp24.y;
                }
        }
return(0);
}
extern "C" int gpufouriertransform(cuComplex *original, cuComplex *transform, unsigned int width, unsigned int height){

	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	cufftHandle plan;
	cufftPlan2d(&plan,height, width, CUFFT_C2C);

	cuComplex *gpuoriginal, *gputransform;

	//printf("Starting the gpu FFT\n");

	cudaMalloc(&gpuoriginal, sizeof(cuComplex) * width * height);
	cudaMalloc(&gputransform, sizeof(cuComplex) * width * height);

	cudaDeviceSynchronize();
        //printf("Allocating Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(gpuoriginal, original, sizeof(cuComplex) * width * height, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
        //printf("Copying Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));
	
	cufftExecC2C(plan,gpuoriginal,gputransform, -1);

	cudaMemcpy(transform, gputransform, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);		
	cudaDeviceSynchronize();
        //printf("Copying back, Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

	//printf("Performing the FFT Shift\n");
	
	fftshift(transform, width, height); 
/*
	for(offset = 0; offset < (width * height); offset ++){
		printf("%g%+g\n", transform[offset].x, transform[offset].y);
	}

*/
	cudaFree(gpuoriginal);
	cudaFree(gputransform);
	cufftDestroy(plan);

	return(0);
}

extern "C" int gpuifouriertransform(cuComplex *original, cuComplex *transform, unsigned int width, unsigned int height){

        dim3 threadsPerBlock(1,1);
        dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
        cufftHandle plan;
        cufftPlan2d(&plan,height, width, CUFFT_C2C);

        cuComplex *gpuoriginal, *gputransform;

        //printf("Starting the gpu FFT\n");

        cudaMalloc(&gpuoriginal, sizeof(cuComplex) * width * height);
        cudaMalloc(&gputransform, sizeof(cuComplex) * width * height);

        cudaDeviceSynchronize();
        //printf("Allocating Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaMemcpy(gpuoriginal, original, sizeof(cuComplex) * width * height, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        //printf("Copying Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));

        cufftExecC2C(plan,gpuoriginal,gputransform, 1);

        cudaMemcpy(transform, gputransform, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //printf("Copying back, Memory errors (?): %s\n", cudaGetErrorString(cudaGetLastError()));


/*
        for(offset = 0; offset < (width * height); offset ++){
                printf("%g%+g\n", transform[offset].x, transform[offset].y);
        }

*/

        cudaFree(gpuoriginal);
        cudaFree(gputransform);
        cufftDestroy(plan);



        return(0);
}

