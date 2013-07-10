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
	float xcontrib, ycontrib, zcontrib, pixelsize;
	pixelsize = 6.45;

	offset = (ycoord * width) + xcoord;
	xcontrib = ((float)xcoord - (((float)width / 2)-1)) * pixelsize;
	ycontrib = ((float)ycoord - (((float)height / 2)-1)) * pixelsize;
	zcontrib = realz;

	gpudistance[offset] = sqrtf( (xcontrib * xcontrib) + (ycontrib * ycontrib) + (zcontrib * zcontrib));
		
}

extern "C" int gpudistance(float *distance, unsigned int width, unsigned int height){
	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	float realz;
	float *gpudistance;

	realz = 43048; /* Distance from pinhold in Z axis */

	cudaMalloc(&gpudistance, sizeof(cuComplex) * (width * height));
	distancekernel<<<numBlock, threadsPerBlock >>>(gpudistance, width, height, realz);	
	cudaDeviceSynchronize();
	printf("GPU Distance: %s\n", cudaGetErrorString(cudaGetLastError()));
	
	cudaMemcpy(distance, gpudistance, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
/*
	int offset;
	for (offset = 0; offset < (width * height); offset ++){
		printf("GPU Distance Function: %f \n", distance[offset]);
	}
*/	
	cudaFree(gpudistance);
	cudaDeviceReset();

	return(0);

}

extern "C" int gpusqrt(float *data, unsigned int width, unsigned int height){
	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	float *gpudata, *gpuresult;	
	int offset;

	cudaMalloc(&gpudata, sizeof(float) * (width * height));
	cudaMalloc(&gpuresult, sizeof(float) * (width * height));	
	cudaMemcpy(gpudata, data, sizeof(float) * (width * height), cudaMemcpyHostToDevice);

	sqrtkernel<<<numBlock, threadsPerBlock >>>(gpudata,gpuresult, width);
	cudaDeviceSynchronize();
	printf("GPUSQRT: %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(gpudata);

	cudaMemcpy(data, gpuresult, sizeof(float) * (width * height), cudaMemcpyDeviceToHost);

/*	for(offset = 0; offset < (width * height); offset++){
		printf("GPU Sqrt Function Results: %f\n", data[offset]);

	} */

 	/*	Works Fine upto here */

	cudaFree(gpuresult);
	cudaDeviceReset();

	return(0);
}

__global__ void wavecalckernel(cuComplex *gpureferencewave, float *gpudistancegrid, float *gpudata, float k, unsigned int width){

        int bidx = blockIdx.x;
        int bidy = blockIdx.y;
        int offset;
        float sincomp, coscomp;

        offset = (bidy * width) + bidx;

        sincosf(k * gpudistancegrid[offset], &sincomp, &coscomp);

        gpureferencewave[offset].x = (gpudata[offset] * coscomp)/(gpudistancegrid[offset]);
        gpureferencewave[offset].y = (gpudata[offset] * sincomp)/(gpudistancegrid[offset]);

}

extern "C" int gpurefwavecalc(cuComplex *referencewave,float *data,float *distancegrid,float k,unsigned int width,unsigned int height){

	dim3 threadsPerBlock(1,1);
	dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);
	printf("K Value is: %g\n", k);	

	cuComplex *gpureferencewave;
	float *gpudistancegrid, *gpudata;

	printf("Allocating the memory on the GPU\n");
	
	cudaMalloc(&gpureferencewave, sizeof(cuComplex) * (width * height));
	cudaMalloc(&gpudistancegrid, sizeof(float) * (width * height));
	cudaMalloc(&gpudata, sizeof(float) * width * height);
	
	printf("Copying the CPU memory onto the newly allocated GPU\n");

	cudaMemcpy(gpudistancegrid, distancegrid, sizeof(float) * width * height,cudaMemcpyHostToDevice);
	cudaMemcpy(gpudata, data, sizeof(float) * width * height,cudaMemcpyHostToDevice);

	printf("Starting a GPU Kernel\n");

	wavecalckernel<<<numBlock, threadsPerBlock>>>(gpureferencewave, gpudistancegrid, gpudata, k, width);
	cudaDeviceSynchronize();

	printf("ReferenceWaveCalc: %s\n", cudaGetErrorString(cudaGetLastError()));

/*	cudaMemcpy(referencewave, gpureferencewave, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost); */

/*	int offset;
	for(offset = 0; offset < (width * height); offset++){
		printf("Reference Wave: %f%+f\n", referencewave[offset].x, referencewave[offset].y);
	} */

	cudaDeviceReset();

	return(0);
};

__global__ void reducedhologramkernel(cuComplex *gpureducedhologram, cuComplex *gpurefwave, float* gpusubimage, unsigned int width){

	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int offset;
	offset = (bidy * width) + bidx;

/*	gpureducedhologram[offset].x = gpusubimage[offset] / (1 * gpurefwave[offset].x);
	gpureducedhologram[offset].y = gpusubimage[offset] / (-1 * gpurefwave[offset].y);
*/

	gpureducedhologram[offset].x = 1;
}

extern "C" int subimagedivref(cuComplex *reducedhologram, float *subimage, cuComplex *refwave, unsigned int width, unsigned int height){
	int offset;
        dim3 threadsPerBlock(1,1);
        dim3 numBlock(width/threadsPerBlock.x, height/threadsPerBlock.y);

	printf("Starting the division of the subtracted image and the reference wave\n");

/*	for(offset = 0; offset < (width * height); offset ++){
		printf("SubImage: %f\n",subimage[offset]);
		printf("refwave: %f%+fi\n", refwave[offset].x, refwave[offset].y);
	} */


	cuComplex *gpureducedhologram, *gpurefwave;
	float *gpusubimage;
	printf("Starting to allocate the memory\n");
	
	cudaMalloc(&gpureducedhologram, sizeof(cuComplex)*(width*height));
	cudaMalloc(&gpurefwave, sizeof(cuComplex)*(width*height));
	cudaMalloc(&gpusubimage, sizeof(float)*(width*height));
	
	printf("Finished Allocating the Memory\n");	


	cudaMemcpy(gpurefwave,refwave, sizeof(cuComplex)*(width*height), cudaMemcpyHostToDevice);
	cudaMemcpy(gpusubimage, subimage, sizeof(float)*(width * height), cudaMemcpyHostToDevice);

	reducedhologramkernel<<<numBlock, threadsPerBlock>>>(gpureducedhologram,gpurefwave,gpusubimage, width);
	cudaDeviceSynchronize();
	
	printf("ReducedHologram: %s\n", cudaGetErrorString(cudaGetLastError()));
	
	cudaMemcpy(reducedhologram, gpureducedhologram, sizeof(cuComplex) * width * height, cudaMemcpyDeviceToHost);

/*	for(offset = 0; offset < (width * height); offset ++){
		printf("%f%+fi \n", reducedhologram[offset].x, reducedhologram[offset].y);
	} */

	cudaFree(gpureducedhologram);
	cudaFree(gpurefwave);
	cudaFree(gpusubimage);

	cudaDeviceReset();

	return(0);

}
