#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>
#include <complex.h>
#include "gpu.h"
#include "matrix.h"
#include "tiffman.h"
#include "dftandprop.h"
#define PI 3.1415926535


int main(int argc, char *argv[]){
	uint16 *buffer;
	uint16 *image1, *image2;
	float *matrix;
	unsigned int width, length, x, row, offset, y, currow;
	TIFF *image;
	float k;
	k =  2 * PI / 0.780241;

	currow = 0;
	argc = argc;

        if((image = TIFFOpen(argv[1],"r")) == NULL){
                printf("Error opening the TIFF Image\n");
                return(1);
        }
	
	fileopen(argv,&width,&length,image);

	printf("Width: %u, Height: %u\n", width, length);
	printf("This is actually comprised of two images\n");
	printf("So when we read the data into the memory\nwe will split this into two arrays\n");

	printf("Width (Image1): %u, Height (Image1): %u\n",width,length/2);
	printf("Width (Image2): %u, Height (Image2): %u\n",width,length/2);
	
/* Storing all the data into a 1d array for ease. Will need to carry around the width to find the offset */

	buffer = _TIFFmalloc(TIFFScanlineSize(image));

		

	image1 = malloc((width * length/2) * sizeof(uint16));	
	image2 = malloc((width * length/2) * sizeof(uint16));
	matrix = malloc((width * length/2) * sizeof(float));


	if(image1 == NULL || image2 == NULL){
		printf("An error has occured while allocating memory for the images\n");
		return(1);
	}

	printf("Loading the data from the images\n");
       
	printf("Loading Image 1\n");
	for( row = 0; row < ((length / 2) - 1); row++ ){
		if(TIFFReadScanline(image,buffer,row,0) == -1){
			printf("An error occured when loading the image\n");
		}

		for( x = 0; x < width; x++ ){
			offset = (row * width) + x;
			image1[offset] = (uint16)buffer[x];
/*			printf("Image 1 - Original: %hu Memory: %hu\n", buffer[i], image1[offset]); */
		}
	}
	printf("Image 1: Loaded\n");

	printf("Loading Image 2\n");
	for(row = 1040; row < length; row++){
		if(TIFFReadScanline(image,buffer,row,0) == -1){
			printf("An error occured when loading the image\n");
		}
		for( x = 0; x < width; x++){
			offset = (currow * width) + x;
			image2[offset] = (uint16)buffer[x];
/*			printf("Image 2 - Original: %hu Memory: %"PRId16"\n", buffer[i], image2[offset]); */
			}
		currow = currow + 1;
		}

	printf("Image 2: Loaded\n");

	printf("Images have been split, need to subtract and take the result as floats\n");

	for(y = 0; y < (length / 2); y++){
		for(x = 0; x < width; x++){
			offset = (y * width) + x;
			matrix[offset] = (float)image1[offset] - (float)image2[offset];
/*			printf("%g \n", matrix[offset]); */
		}
	}

	free(image1);

	float *image2float, *distancegrid;
	image2float = (float *)malloc((width * length/2) * sizeof(float));
	distancegrid = (float *)malloc((width * length/2) * sizeof(float));		

	for(offset = 0; offset < (width * (length / 2)); offset ++){
		image2float[offset] = (float)image2[offset];
/*		printf("Original: %f New: %f \n", (float)image2[offset],image2float[offset]); */
	}
	
	free(image2);
	
	/* Reference Amplitude Bit ! */
	/* gpusqrt(image2float, width, (length/2)); use this line to do single pinhole stuff */

	/* Temporarily holding the reference amplitude to 1 */
	for (offset = 0; offset < (width * length/2); offset++){
		image2float[offset] = 1;
	}


	unsigned int height = (length / 2);

/*	for(offset = 0; offset < (width * height); offset++){
		printf("After GPUSqrt in Main Function: %f\n", image2float[offset]);
	} */

	gpudistance(distancegrid, width, height);
	cuComplex *referencewave;
	referencewave = (cuComplex *)malloc(sizeof(float) * width * height);

	/* Works up to here 10/07/2013 11:29 */

	gpurefwavecalc(referencewave,image2float, distancegrid, k, width, height);
/*
	for(offset = 0; offset < (width * height); offset ++){
		printf("%+f %+fi\n", referencewave[offset].x, referencewave[offset].y);
	}
*/
	free(image2float);
	_TIFFfree(buffer);
	TIFFClose(image);
	
	printf("Successfully free'd the buffer & closed the image\n");
	printf("Calculated the Reference Wave on the GPU, bet you didn't notice that\n");

	printf("This bit was James' idea he thinks that we 'multiply' by the reference wave because of 'reasons'\n");
	
	cuComplex *reducedhologram;
	reducedhologram = (cuComplex *)malloc(sizeof(cuComplex) * width * height);

	subimagedivref(reducedhologram, matrix, referencewave, width, height);

	/* Starting the Fourier Transform stuff */

	float startz;
	float endz;
	startz = 0;
	endz = 1;

	gpudftwithprop(reducedhologram, width, height,startz, endz);

	printf("Finished \n");

return(0);
}
