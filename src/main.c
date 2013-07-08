#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include <cuda.h>
#include "gpu.h"
#include "matrix.h"
#include "tiffman.h"

int main(int argc, char *argv[]){
	uint16 *buffer;
	uint16 *image1, *image2;
	float *matrix;
	unsigned int width, length, x, row, offset, y, currow;
	TIFF *image;

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
			/*  printf("%g \n", matrix[offset]); */
		}
	}

	free(image1);
	float *image2float;
	image2float = malloc((width * length/2) * sizeof(float));
		
	for(offset = 0; offset < (width * (length / 2)); offset ++){
		image2float[offset] = (float)image2[offset];	
	}
	
	free(image2);		

	/* Reference Amplitude Bit ! */
	gpusqrt(image2float, width, (length/2));
	unsigned int height = (length / 2);

	referencephase(image2float, width, height);

	printf("Successfully subtracted the matrices\n");

	_TIFFfree(buffer);
	TIFFClose(image);

	printf("Successfully free'd the buffer & closed the image\n");


/* //Cleaning up the two images */


	printf("Starting GPU based stuff\n");

	
	printf("Finished \n");

return(0);
}
