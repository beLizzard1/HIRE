#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include <cuda.h>
#include "cufft.cu.h"
#include "matrix.h"
#include "tiffman.h"

int main(int argc, char *argv[]){
	unsigned short *buffer, *temp1, *temp2;
	float *image1, *image2;
	unsigned int width, length, x, y, row, offset;
	TIFF *image;

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
	temp1 = malloc((width * (length/2)) * sizeof(unsigned short));
	temp2 = malloc((width * (length/2)) * sizeof(unsigned short));
	
	image1 = (float *)malloc((width * length/2) * sizeof(float));	
	image2 = (float *)malloc((width * length/2) * sizeof(float));

	if(image1 == NULL || image2 == NULL){
		printf("An error has occured while allocating memory for the images\n");
		return(1);
	}

	printf("Loading the data from the images\n");
       
	for( row = 0; row < ((length / 2) - 1); row++ ){
		if(TIFFReadScanline(image,buffer,row,0) == -1){
			printf("An error occured when loading the image\n");
		}

		for( x = 0; x < width; x++ ){
			offset = (row * width) + x;
			temp1[offset] = buffer[width];
		}
	}

 

        for (row = 1040; row < length; row++){
                if(TIFFReadScanline(image,buffer,row,0) == -1){
                        printf("An error occured when processing the image\n");
                        return(1);
                }
		
		for( x = 0; x < width; x++){
			offset = (row * width) + x;
			temp2[offset] = buffer[width];
		}
 	}

	printf("Images have been split just going to print the values as a test \n");

	for(x = 0; x < width; x++){
		for(y = 0; y < (length/2)-1; y++){
			printf("Intensity Image 1: %hu\n", temp1[((y * width) + x)]);			
		}
	}


	free(image1);
	free(image2);
	_TIFFfree(buffer);
	TIFFClose(image);
		
	
/* //Cleaning up the two images */


	printf("Starting GPU based stuff\n");

	
	printf("Finished \n");



return(0);
}
