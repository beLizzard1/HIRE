#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include <cuda.h>
#include "cufft.cu.h"
#include "matrix.h"

int main(int argc, char *argv[]){
	short config;
	argc = argc;
	uint16 bps,spp, *buffer;
	uint32 width, length, currow, totalpixel;
	TIFF *image;
	

	if((image = TIFFOpen(argv[1],"r")) == NULL){
		printf("Error opening the TIFF Image\n");
		return(1);
	}

	printf("Loading image properties\n");
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bps);
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &length);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &spp);
	TIFFGetField(image, TIFFTAG_PLANARCONFIG, &config);

	printf("Planar Config: %d, Bits Per Sample: %"PRId16", Samples Per Pixel: %"PRId16", Image Width: %"PRId32", Image Length/Height: %"PRId32"\n", config, bps,spp,width,length);  

	totalpixel = length * width;

	unsigned int i;
	uint16 **image1, **image2;
	
	creatematrix(&image1, width, length);
	creatematrix(&image2, width, length);

	buffer = _TIFFmalloc(TIFFScanlineSize(image));

	printf("Created the blocks in the host memory\n");
        
	for (currow = 0; currow < 1039; currow ++ ){
                if(TIFFReadScanline(image,buffer,currow,0) == -1){
                        printf("An error occured when processing the image\n");
                        return(1);
                }
                for(i = 0; i < width; i++){
                       image1[i][currow] = buffer[i];
                }
        }

	uint32 row;	

        for (currow = 1040; currow < length; currow++){
                if(TIFFReadScanline(image,buffer,currow,0) == -1){
                        printf("An error occured when processing the image\n");
                        return(1);
                }
                
                for(i = 0; i < width; i++){
			row = currow - 1040;
			image2[i][row] = buffer[i];
                }
        }


	// Finishing loading the image ( & splitting it ), now have two arrays of unsigned 16 bit integers to play around with. 

	// Cleaning up stuff that isn't needed any more
	
	_TIFFfree(buffer);
	TIFFClose(image);
		
	// Now we need to subtract the two images to remove the noise and prepare the resulting array/matrix for the result.
	short int **matrix;
	unsigned int x,y;

	matrix = (int16 **)calloc(width, sizeof(int16 *));
	for(i=0; i < width; i++){
		matrix[i] = (int16 *)calloc(length,sizeof(int16));
	}


	for(x = 0; x < width; x++){
		for(y = 1; y < length; y++){
			matrix[x][y] = (image2[x][y] - image1[x][y]);
			// printf("Intensity: %+d\n", matrix[x][y]);
			}
	}
	
	//Cleaning up the two images
	freematrix(image1,width);
	freematrix(image2,width);


	//Move to GPU
	printf("Starting GPU based stuff\n");
	gpucalculation(width, length/2, matrix);

	
	printf("Finished \n");

	//Cleaning up the subtracted image
	for(i=0; i < width; i++){
		free(matrix[i]);
	}
	free(matrix);


return(0);
}
