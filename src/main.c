#include <stdio.h>

#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>
#include "rasterarray.h"

#define TIFFGetR(abgr) ((abgr) & 0xff)
#define TIFFGetG(abgr) (((abgr) >> 8) & 0xff)
#define TIFFGetB(abgr) (((abgr) >> 16) & 0xff)
#define TIFFGetA(abgr) (((abgr) >> 24) & 0xff)
#define grayscale(v) ((TIFFGetR(v)+TIFFGetG(v)+TIFFGetB(v))/3)

int main(int argc, char *argv[]){
	
	argc = argc;
	uint16 bps,spp;
	uint32 width, height, length, *array;
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

	printf("Bits Per Sample: %"PRId16", Samples Per Pixel: %"PRId16", Image Width: %"PRId32", Image Length/Height: %"PRId32"\n", bps,spp,width,length);  

	height = length;

	printf("\nWe now going to load the entire image into a raster \n");	

	array = (uint32 *)_TIFFmalloc((width * height) * sizeof(uint32));

	TIFFReadRGBAImage(image, width, height, array, 0);

	// Now need to be smart about how we read the image, raster images start at the bottom left.

	unsigned int x,y,arrayint;
	uint32 data;

	for(x = 0; x < width; x++){

		for(y = 0; y < height; y++){

			arrayint = coordtoraster(width, x, y);
			data = array[arrayint];
			printf("%d\n",grayscale(data));			
//			printf("(%d,%d) Value: %d\n",x,y,grayscale(data));	

		}
	}
	
		
	//Freeing the allocated memory & Closing stuff
	TIFFClose(image);

	_TIFFfree(array);

return(0);

}
