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
	short config;
	argc = argc;
	uint16 bps,spp;
	uint32 width, height, length, currow;
	TIFF *image;
	tdata_t buffer;
	
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


	printf("Planar Config: %d Bits Per Sample: %"PRId16", Samples Per Pixel: %"PRId16", Image Width: %"PRId32", Image Length/Height: %"PRId32"\n", config, bps,spp,width,length);  

	height = length;

	buffer = _TIFFmalloc(TIFFScanlineSize(image));
	
	for (currow = 0; currow < height; currow ++ ){
		printf("Row: %"PRId32"\n", currow);
		if(TIFFReadScanline(image,buffer,currow,0) == -1){
			printf("An error occured when processing the image\n");
			return(1);
		}

	}



	_TIFFfree(buffer);
	TIFFClose(image);

return(0);

}
