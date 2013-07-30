#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <tiffio.h>
#include <cuComplex.h>

int writeimage(TIFF* image, unsigned int width, unsigned int height, void *data, tsize_t scanlinesize){
	unsigned int row;

	printf("Writing an image\n");

	TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, sizeof(float));
	TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 1);
	TIFFSetField(image, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(image, TIFFTAG_ARTIST,"RobertJames");
	TIFFSetField(image, TIFFTAG_COMPRESSION, 1);
	TIFFSetField(image, TIFFTAG_SUBFILETYPE, 1);

	printf("Creating Buffer for each line to be loaded\n");

	float *buffer;
	buffer = _TIFFmalloc(scanlinesize);

	for(row = 0; row < height; row++){
		memcpy(buffer, &data[(row*width)].x, (width * sizeof(scanlinesize)));
		TIFFWriteScanline(image, buffer, row,0);
	}

	TIFFClose(image);
	_TIFFfree(buffer);	
	return(0);
}
