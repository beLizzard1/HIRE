#include <tiffio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
#include <tiffio.h>


int fileopen(char **filename, unsigned int *width, unsigned int *length, TIFF* image){
	unsigned short bps, spp;
	unsigned int config;
        printf("Loading image properties\n");
        TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bps);
        TIFFGetField(image, TIFFTAG_IMAGEWIDTH, width);
        TIFFGetField(image, TIFFTAG_IMAGELENGTH, length);
        TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetField(image, TIFFTAG_PLANARCONFIG, &config);
        printf("Planar Config: %u, Bits Per Sample: %u, Samples Per Pixel: %u \n", config, bps,spp);

	return(0);
}

int filewrite(unsigned int width, unsigned int length,tmsize_t scanlinesize, float *data){
	TIFF *output;
	unsigned int row, offset,i;
	uint16 *buffer;

	output = TIFFOpen("output.tiff", "w");

	TIFFSetField(output, TIFFTAG_BITSPERSAMPLE, 16);
	TIFFSetField(output, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(output, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(output, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(output, TIFFTAG_IMAGELENGTH, length);

	buffer = _TIFFmalloc(scanlinesize);

	for(row = 0; row < (length/2); row++){
		for(i=0; i < width; i++){
			offset = (row * width) + i;
			buffer[i] = data[offset];
			TIFFWriteScanline(output, buffer, row, 1);
		}
	}

	return(0);

}
