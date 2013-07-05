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
