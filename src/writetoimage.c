#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <tiffio.h>
#include <cuComplex.h>

int writerealimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize){
	int row, offset;
	float *real, *imag;

	printf("Writing an image\n");

	TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 16);
	TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(image, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 1);

	/* cuComplex data  is complex !!! meaning that it has an .x and a .y which are real and imaginary components respectivly need to split them so that it makes sense */

	printf("Allocating memory(temporarily) to split the complex numbers\n");
	real = malloc(sizeof(float) * width * height);
	imag = malloc(sizeof(float) * width * height);

	printf("Splitting into real and imaginary components\n");

	for(offset = 0; offset < ( width * height ); offset++){
		real[offset] = data[offset].x;
		imag[offset] = data[offset].y;
	}

	printf("Split\n");

	float *buffer;
	buffer = malloc(sizeof(float) * width);

	for(row = 0; row < height; row++){
		memcpy(buffer, &real[row * width], sizeof(float) * width);
		if((TIFFWriteScanline(image, buffer, row,1)) == -1){
			fprintf(stderr, "Something went wrong writing the file\n");  
		}
	}

	TIFFClose(image);

	return(0);
}

int writecompleximage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize){
        int row, offset;
        float *real, *imag;

        printf("Writing an image\n");

        TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(image, TIFFTAG_PLANARCONFIG, 1);

        /* cuComplex data  is complex !!! meaning that it has an .x and a .y which are real and imaginary components respectivly need to split them so that it makes sense */

        printf("Allocating memory(temporarily) to split the complex numbers\n");
        real = malloc(sizeof(float) * width * height);
        imag = malloc(sizeof(float) * width * height);

        printf("Splitting into real and imaginary components\n");

        for(offset = 0; offset < ( width * height ); offset++){
                real[offset] = data[offset].x;
                imag[offset] = data[offset].y;
        }

        printf("Split\n");

        float *buffer;
        buffer = malloc(sizeof(float) * width);

        for(row = 0; row < height; row++){
                memcpy(buffer, &imag[row * width], sizeof(float) * width);
                if((TIFFWriteScanline(image, buffer, row,1)) == -1){
                        fprintf(stderr, "Something went wrong writing the file\n");
                }
        }

        TIFFClose(image);

        return(0);
}

