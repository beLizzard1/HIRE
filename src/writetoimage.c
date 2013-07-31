#include <stdio.h>
#include <errno.h>
#include <fenv.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <tiffio.h>
#include <cuComplex.h>

int writerealimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize){
	unsigned int row, offset;
	uint16 *real;

	TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 16);
	TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(image, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 1);

	/* cuComplex data  is complex !!! meaning that it has an .x and a .y which are real and imaginary components respectivly need to split them so that it makes sense */

	real = malloc(sizeof(uint16) * width * height);

	/* Normalising the image so that the smallest value is equal to 0 and the largest value is equal to the maximum size of an unsigned 16bit integer 0 - 65,535 */
	float big, small;
	big = 0;
	small = 0;

	for(offset = 0; offset < (width * height); offset++){
		if(big < data[offset].x){
			big = data[offset].x;
		}
	}
	printf("Largest Value: %f\n", big);


	for(offset = 0; offset < (width * height); offset++){
		if(small > data[offset].x){
			small = data[offset].x;
		}
	}
	printf("Smallest Value: %f\n", small);

	float *tmpreal;
	tmpreal = malloc(sizeof(float) * width * height);
	for(offset = 0; offset < (width * height); offset++){
		tmpreal[offset] = data[offset].x - small;
		tmpreal[offset] = tmpreal[offset] / (big-small) * 65536;
	}	

	for(offset = 0; offset < (width * height); offset++){
		real[offset] = (uint16)tmpreal[offset];
	}

	uint16 *buffer;
	buffer = malloc(sizeof(uint16) * width);

	for(row = 0; row < height; row++){
		memcpy(buffer, &real[row * (width)], scanlinesize);
		if((TIFFWriteScanline(image, buffer, row,1)) == -1){
			fprintf(stderr, "Something went wrong writing the file\n");  
		}
	}

	TIFFClose(image);

	return(0);
}

int writecompleximage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize){
        unsigned int row, offset;
        uint16 *imag;

        TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(image, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 1);

        /* cuComplex data  is complex !!! meaning that it has an .x and a .y which are real and imaginary components respectivly need to split them so that it makes sense */

        imag = malloc(sizeof(uint16) * width * height);

      /* Normalising the image so that the smallest value is equal to 0 and the largest value is equal to the maximum size of an unsigned 16bit integer 0 - 65,535 */
        float big, small;
        big = 0;
        small = 0;
        for(offset = 0; offset < (width * height); offset++){
                if(big < data[offset].y){
                        big = data[offset].y;
                }
        }
        printf("Largest Value: %f\n", big);


        for(offset = 0; offset < (width * height); offset++){
                if(small > data[offset].y){
                        small = data[offset].y;
                }
        }
        printf("Smallest Value: %f\n", small);

        float *tmpimag;
        tmpimag = malloc(sizeof(float) * width * height);
        for(offset = 0; offset < (width * height); offset++){
                tmpimag[offset] = data[offset].x - small;
                tmpimag[offset] = tmpimag[offset] / (big-small) * 65536;
        }

        for(offset = 0; offset < (width * height); offset++){
                imag[offset] = (uint16)tmpimag[offset];
        }


        uint16 *buffer;
        buffer = malloc(sizeof(uint16) * width);

        for(row = 0; row < height; row++){
                memcpy(buffer, &imag[row * width], scanlinesize);
                if((TIFFWriteScanline(image, buffer, row,1)) == -1){
                        fprintf(stderr, "Something went wrong writing the file\n");
                }
        }

        TIFFClose(image);

        return(0);
}


int writeabsimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize){
        unsigned int row, offset;
        uint16 *abs;
	abs = malloc(sizeof(uint16) * width * height);

        TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(image, TIFFTAG_PLANARCONFIG, 1);
        TIFFSetField(image, TIFFTAG_PHOTOMETRIC, 1);

	float *tmp;
	tmp = malloc(sizeof(float) * width * height);

	float big, small;
        big = 0;
        small = 0;
        for(offset = 0; offset < (width * height); offset++){
                if(big < cuCabsf(data[offset])){
                        big = cuCabsf(data[offset]);
                }
        }
        printf("Largest Value: %f\n", big);


        for(offset = 0; offset < (width * height); offset++){
                if(small > cuCabsf(data[offset])){
                        small = cuCabsf(data[offset]);
                }
        }
        printf("Smallest Value: %f\n", small);

        for(offset = 0; offset < (width * height); offset++){
                tmp[offset] = (float)cuCabsf(data[offset]) - (float)small;
                tmp[offset] = (float)(cuCabsf(data[offset]) / (float)(big-small)) * 65536;
        }

	for(offset = 0; offset < (width * height); offset++){
		abs[offset] = (uint16)tmp[offset];
	}

        uint16 *buffer;

        buffer = malloc(sizeof(uint16) * width);

        for(row = 0; row < height; row++){
                memcpy(buffer, &abs[row * width], scanlinesize);
                if((TIFFWriteScanline(image, buffer, row,1)) == -1){
                        fprintf(stderr, "Something went wrong writing the file\n");
                }
        }

        TIFFClose(image);

        return(0);
}
