#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>

#include "subtractimages.h"
#include "gpudistancegrid.h"
#include "gpurefwavecalc.h"
#include "gpusubdivref.h"
#include "gpufouriertransform.h"
#include "writetoimage.h"

#define PI 3.1415926535


int main(int argc, char *argv[]){

	TIFF *image, *output;
	unsigned int width, length, x, y, offset, height, i;
	unsigned short bps, spp;
	float *subtractedimage, *dist2pinhole, pinholedist, k;
	tsize_t scanlinesize, objsize;


	k = ( 2 * PI ) / 0.780241;

	if(argc < 2){
		printf("You didn't provide the correct input\n");
		return(1);
	}


	if((image = TIFFOpen(argv[1], "r")) == NULL){
		printf("Error Opening the image\n");
		return(1);
	}

	scanlinesize = TIFFScanlineSize(image);
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bps);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &spp);
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &length);
	height = length / 2;
	printf("Image Properties: ");
	printf("BitsPerSample: %u, SamplesPerPixel: %u, Image Width: %u, Image Length: %u\n", bps, spp, width, length);


	scanlinesize = TIFFScanlineSize(image);
	objsize = (scanlinesize) / (width * spp);

	uint16 *image1, *image2;
	uint16 *buffer;
	image1 = _TIFFmalloc(objsize * width * height);
	image2 = _TIFFmalloc(objsize * width * height);
	buffer = _TIFFmalloc(objsize * width);


        if( image1 == NULL || image2 == NULL || buffer == NULL ){
		fprintf(stderr, "An Error occured while allocating memory for the images\n");
		return(0);
	}

	printf("Now to load the data from the provided image\n");

	for( i = 0; i < (height - 1); i++){
		TIFFReadScanline(image, buffer, i,0);
		memcpy(&image1[i * width], buffer, scanlinesize);
	}

	for( i = (height); i < length; i++){
		TIFFReadScanline(image, buffer, i, 0);
		memcpy(&image2[( i - height ) * width], buffer, scanlinesize);
	}

	free(buffer);

/*	subtractimages(image1, image2, subtractedimage, width, height);

	dist2pinhole = malloc(width * height * sizeof(float));
*/
	pinholedist = 43048; /* Distance to the central pixel of the sensor in micron */
/*

	gpudistancegrid(dist2pinhole, pinholedist, width, height);

*/
	/* Starting Reference Wave calculations */
	/* Several difference ways of doing this, because the image I am working with is a double pinhole, I need to take the mean amplitude of image2f */	

	/* Double Pinhole stuff takes the mean */

/*
	float total;
	total = 0;
	for(offset = 0; offset < width * height; offset++){
		total += (float)image2[offset];
	}	
	printf("Sqrt of Mean Value: %g\n", (float)sqrtf(total/(offset)));
	printf("Now assigning that to every field within image2f\n");
	for(x = 0; x < width; x++){
		for(y = 0; y < height; y++){
			offset = (y * width) + x;
			image2[offset] = sqrtf((float)total / (float)(width * height));
		}
	}
*/

	/* Single Pinhole takes each indiviual value uncomment to use */
/*
	for(x = 0; x < width; x++){
                for(y = 0; y < height; y++){
                        offset = (y * width) + x;
                        image2f[offset] = (float)image2[offset];
                }
        }
*/	
/*	cuComplex *referencewave;

	referencewave = malloc(sizeof(cuComplex) * width * height);	

	gpurefwavecalc(referencewave, image2, dist2pinhole, k, width, height);
*/	
/*
	for(x = 0; x < width; x++){
		for(y = 0; y < height; y++){
			offset = (y * width) + x;
			printf("%g%+gi\n", referencewave[offset].x, referencewave[offset].y);
		}
	}
*/


	/* Now we need to divide by the reference wave calculated above to find the reduced hologram */
/*	cuComplex *reducedhologram;
	reducedhologram = malloc(sizeof(cuComplex) * width * height);

	gpusubdivref(reducedhologram, subtractedimage, referencewave, width, height);
*/

	/* Starting Fourier Transform stuff */

	/* Temporary Bit just to get a reconstruction */

	cuComplex *sub;
	sub = malloc(sizeof(cuComplex) * width * height);

	for(offset = 0; offset < (width * height); offset++){
		sub[offset].x = (float)image1[offset] - (float)image2[offset];
		sub[offset].y = 0;
	}

/*	cuComplex *treducedhologram;
	treducedhologram = malloc(sizeof(cuComplex) * width * height);
	gpufouriertransform(reducedhologram, treducedhologram, width, height);
*/
	cuComplex *tsub;
	tsub = malloc(sizeof(cuComplex) * width * height);
	gpufouriertransform(sub, tsub, width, height);
		
	printf("Writing the output.tiff\n");

	if((output = TIFFOpen("result.tiff", "w")) == NULL){
		printf("Error opening image\n");
		return(1);
	}
	writerealimage(output, width, height, tsub, scanlinesize);

	if((output = TIFFOpen("resultcomplex.tiff", "w")) == NULL){
		printf("Error opening image \n");
		return(1);
	}
	writecompleximage(output, width, height, tsub, scanlinesize);

	
/*	if((output = TIFFOpen("/tmp/image.tiff", "w")) == NULL){
                printf("Error Opening the image\n");
                return(1);
        } 

	writeimage(output, width, height, treducedhologram, scanlinesize);
*/


	/* Freeing stuff that has been allocated to the host */
/*	free(subtractedimage);
	free(dist2pinhole);
	free(referencewave);
	free(reducedhologram);
	_TIFFfree(buffer);
	TIFFClose(image);
*/	
	return(0);
}
