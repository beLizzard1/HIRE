#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>

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

	TIFF *image;
	unsigned int width, length, row, x, y, offset, column;
	unsigned short bps, spp;
	float *subtractedimage, *image2f, *dist2pinhole, pinholedist, k;
	tsize_t scanlinesize;


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
	printf("Image Properties: ");
	printf("BitsPerSample: %u, SamplesPerPixel: %u, Image Width: %u, Image Length: %u\n", bps, spp, width, length);


	printf("This data is actually comprised of two seperate images so we need to split them\n");
	uint16 *buffer;
	uint16 *image1, *image2;

	image1 = malloc(width * length * sizeof(uint16));
	image2 = malloc(width * length * sizeof(uint16));
	buffer = _TIFFmalloc(TIFFScanlineSize(image));

	if( image1 == NULL || image2 == NULL || buffer == NULL ){
		fprintf(stderr, "An Error occured while allocating memory for the images\n");
		return(0);
	}

	printf("Now to load the data from the image\n");

	unsigned int height;
	height = length / 2;

	for( row = 0; row < ((length/2)-1); row++){
		TIFFReadScanline(image,buffer,row,0);
		for(column = 0; column < width; column++){
			offset = (row * width) + column;
			image1[offset] = buffer[column];
/*			printf("%hu\n", image1[offset]); */
		}
	}	      

	for( row = 1040; row < (length - 1); row++){
		TIFFReadScanline(image,buffer,row,0);
		for(column = 0; column < width; column++){
			offset = ((row - 1040) * width) + column;
			image2[offset] = buffer[column];
/*			printf("%hu\n", image2[offset]); */
		}
	}

	printf("Now to Subtract the images\n");

	subtractedimage = malloc(width * height * sizeof(float));

	subtractimages(image1, image2, subtractedimage, width, height);

	image2f = malloc(width * height * sizeof(float));

	dist2pinhole = malloc(width * height * sizeof(float));

	pinholedist = 43048; /* Distance to the central pixel of the sensor in micron */

	gpudistancegrid(dist2pinhole, pinholedist, width, height);

	/* Starting Reference Wave calculations */
	/* Several difference ways of doing this, because the image I am working with is a double pinhole, I need to take the mean amplitude of image2f */	

	/* Double Pinhole stuff takes the mean */

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
			image2f[offset] = sqrtf((float)total / (width * height));
		}
	}

	/* Single Pinhole takes each indiviual value uncomment to use */
/*
	for(x = 0; x < width; x++){
                for(y = 0; y < height; y++){
                        offset = (y * width) + x;
                        image2f[offset] = (float)image2[offset];
                }
        }
*/	
	cuComplex *referencewave;

	referencewave = malloc(sizeof(cuComplex) * width * height);	

	gpurefwavecalc(referencewave, image2f, dist2pinhole, k, width, height);
	
/*
	for(x = 0; x < width; x++){
		for(y = 0; y < height; y++){
			offset = (y * width) + x;
			printf("%g%+gi\n", referencewave[offset].x, referencewave[offset].y);
		}
	}
*/


	/* Now we need to divide by the reference wave calculated above to find the reduced hologram */
	cuComplex *reducedhologram;
	reducedhologram = malloc(sizeof(cuComplex) * width * height);

	gpusubdivref(reducedhologram, subtractedimage, referencewave, width, height);

	/* Starting Fourier Transfrom stuff */

	cuComplex *treducedhologram;
	treducedhologram = malloc(sizeof(cuComplex) * width * height);
	gpufouriertransform(reducedhologram, treducedhologram, width, height);
	
	TIFF* output;
	
	if((output = TIFFOpen("/tmp/image.tiff", "w")) == NULL){
                printf("Error Opening the image\n");
                return(1);
        }

	writeimage(output, width, height, treducedhologram, scanlinesize);



	/* Freeing stuff that has been allocated to the host */
	free(image1);
	free(image2);
	free(subtractedimage);
	free(image2f);
	free(dist2pinhole);
	free(referencewave);
	free(reducedhologram);
	_TIFFfree(buffer);
	TIFFClose(image);
	
	return(0);
}
