#define _USE_MATH_DEFINES
#define GLFW_INCLUDE_GLU
#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <tiffio.h>
#include <cuda.h>
#include <SDL/SDL.h>
#include <cuComplex.h>

#include <GL/glfw.h>

#include "gpupropagate.h"
#include "gpudistancegrid.h"
#include "gpurefwavecalc.h"
#include "gpufouriertransform.h"
#include "writetoimage.h"


int main(int argc, char *argv[]){

	float planeseparation;
	planeseparation = 100;

	TIFF *image, *output;
	unsigned int i;
	unsigned int width, length, offset, height;
	unsigned short bps, spp;
	float k, pixelsize, pinholedist;
	tsize_t scanlinesize, objsize;

	pixelsize = 6.45;
	k = ( 2 * M_PI ) / 0.780241;

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

//	printf("Now to load the data from the provided image\n");

	for( i = 0; i < (height - 1); i++){
		TIFFReadScanline(image, buffer, i,0);
		memcpy(&image1[i * width], buffer, scanlinesize);
	}

	for( i = (height); i < length; i++){
		TIFFReadScanline(image, buffer, i, 0);
		memcpy(&image2[( i - height ) * width], buffer, scanlinesize);
	}

	free(buffer);

	long *longimg1, *longimg2;
	float *fimg1, *fimg2;

	longimg1 = malloc(sizeof(long) * width * height);
	longimg2 = malloc(sizeof(long) * width * height);
	fimg1 = malloc(sizeof(float) * width * height);
	fimg2 = malloc(sizeof(float) * width * height);

        for(offset = 0; offset < (width * height); offset++){
                longimg1[offset] = (long)image1[offset];
                longimg2[offset] = (long)image2[offset];
        }

	for(offset = 0; offset < (width * height); offset++){
		fimg1[offset] = (float)longimg1[offset];
		fimg2[offset] = (float)longimg2[offset];
	}

	free(image1);
	free(image2);
	free(longimg1);
	free(longimg2);

	/* View img1, img2 */
	cuComplex *cuimg1, *cuimg2;
	cuimg1 = malloc(sizeof(cuComplex) * width * height);
	cuimg2 = malloc(sizeof(cuComplex) * width * height);

	for(offset = 0; offset < (width * height); offset ++){
		cuimg1[offset].x = fimg1[offset];
		cuimg2[offset].x = fimg2[offset];
	}

	        if((output = TIFFOpen("img1.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writerealimage(output, width, height, cuimg1, scanlinesize);

                if((output = TIFFOpen("img2.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writerealimage(output, width, height, cuimg2, scanlinesize);

	free(cuimg1);
	free(cuimg2);


        cuComplex *sub;
        sub = malloc(sizeof(cuComplex) * width * height);

        for(offset = 0; offset < (width * height); offset++){
                sub[offset].x = fimg1[offset] - fimg2[offset];
                sub[offset].y = 0;
        }

		if((output = TIFFOpen("sub.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
	writerealimage(output, width, height, sub, scanlinesize);

	/* Calculating the mean value of the subtracted image and subtract it */
	float sum;
	for(offset = 0; offset < (width * height); offset++){
		sum += sub[offset].x;	
	}
	sum = (sum / (width * height));

	for(offset = 0; offset < (width * height); offset++){
		sub[offset].x -= sum;
	}
                if((output = TIFFOpen("meansubtractedsub.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writerealimage(output, width, height, sub, scanlinesize);



	free(fimg1);
	free(fimg2);

	pinholedist = 57000; /* Distance to the central pixel of the sensor in micron */

	cuComplex *referencewave;

	referencewave = malloc(sizeof(cuComplex) * width * height);	

	gpurefwavecalc(referencewave, width, height, pinholedist,k,pixelsize);

        if((output = TIFFOpen("referencewave/realrefresult.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writerealimage(output, width, height, referencewave, scanlinesize);

	        if((output = TIFFOpen("referencewave/imagrefresult.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
	writecompleximage(output, width, height, referencewave, scanlinesize);
                if((output = TIFFOpen("referencewave/absrefresult.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writeabsimage(output, width, height, referencewave, scanlinesize);



	/* Dividing the Subtracted Image by the Reference Wave */

	cuComplex *reducedhologram;
	reducedhologram = malloc(sizeof(cuComplex) * width * height);

	for(offset = 0; offset < (width * height); offset++){
		reducedhologram[offset].x = (sub[offset].x * referencewave[offset].x) / ((referencewave[offset].x * referencewave[offset].x) + (referencewave[offset].y * referencewave[offset].y));

		reducedhologram[offset].y = (sub[offset].x * referencewave[offset].y) / ((referencewave[offset].x * referencewave[offset].x) + (referencewave[offset].y * referencewave[offset].y));


	}

	        if((output = TIFFOpen("reducedhologram/realreducedhologram.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
	writerealimage(output, width, height, reducedhologram, scanlinesize);

		        if((output = TIFFOpen("reducedhologram/imagreducedhologram.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
	writecompleximage(output, width, height, reducedhologram, scanlinesize);

                if((output = TIFFOpen("reducedhologram/absreducedhologram.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writeabsimage(output, width, height, reducedhologram, scanlinesize);



	/* Starting Fourier Transform stuff */

	cuComplex *tredhologram;
	tredhologram = malloc(sizeof(cuComplex) * width * height);
	gpufouriertransform(reducedhologram, tredhologram, width, height);

//	printf("Writing the output.tiff\n");

	if((output = TIFFOpen("fftreducedhologram/result.tiff", "w")) == NULL){
		printf("Error opening image\n");
		return(1);
	}
	writerealimage(output, width, height, tredhologram, scanlinesize);

	if((output = TIFFOpen("fftreducedhologram/resultimag.tiff", "w")) == NULL){
		printf("Error opening image \n");
		return(1);
	}
	writecompleximage(output, width, height, tredhologram, scanlinesize);

	if((output = TIFFOpen("fftreducedhologram/abs.tiff","w")) == NULL){
		printf("Error opening the image\n");
		return(1);
	}
	writeabsimage(output, width, height, tredhologram, scanlinesize);


 	/* Doing another fourier transform, to see if the input and output are the same */
	cuComplex *itredhologram;
	itredhologram = malloc(sizeof(cuComplex) * width * height);
	gpuifouriertransform(tredhologram, itredhologram, width, height);

        if((output = TIFFOpen("ifftreducedhologram/realifftresult.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writerealimage(output, width, height, itredhologram, scanlinesize);

        if((output = TIFFOpen("ifftreducedhologram/imagifftresult.tiff", "w")) == NULL){
                printf("Error opening image \n");
                return(1);
        }
        writecompleximage(output, width, height, itredhologram, scanlinesize);

	if((output = TIFFOpen("ifftreducedhologram/absifftresult.tiff", "w")) == NULL){
                printf("Error opening image \n");
                return(1);
        }
	writeabsimage(output, width, height, itredhologram, scanlinesize);




	/* Propagating the transformed image */
	cuComplex *propagatedimage, *ipropagatedimage;
	float *absipropagatedimage;
	propagatedimage = malloc(sizeof(cuComplex) * width * height);
	ipropagatedimage = malloc(sizeof(cuComplex) * width * height);
	absipropagatedimage = malloc(sizeof(float) * width * height);
	float dist, maxdist;
	char absdist[50];
	maxdist = 120000;

	/* GLFW Interface Stuff */
	int running = GL_TRUE;
	if(!glfwInit()){
		exit(EXIT_FAILURE);
	}
	if(!glfwOpenWindow(width/2,0,0,0,0,0,0,0, GLFW_WINDOW)){
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	float max;
	/* Propagation Loop */
	glClearColor(0,0,0,0);

	while(running){
		glClear(GL_COLOR_BUFFER_BIT);
		for(dist = 30000; dist < maxdist; dist = dist + planeseparation){
		snprintf( absdist, 50, "PropagatedDistance: %f \316 \274 m", dist);
		glfwSetWindowTitle(absdist);	
		gpupropagate(tredhologram, propagatedimage, width, height, k, pixelsize,dist, scanlinesize);

		gpuifouriertransform(propagatedimage, ipropagatedimage, width, height);

		for(offset = 0; offset < (width * height); offset++){
			absipropagatedimage[offset] = sqrtf( (ipropagatedimage[offset].x * ipropagatedimage[offset].x) + (ipropagatedimage[offset].y * ipropagatedimage[offset].y));
			if( max < absipropagatedimage[offset] ){
				max = absipropagatedimage[offset];
			}
		}	

		for(offset = 0; offset < (width * height); offset++){
			absipropagatedimage[offset] = absipropagatedimage[offset] / max;
		}



		glDrawPixels(width, height, GL_LUMINANCE, GL_FLOAT, absipropagatedimage);

		glfwSwapBuffers();

		running = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam(GLFW_OPENED);
	}
/*
        if((output = TIFFOpen(absdist,"w")) == NULL){
                printf("Error opening the image\n");
                return(1);
        }
        writeabsimage(output, width, height, ipropagatedimage, scanlinesize);
*/

	}	


	cudaDeviceReset();
	glfwTerminate();

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
