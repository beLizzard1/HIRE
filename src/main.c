#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <string.h>
<<<<<<< HEAD
=======

>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e
#include <tiffio.h>
#include <cuda.h>

#include <cuComplex.h>

#include "gpupropagate.h"
#include "gpudistancegrid.h"
#include "gpurefwavecalc.h"
#include "gpufouriertransform.h"
#include "writetoimage.h"
#define M_PI 3.14159265359

int main(int argc, char *argv[]){

<<<<<<< HEAD
	float planeseparation;
	planeseparation = 500;
=======
	TIFF *image, *img1;
	unsigned int width, length, row, x, y, offset, column;
	unsigned short bps, spp, pm;
	float *subtractedimage, *image2f, *dist2pinhole, pinholedist, k;
	tsize_t scanlinesize;
>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e

	TIFF *image, *output;
	unsigned int width, length, offset, height, i;
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

	
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bps);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &spp);
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &length);
<<<<<<< HEAD
	height = length / 2;
=======
	TIFFGetField(image, TIFFTAG_PHOTOMETRIC, &pm);
>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e
	printf("Image Properties: ");
	printf("BitsPerSample: %u, SamplesPerPixel: %u, Image Width: %u, Image Length: %u, Photometric: %u\n", bps, spp, width, length, pm);


<<<<<<< HEAD
	objsize = (scanlinesize) / (width * spp);

	uint16 *image1, *image2;
	uint16 *buffer;
	image1 = _TIFFmalloc(objsize * width * height);
	image2 = _TIFFmalloc(objsize * width * height);
	buffer = _TIFFmalloc(objsize * width);

=======
	printf("This data is actually comprised of two seperate images so we need to split them\n");
	float *buffer;
	float *image1, *image2;

	image1 = malloc(width * (length/2) * sizeof(float));
	image2 = malloc(width * (length/2) * sizeof(float));
	buffer = _TIFFmalloc(TIFFScanlineSize(image));
>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e

        if( image1 == NULL || image2 == NULL || buffer == NULL ){
		fprintf(stderr, "An Error occured while allocating memory for the images\n");
		return(0);
	}

	printf("Now to load the data from the provided image\n");

<<<<<<< HEAD
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
=======
	unsigned int height;
	height = length / 2;
	scanlinesize = (width * sizeof(float));

	for( row = 0; row < (height); row++){
		TIFFReadScanline(image,buffer,row,0);
		for(column = 0; column < (width - 1); column++){
			offset = (row * (width)) + column;
			image1[offset] = buffer[column];
/*			printf("%hu\n", image1[offset]);  */
		}
	}	      

	for( row = 1040; row < (length); row++){
		TIFFReadScanline(image,buffer,row,0);
		for(column = 0; column < (width - 1); column++){
			offset = ((row - 1040) * width) + column;
			image2[offset] = buffer[column]; 
/*			printf("%hu\n", image2[offset]); */
		}
	}

        if((img1 = TIFFOpen("img1.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writeimage(img1, width, height, image1, scanlinesize);

	        if((img1 = TIFFOpen("img2.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writeimage(img1, width, height, image2, scanlinesize);


	printf("Now to Subtract the images\n");
>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e

	longimg1 = malloc(sizeof(long) * width * height);
	longimg2 = malloc(sizeof(long) * width * height);
	fimg1 = malloc(sizeof(float) * width * height);
	fimg2 = malloc(sizeof(float) * width * height);

<<<<<<< HEAD
        for(offset = 0; offset < (width * height); offset++){
                longimg1[offset] = (long)image1[offset];
                longimg2[offset] = (long)image2[offset];
        }
=======
	subtractimages(image2, image1, subtractedimage, width, height);

		if((img1 = TIFFOpen("asubimage.tiff", "w")) == NULL){
			printf("Error opening image\n");
			return(1);
	}
	writeimage(img1, width, height, subtractedimage, scanlinesize);

	/* Testing the subtraction */
	cuComplex *cusubtracted, *ftcusubtracted;
	ftcusubtracted = malloc(sizeof(cuComplex) * width * height);
        cusubtracted = malloc(sizeof(cuComplex) * width * height);
	for(offset = 0; offset < (width * height); offset++){
		cusubtracted[offset].x = subtractedimage[offset];
	}

	gpufouriertransform(cusubtracted, ftcusubtracted, width, height);

		if((img1= TIFFOpen("ftsub.tiff", "w"))== NULL){
			printf("Error opening file \n");
			return(1);
	}
	writeimage(img1, width, height,ftcusubtracted, (width) * sizeof(cuComplex));

>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e

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



	free(fimg1);
	free(fimg2);

	pinholedist = 43048; /* Distance to the central pixel of the sensor in micron */
	pinholedist = 80000;

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


<<<<<<< HEAD
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
	
	printf("Writing the output.tiff\n");

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
	propagatedimage = malloc(sizeof(cuComplex) * width * height);
	ipropagatedimage = malloc(sizeof(cuComplex) * width * height);
	float dist, maxdist;
	char realdist[50], imagdist[50], absdist[50];
	maxdist = 40000;

	for(dist = 0; dist < maxdist; dist = dist + planeseparation){
=======
        if((img1 = TIFFOpen("reducedhologram.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writeimage(img1, width, height, reducedhologram, scanlinesize);

	/* Starting Fourier Transfrom stuff */

	cuComplex *treducedhologram;
	treducedhologram = malloc(sizeof(cuComplex) * width * height);
	gpufouriertransform(reducedhologram, treducedhologram, width, height);

        if((img1 = TIFFOpen("FTReducedHologram.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
        }
        writeimage(img1, width, height, treducedhologram, scanlinesize);

	char command[500];
	strcpy(command, "eog img1.tiff img2.tiff asubimage.tiff reducedhologram.tiff FTReducedHologram.tiff &");
	system(command);
	
>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e

	snprintf( realdist, 50, "/mnt/tmpfs/propagatedplanes/real/%lg.tiff", dist);
	snprintf( imagdist, 50, "/mnt/tmpfs/propagatedplanes/imag/%lg.tiff", dist);
	snprintf( absdist, 50, "/mnt/tmpfs/propagatedplanes/abs/%lg.tiff", dist);

	gpupropagate(tredhologram, propagatedimage, width, height, k, pixelsize,dist, scanlinesize);

	gpuifouriertransform(propagatedimage, ipropagatedimage, width, height);

        if((output = TIFFOpen(absdist,"w")) == NULL){
                printf("Error opening the image\n");
                return(1);
        }
        writeabsimage(output, width, height, ipropagatedimage, scanlinesize);

	if((output = TIFFOpen(realdist, "w")) == NULL){
		printf("Error opening the image\n");
		return(1);
	}
	writerealimage(output, width, height, ipropagatedimage, scanlinesize);

	if((output = TIFFOpen(imagdist, "w")) == NULL){
		printf("Error opening the image\n");
		return(1);
	}
	writecompleximage(output, width, height, ipropagatedimage, scanlinesize);

	}	
	



	cudaDeviceReset();

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
