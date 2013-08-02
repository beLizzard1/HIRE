#define _USE_MATH_DEFINES
#define GLFW_INCLUDE_GLU
#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <tiffio.h>
#include <cuda.h>
#include <cuComplex.h>

#include <GL/glfw.h>

#include "gpupropagate.h"
#include "gpudistancegrid.h"
#include "gpurefwavecalc.h"
#include "gpufouriertransform.h"
#include "writetoimage.h"
#define M_PI 3.14159265358979323846264338327

int normalisecucomplex(float *absipropagatedimage, cuComplex *ipropagatedimage, unsigned int width, unsigned int height){
	unsigned int offset;
	float max;
	for(offset = 0; offset < (width * height); offset++){
				absipropagatedimage[offset] = sqrtf( (ipropagatedimage[offset].x * ipropagatedimage[offset].x) + (ipropagatedimage[offset].y * ipropagatedimage[offset].y));
				if( max < absipropagatedimage[offset]){
					max = absipropagatedimage[offset];
				}
			}

			for(offset = 0; offset < (width * height); offset++){
				absipropagatedimage[offset] = absipropagatedimage[offset] / max;
			}
return(0);
}
void normalise_and_convert_8(float *p, size_t size, unsigned char *out)
{
    if(p && out)
    {
        size_t i;
        float max = 0.0;
        /* Get max value */
        for(i=0;i<size;i++)
        {
            if(p[i] > max)
                max = p[i];
        }

        fprintf(stderr, "max value is %f\n", max);
        if(max > 0.0)
        {
            for(i=0;i<size;i++)
            {
                float val;
                unsigned char cv;
                val = (*p++ / max) * 255.0;

                if(val <= 0.0)
                {
                    cv = 0;
                }
                else if(val >= 255.0)
                {
                    cv = 255;
                }
                else
                {
                    cv = floor(val);
                }
                *out++= cv;
                *out++= cv;
                *out++= cv;
            }
        }
        else
        {
            /* black buffer */
            memset(out, 0, size * 3);
        }
    }
}

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

    cuComplex *tsub;
    tsub = malloc(sizeof(cuComplex) * width * height);

    for(offset = 0; offset < (width * height); offset++){
                tsub[offset].x = fimg1[offset] - fimg2[offset];
                tsub[offset].y = 0;
    }

	if((output = TIFFOpen("sub.tiff", "w")) == NULL){
                printf("Error opening image\n");
                return(1);
    }
	writerealimage(output, width, height, tsub, scanlinesize);

//	Now we need to expand the size of the subtracted image by a factor t
	int scalefactor = 1;
	unsigned int newwidth, newheight, oldoffset;
	newwidth = scalefactor * width;
	newheight = (scalefactor * height);
	cuComplex *sub;
	sub = malloc(sizeof(cuComplex) * newwidth * newheight);

	for(unsigned int x = 0; x < newwidth; x++){
		for(unsigned int y = 0; y < newheight; y++){
			oldoffset = width * floor((y / scalefactor)) + floor((x / scalefactor));
			offset = newwidth * y + x;

			sub[offset].x = tsub[oldoffset].x;
			sub[offset].y = tsub[oldoffset].y;
			}
	}
	free(tsub);

//	Now ensuring that things don't break changing the values of width & height to reflect the expanded matrices
	width = newwidth;
	height = newheight;

	/* Calculating the mean value of the subtracted image and subtract it */
	float sum = 0;
	for(offset = 0; offset < (width * height); offset++){
		sum += sub[offset].x;	
	}
	sum = (sum / (width * height));

	for(offset = 0; offset < (width * height); offset++){
		sub[offset].x -= sum;
	}
   	free(fimg1);
	free(fimg2);

	pinholedist = 57738; /* Distance to the central pixel of the sensor in micron */

	cuComplex *referencewave;

	referencewave = malloc(sizeof(cuComplex) * width * height);	

	gpurefwavecalc(referencewave, width, height, pinholedist,k,pixelsize);


	/* Dividing the Subtracted Image by the Reference Wave */

	cuComplex *reducedhologram;
	reducedhologram = malloc(sizeof(cuComplex) * width * height);

	for(offset = 0; offset < (width * height); offset++){
		reducedhologram[offset].x = (sub[offset].x * referencewave[offset].x) / ((referencewave[offset].x * referencewave[offset].x) + (referencewave[offset].y * referencewave[offset].y));

		reducedhologram[offset].y = (sub[offset].x * referencewave[offset].y) / ((referencewave[offset].x * referencewave[offset].x) + (referencewave[offset].y * referencewave[offset].y));


	}

	/* Starting Fourier Transform stuff */

	cuComplex *tredhologram;
	tredhologram = malloc(sizeof(cuComplex) * width * height);
	gpufouriertransform(reducedhologram, tredhologram, width, height);


 	/* Doing another fourier transform, to see if the input and output are the same */
	cuComplex *itredhologram;
	itredhologram = malloc(sizeof(cuComplex) * width * height);
	gpuifouriertransform(tredhologram, itredhologram, width, height);

	/* Propagating the transformed image */
	cuComplex *propagatedimage, *ipropagatedimage;
	float *absipropagatedimage;
	propagatedimage = malloc(sizeof(cuComplex) * width * height);
	ipropagatedimage = malloc(sizeof(cuComplex) * width * height);
	absipropagatedimage = malloc(sizeof(float) * width * height);
	float dist, maxdist;
	char absdist[50];
	unsigned char *rgbbuffer;
	maxdist = pinholedist;

	rgbbuffer = malloc(sizeof(unsigned char)* width * height * 3);

	/* GLFW Interface Stuff */

	if(!glfwInit()){
		exit(EXIT_FAILURE);
	}

	if(!glfwOpenWindow(width,height,0,0,0,0,0,0, GLFW_WINDOW)){
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	int windowwidth, windowheight;

	/* Propagation Loop */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	GLuint tex_id = 0;  glGenTextures(1, &tex_id);

	while(dist < maxdist){
	glfwGetWindowSize(&windowwidth, &windowheight);
	glViewport(0, 0, windowwidth, windowheight);

	/* Propagation Loop */
	for(dist = 30000; dist <= maxdist; dist = dist + planeseparation){

		snprintf( absdist, 50, "PropagatedDistance: %f micron", dist);
		glfwSetWindowTitle(absdist);	
		gpupropagate(tredhologram, propagatedimage, width, height, k, pixelsize,dist, scanlinesize, scalefactor);
		gpuifouriertransform(propagatedimage, ipropagatedimage, width, height);

		for(offset = 0; offset < (width * height); offset++){
		absipropagatedimage[offset] = sqrtf( (ipropagatedimage[offset].x * ipropagatedimage[offset].x) + (ipropagatedimage[offset].y * ipropagatedimage[offset].y));
		}

		normalise_and_convert_8(absipropagatedimage, width * height, rgbbuffer);

		glBindTexture(GL_TEXTURE_2D, tex_id);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgbbuffer);

		/* Drawing the results in opengl */
		glDisable(GL_LIGHTING);
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgbbuffer);

		glBegin(GL_TRIANGLE_STRIP);

		glTexCoord2f(0,0); glVertex3f(0,0,0); glTexCoord2f(1,0); glVertex3f(1,0,0); glTexCoord2f(0,1);
		glVertex3f(0,1,0); glTexCoord2f(1,1); glVertex3f(1,1,0);

		glEnd();

		glfwSwapBuffers();
	}
}

	glfwTerminate();

	/* Freeing stuff that has been allocated to the host */

	_TIFFfree(buffer);
	TIFFClose(image);

	return(0);
}
