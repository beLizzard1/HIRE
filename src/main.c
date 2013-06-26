#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "datamanager.h"
#include <time.h>

int main(int argc, char **argv){

	if( (argc != 2) ){

		printf("Please provide the correct operating parameters\n");
		printf("Run the program with the image file you require\n");
		printf("to be reconstructed\n");
		return(1);
	}

	printf("This is an implementation of the MATLAB holographic reconstruction program in C\n");
	printf("Later it will be enhanced with CUDA calls to improve performance\n");


	argv=argv;
	
	
	double** array;
	unsigned short int width, height;
	width = 1392;
	height = 2080;

	createdoublematrix(&array,width,height);


	/* So we have a big 2d matrix allocated in the Host memory we want to split this matrix in half so we can deal with the two images contained within the array */

	double **image1, **image2;

	splittingdoublematrix(array,&image1,&image2);

	freedoublematrix(array,height);

	/* Now we have the two separate arrays loaded into the Host memory */


return(0);	
}
