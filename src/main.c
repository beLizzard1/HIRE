#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <tiffio.h>

#include <tiffmanager.h>

int main(int argc, char **argv){

	if( (argc != 2) ){

		printf("Please provide the correct operating parameters\n");
		printf("Run the program with the .tiff tile you require\n");
		return(1);
	}

	printf("This is an implementation of the MATLAB holographic reconstruction program in C\n");
	printf("Later it will be enhanced with CUDA calls to improve performance\n");
	

	TIFF *image;
	
	image = TIFFOpen(argv[1],"r");

	
	if((image == NULL)){
	
		printf("Could not open the image!\n");
		return(1);
	}


	TIFFSetDirectory(image,0);


	TIFFClose(image);
return(0);
}
