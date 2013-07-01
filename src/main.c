#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>

int main(int argc, char *argv[]){

	argc = argc;
	int error = 0;
	uint16 bps;
	uint32 width, height, array, length;
	TIFF *image;
		
	if((image = TIFFOpen(argv[1],"r")) == NULL){

		printf("Error opening the TIFF Image\n");
	}

	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bps);
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &length);

	printf("Bits Per Sample: %"PRId16", Image Width: %"PRId32", Image Length/Height: %"PRId32"\n", bps, width,length);  

	height = length;

	while(error != 0){
		TIFFReadRGBAImage(image, width, height, &array, error);
	}


	TIFFClose(image);


return(0);

}
