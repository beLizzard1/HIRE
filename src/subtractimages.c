#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>
#include <tiffio.h>


int subtractimages(void *image1,void *image2, float *subtractedimage, unsigned int width, unsigned int height){

	unsigned int offset, x, y;

	for( x = 0; x < width; x ++){
		for(y=0; y < height; y++){
			offset = (y * width) + x;
/*			printf("%f\n", (float)image1[offset] - (float)image2[offset]); */
			printf("");
			/*subtractedimage[offset] = image1[offset] - image2[offset]; */
		}
	}

	return(0);
}
