#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int coordtoraster(unsigned int width, unsigned int x, unsigned int y){
	unsigned int raster;
	
	raster = ( ( y * width ) + x);

	return(raster);

}
