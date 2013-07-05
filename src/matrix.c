#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <tiffio.h>


int creatematrix(uint16 ***mat,uint32 width,uint32 height){

	uint16 **matrix;
	unsigned int i;

	if((width == 0 || height == 0)){
		printf("Something has gone wrong, no width / height was provided\n");
		return(1);
	}

	matrix = (uint16 **)calloc(width, sizeof(uint16 *));
	
	for (i = 0; i < width;i++ ){

		matrix[i] = (uint16 *)calloc(height, sizeof(uint16));

	}


	*mat = matrix;
	return(0);
}

int freematrix(uint16 **mat, uint32 width){
	unsigned int i;

	for(i=0; i < width; i++){
		free(mat[i]);
	}

free(mat);

return(0);

}
