#include <stdio.h>
#include <stdlib.h>


int createdoublematrix(double ***mat, unsigned short int rows, unsigned short int cols){

	double **matrix;
		
		if((rows == 0 || cols == 0)){
			printf("Something has gone wrong, you didnt provide size information\n");
		}

	matrix = (double **)calloc(cols, sizeof(double *));
	
	for(unsigned int i = 0; i < cols; i++){

		matrix[i] = (double *)calloc(rows, sizeof(double));

	}

	if(matrix == NULL){
		
		printf("Error no memory was allocated!\n");
		exit(1);

	}

	*mat = matrix;
	
	return(0);
}


int freedoublematrix(double **mat, unsigned int cols){

	for(unsigned int i=0; i < cols; i++){
		free(mat[i]);
	}

free(mat);
return(0);
}
