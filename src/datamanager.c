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


int splittingdoublematrix(double **mat, double ***img1, double ***img2){

	int indivimageheight, indivimagewidth;
	indivimageheight = 1040;
	indivimagewidth = 1392; 

	double **image1, **image2;

	createdoublematrix(&image1,indivimagewidth,indivimageheight);
	createdoublematrix(&image2,indivimagewidth,indivimageheight);

	for(int width=0; width != indivimagewidth; width++){
		
		for(int height=0; height != indivimageheight; height++){

			image1[width][height] = mat[width][height];
		}
 	}


	for(int width=0; width != indivimagewidth; width++){
	
		for(int height=1392; height != (1392*2); height++){
		
			image2[width][height] = mat[width][height];

		}
	}

	*img1 = image1;
	*img2 = image2;

	return(0);
}
