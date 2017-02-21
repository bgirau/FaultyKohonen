#include "malloc.h"

double ** calloc_2darray(int dim1, int dim2){
	double 	** arr;
	int i;

	arr = (double **) malloc(sizeof(double *) * dim1);
	for (i = 0; i < dim1; i++){
		arr[i] = (double *) calloc(dim2, sizeof(double));
	}
	return arr;
}
	
int ** malloc_2darray(int height, int width){
	int ** cl;
	int i;

	cl = (int **)malloc(sizeof(int *) * height);
	for (i = 0; i < height; i++){
		cl[i] = (int *)malloc(sizeof(int) * width); 
	}
	return cl;
}

double ** malloc_2darray_f(int height, int width){
	double ** cl;
	int i;

	cl = (double **)malloc(sizeof(double *) * height);
	for (i = 0; i < height; i++){
		cl[i] = (double *)malloc(sizeof(double) * width); 
	}
	return cl;
}

int *** malloc_3darray(int dim1, int dim2, int dim3){
	int *** 	arr;
	int 		i, j;
	arr = (int ***) malloc(sizeof(int **) * dim1);
	for (i = 0; i < dim1; i++){
		arr[i] = (int **) malloc(sizeof(int *) * dim2);
		for (j = 0; j < dim2; j++){
			arr[i][j] = (int *) malloc(sizeof(int) * dim3);
		}
	}
	return arr;
}

double *** malloc_3darray_f(int dim1, int dim2, int dim3){
    double *** 	arr;
    int 		i, j;
    arr = (double ***) malloc(sizeof(double **) * dim1);
    for (i = 0; i < dim1; i++){
        arr[i] = (double **) malloc(sizeof(double *) * dim2);
        for (j = 0; j < dim2; j++){
            arr[i][j] = (double *) malloc(sizeof(double) * dim3);
        }
    }
    return arr;
}

float **** malloc_4darray_f(int dim1, int dim2, int dim3, int dim4){
	float **** arr;
	int i, j, k;
	arr = (float ****) malloc(sizeof(float ***) * dim1);
	for (i = 0; i < dim1; i++){
		arr[i] = (float ***) malloc(sizeof(float **) * dim2);
		for(j = 0; j < dim2; j++){
			arr[i][j] = (float **) malloc(sizeof(float *) * dim3);
			for (k = 0; k < dim3; k++){
				arr[i][j][k] = (float *) malloc(sizeof(float) * dim4);
			}
		}
	}
	return arr;
}
