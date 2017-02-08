#include "malloc.h"
	
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
