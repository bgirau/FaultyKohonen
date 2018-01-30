
#include "gss_filter.h"
/******************************************************************************
 * 		x_node ->
 *	y_node | 0 0 0 0
 *  	  \/ 0 0 0 0
 * 		   	 0 0 0 0
 *     		 0 0 0 0
 *
 ******************************************************************************/


float * gauss_kernel(int map_size, float std_dev, int node){
	int i;
	float sum = 0.0;
	float * kernel = calloc(map_size, sizeof(float));

	for (i = -node; i < map_size - node; i++){
		kernel[i+node] = exp(-i*i/(2*std_dev*std_dev));
		sum += kernel[i+node];
	}
	for(i = 0; i < map_size; i++){
		kernel[i] /= sum;
	}
	return kernel;
}


float * gauss_kernel_normalized(int map_size, float std_dev, int node){
	int i;
	float sum = 0.0;
	float * kernel = calloc(map_size, sizeof(float));

	for (i = -node; i < map_size - node; i++){
	  float x=i*1.0/map_size;
		kernel[i+node] = exp(-x*x/(2*std_dev*std_dev));
		sum += kernel[i+node];
	}
	for(i = 0; i < map_size; i++){
		kernel[i] /= sum;
	}
	return kernel;
}

void 		print_kernel(float *kernel, int map_size){
	// iterator
	int i;
	for( i = 0; i < map_size; i = i + 1){
		printf("%-f\t", kernel[i]);
	}
	printf("\n");
}
