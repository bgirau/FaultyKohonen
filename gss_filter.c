
#include "gss_filter.h"
/******************************************************************************
 * 		x_node ->
 *	y_node | 0 0 0 0
 *  	  \/ 0 0 0 0
 * 		   	 0 0 0 0
 *     		 0 0 0 0
 *
 ******************************************************************************/


float 	**	gauss_kernel(int map_size, float std_dev, int x_node, int y_node){
	// iterators
	int 	i,j;
	double	sum = 0.0;
	// allocate memory for kernel
	float **kernel = malloc(sizeof(double *) * map_size);
	for (i = 0; i < map_size; i = i + 1){
		kernel[i] = malloc(map_size * sizeof(float));
	}

	for(i = -y_node; i < map_size - y_node; i = i + 1){
		for (j = -x_node; j < map_size - x_node; j = j + 1){
			kernel[i+y_node][j+x_node] = (1-(i*i+j*j)/(2*std_dev*std_dev))*exp(-(i*i + j*j)/(2*std_dev*std_dev))/(2*PI*std_dev*std_dev);
			// kernel[i+y_node][j+x_node] = exp(-(i*i + j*j)/(2*std_dev*std_dev))/(2*PI*std_dev*std_dev);
			// sum += kernel[i+y_node][j+x_node]; // normalization
		}
	}
	// for (i = 0; i < map_size; i++){
	// 	for (j = 0; j < map_size; j++){
	// 		kernel[i][j] /= sum;
	// 	}
	// }

	return	kernel;
}

void 		print_kernel(float **kernel, int map_size){
	// iterator
	int i, j;
	for( i = 0; i < map_size; i = i + 1){
		for ( j = 0; j < map_size; j = j + 1){
			printf("%-f\t", kernel[i][j]);
		}
		printf("\n");
	}
}
