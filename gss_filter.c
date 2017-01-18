
#include "gss_filter.h"

/* Apply gauss_filter to Kohonen map */

Kohonen 	gauss_filter(Kohonen map, float std_dev){
	// iterator
	int 	i, j, k;
	double 	**kernel;

	for (i = 0; i < map.size; i = i + 1){
		for (j = 0; j < map.size; j = j + 1){
			kernel = (double **) gauss_kernel(map.size, std_dev, j, i);
			for (k = 0; k < map.nb_inputs; k = k + 1){
				map.weights[i][j][k] = map.weights[i][j][k] * kernel[i][j];
			}
		}
	}

	return map;

}
/******************************************************************************
 * 		x_node ->
 *	y_node | 0 0 0 0
 *  	  \/ 0 0 0 0
 * 		   	 0 0 0 0
 *     		 0 0 0 0
 *
 ******************************************************************************/


double 	**	gauss_kernel(int map_size, double std_dev, int x_node, int y_node){
	// iterators
	int 	i,j;
	// double	sum;
	// allocate memory for kernel
	double **kernel = malloc(sizeof(double *) * map_size);
	for (i = 0; i < map_size; i = i + 1){
		kernel[i] = malloc(map_size * sizeof(double));
	}

	for(i = -y_node; i < map_size - y_node; i = i + 1){
		for (j = -x_node; j < map_size - x_node; j = j + 1){
			kernel[i+y_node][j+x_node] = exp(-(i*i + j*j)/(2*std_dev*std_dev));//
														//(2*PI*std_dev*std_dev);
			// sum = sum + kernel[i+kernel_size/2][j+kernel_size/2]; // normalization
		}
	}
	return	kernel;
}

void 		print_kernel(double **kernel, int map_size){
	// iterator
	int i, j;
	for( i = 0; i < map_size; i = i + 1){
		for ( j = 0; j < map_size; j = j + 1){
			printf("%-f\t", kernel[i][j]);
		}
		printf("\n");
	}
}
