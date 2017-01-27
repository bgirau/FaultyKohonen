#include "gss_som.h"


float gauss_distance(Kohonen map, int *B, int n, float std_dev, int x_node, int y_node){

	float			filtered_distance = 0.0;
	float	**	kernel = (float **)gauss_kernel(map.size, std_dev, x_node, y_node);
	int 			i, j;

	for(i = 0; i < map.size; i++){
		for (j = 0; j < map.size; j++)
		{
			filtered_distance += distance(map.weights[i][j], B, n) * kernel[i][j];
		}
	}
	return filtered_distance;
}

double distortion_measure_GSS(Kohonen map, int ** inputs, double sig) {
// sig = (0.2 + 0.01)/2*MAPSIZE
// sig = 0.1*SIZE
  Winner win;
  int    i, j, k;
  double dx;
  double dy;
  double coeff;
  double dist;
  double distortion = 0.0;

  for (k = 0;k < NBITEREPOCH; k++) {
    win = recallGSS(map, inputs[k], 1.0);
    for (i = 0; i < map.size; i++) {
      for (j = 0;j < map.size; j++) {
        dx    = 1.0 * (i - win.i);
        dy    = 1.0 * (j - win.j);
        coeff = exp(-1 * (dx * dx + dy * dy) / (2 * sig * sig));
        dist 	= gauss_distance(map, inputs[k], map.nb_inputs, 1.0, j, i);
        distortion += (int) coeff * dist * dist;
      }
    }
  }
  // printf("SOM distortion measure = %f \n", distortion);
  return distortion;
}

Winner recallGSS(Kohonen map, int *input, float std_dev) {
  /* computes the winner, i.e. the neuron that is at minimum distance from the given input (integer or fixed point) */
	int min = gauss_distance(map, input, map.nb_inputs, std_dev, 0, 0);
 	int min_i = 0, min_j = 0;
  	int i,j,k;

  	for (i = 0; i < map.size; i++) {
    	for (j = 0; j < map.size; j++) {
      		float dist = gauss_distance(map, input, map.nb_inputs, std_dev, j, i);
      		// map.dnf[i][j] = 0;
      		if (dist < min) {
				min 	= dist;
				min_i 	= i;
				min_j	= j;
      		}
    	}
  	}
  	Winner win;
  	
  	win.i 		= min_i;
  	win.j 		= min_j;
  	win.value = min;

  	return win;
}


void errorrateGSS(Kohonen map, int ** inputs, double * distortion, double epoch) {
  
  distortion[epoch] = distortion_measure_GSS(map, inputs, 1.0);
  printf("learn distortion after %d learning iterations : %f\n", 
            epoch * NBITEREPOCH, distortion[epoch]);
}

