#include "gss_som.h"


int* gauss_prototype(Kohonen map, int n, float std_dev, int x_node, int y_node){

	
	float  *  kernel_x = (float *)gauss_kernel(map.size, std_dev, x_node);
  float  *  kernel_y = (float *)gauss_kernel(map.size, std_dev, y_node);
	int 			i, j,k;
	int  *    proto = calloc(INS, sizeof(int));
	float     sum_kern = 0.0;

	for(i = 0; i < map.size; i++){
		for (j = 0; j < map.size; j++)
		{
		  float kern = kernel_x[j] * kernel_y[i];
		  sum_kern += kern;
		  for (k=0;k<INS;k++) proto[k] += (int)(map.weights[i][j][k]*kern);
		}
	}
	for (k=0;k<INS;k++) proto[k] = (int) (proto[k]/sum_kern);

  free(kernel_x);
  free(kernel_y);

	return proto;
}

float gauss_distance(Kohonen map, int *B, int n, float std_dev, int x_node, int y_node){

	float			filtered_distance = 0.0;
	float  *  kernel_x = (float *)gauss_kernel(map.size, std_dev, x_node);
    float  *  kernel_y = (float *)gauss_kernel(map.size, std_dev, y_node);
	int 			i, j;

	for(i = 0; i < map.size; i++){
		for (j = 0; j < map.size; j++)
		{
			filtered_distance += distance(map.weights[i][j], B, n) * kernel_x[j] * kernel_y[i];
		}
	}

  free(kernel_x);
  free(kernel_y);
    return filtered_distance;
}

double distortion_measure_GSS(Kohonen map, int** inputs,int inp, double sig) {
// sig = (0.2 + 0.01)/2*MAPSIZE
// sig = 0.1*SIZE
  Winner win;
  int    i, j, k;
  double dx;
  double dy;
  double coeff;
  double dist;
  double distortion;
  double global_distortion=0.0;
  double normalise;

  for (k = 0;k < inp; k++) {
    normalise=0.0;
    distortion=0.0;
    win = recallGSS(map, inputs[k],sig);
    for (i = 0; i < map.size; i++) {
      for (j = 0;j < map.size; j++) {
        dx    = 1.0 * (i - win.i) / map.size;
        dy    = 1.0 * (j - win.j) / map.size;
        coeff = exp(-1 * (dx * dx + dy * dy) / (2 * sig * sig));
	normalise+=coeff;
        dist  = distance(inputs[k], map.weights[i][j], map.nb_inputs);
        distortion += coeff * dist * dist;
      }
    }
    global_distortion+=distortion/normalise;
  }
  return global_distortion/inp;
}

double avg_quant_error_GSS(Kohonen map, int ** inputs,int inp){
  Winner win;
  int i, j;
  double error=0.0;

  for (i = 0; i < inp; i++){
    win = recallGSS(map,inputs[i], SIGMA_GAUSS);
    error += distance(inputs[i], gauss_prototype(map,map.nb_inputs, SIGMA_GAUSS, win.j, win.i),INS);
  }
  error /= inp;
  return error;
}

Winner recallGSS(Kohonen map, int *input, float std_dev) {
  /* computes the soft winner, i.e. the neuron that is at minimum gaussian-filtered distance from the given input (integer or fixed point) */
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


double errorrateGSS(Kohonen map, int ** inputs,int inp, int epoch) {
  
  double aqe = avg_quant_error_GSS(map, inputs, inp);
  printf("(GSS)learn aqe after %d learning iterations : %f\n", 
            epoch * NBITEREPOCH, aqe);
	return aqe;
}

double evaldistortionGSS(Kohonen map, int ** inputs, int inp,int epoch) {

  double aqe = distortion_measure_GSS(map, inputs, inp, SIGMA_GAUSS);
    printf("learn distortion GSS after %d learning iterations : %f\n",
           epoch * NBITEREPOCH, aqe);
    return aqe;
}

