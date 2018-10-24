
/*
to be precised : what is gaussian filtering ? what is DNF filtering ?
what is distortion with gaussian filtering ? what is quantization with gaussian filtering ? what is distortion with DNF filtering ? what is quantization with DNF filtering ?

Let us first consider the recall phase only.

Gaussian prototype : normalized gaussian filtering of the neural prototypes around the given position

Gaussian distance : normalized gaussian filtering of the distances between neural prototypes and the input around the given position (i.e. standard distortion is an average over all inputs of gaussian distances around the corresponding winner)

Gaussian protodistance : distance between the gaussian prototype associated to a position and the input

Gaussian gaussprotodistance : normalized gaussian filtering of the gaussian protodistances around the given position !!!

Gaussian winner : position for which the gaussian protodistance is minimal

was previously : position for which the gaussian distance is minimal

Gaussian quantization : average over inputs of distances between the input and the gaussian prototype around the gaussian winner (i.e. average of minimal protodistances)

Gaussian distortion : average over inputs of gaussian distances around the gaussian winer (only difference w.r.t. standard distortion: around the gaussian winner instead of around the winner)

but Gaussian filtering is meant to approximate DNF filtering.
Thus everything should be based on gaussian prototypes, not neural prototypes.

what is average (over inputs) of gaussian filtered distances between gaussian prototypes and input around the gaussian winner ? something like gaussian protodistortion (average of gaussian gaussprotodistances)

What seems the most meaningful:
- gaussian filtering stands for DNFbibble of activity
- with the DNF, the idea is to code prototypes by populations of neurons, each neural prototype being taken into account proportionally to the local activity
- we measure SOMs (neural prototypes) by aqe and distortion, thus we should apply the same measure to gaussian prototypes
- thus, we are interested in gaussian quantization and gaussian protodistortion
- if we take faults into account, a local fault on MSBF increases a lot the corresponding distance, so ???

DNF winner : emerging from competition

DNF prototype : normalized weighted average of the neural prototypes around the given position, weights=DNF activities (no given position)

DNF distance : normalized DNF filtering of the distances between neural prototypes and the input

DNF protodistance : distance between the DNF prototype and the input (no given position)

DNF quantization : average over inputs of DNF protodistances

DNF distortion : average over inputs of DNF distances, not very meaningful ???

DNF protodistortion : difficult to define since we can not associate a DNF prototype to each position, thus a possibility is to DNF filter the distances between gaussian prototypes and input, taking the excitatory part of the DNF as the gaussian filter (prototyping property of DNFs)

 */

#include "gss_som.h"

/* 
   gaussian prototype : 
   normalized gaussian filtering of the neural prototypes around the given position
*/
int* gauss_prototype(Kohonen map, float std_dev, int x_node, int y_node){

  
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

/*
  computes all gaussian prototypes (for all positions)
  optimized computation by separation of kernels for gaussian filtering
*/
void gaussian_prototypes(Kohonen map,float std_dev) {
  /* WARNING : dnf_weights used as temporary variables */
  float  *  kernel = (float *)gauss_kernel(2*map.size, std_dev, map.size);
  int i,j,k,i2;
  for(i = 0; i < map.size; i++)
    for (j = 0; j < map.size; j++)
      for (k=0;k<INS;k++) map.gss_weights[i][j][k] = 0.0;
  for(i = 0; i < map.size; i++){
    for (j = 0; j < map.size; j++) {
      float sum_kern=0.0;
      for (i2 = 0 ; i2 < map.size ; i2++) {
	float kern = kernel[map.size+i2-j];
	sum_kern += kern;
	for (k=0;k<INS;k++) map.gss_weights[i][j][k] += (int)(map.weights[i][i2][k]*kern);
      }
      for (k=0;k<INS;k++) map.gss_weights[i][j][k] = (int)(map.gss_weights[i][j][k]/sum_kern);
    }
  }
  for(i = 0; i < map.size; i++){
    for (j = 0; j < map.size; j++) {
      float sum_kern=0.0;
      for (k=0;k<INS;k++) map.dnf_weights[i][j][k] = 0;
      for (i2 = 0 ; i2 < map.size ; i2++) {
	float kern = kernel[map.size+i2-i];
	sum_kern += kern;
	for (k=0;k<INS;k++) map.dnf_weights[i][j][k] += (int)(map.gss_weights[i2][j][k]*kern);
      }
      for (k=0;k<INS;k++) map.dnf_weights[i][j][k] = (int)(map.dnf_weights[i][j][k]/sum_kern);
    }
  }
  for(i = 0; i < map.size; i++){
    for (j = 0; j < map.size; j++) {
      for (k=0;k<INS;k++) map.gss_weights[i][j][k] = map.dnf_weights[i][j][k];
    }
  }
}

/*
Gaussian distance : normalized gaussian filtering of the distances between neural prototypes and the input around the given position (i.e. standard distortion is an average over all inputs of gaussian distances around the corresponding winner)
 */
float gauss_distance(Kohonen map, int *B, int n, float std_dev, int x_node, int y_node){
  
  float			filtered_distance = 0.0;
  float  *  kernel_x = (float *)gauss_kernel(map.size, std_dev, x_node);
  float  *  kernel_y = (float *)gauss_kernel(map.size, std_dev, y_node);
  int 			i, j;
  float sum_kern=0.0;
  
  for(i = 0; i < map.size; i++){
    for (j = 0; j < map.size; j++)
      {
	float kern = kernel_x[j] * kernel_y[i];
	sum_kern += kern;
	filtered_distance += distance_L1(map.weights[i][j], B, n) * kern / one;
      }
  }
  
  free(kernel_x);
  free(kernel_y);
  return filtered_distance/sum_kern;
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

  /* we assume that the gaussian prototypes have been computed */
  // gaussian_prototypes(map,SIGMA_GAUSS);
  for (k = 0;k < inp; k++) {
    normalise=0.0;
    distortion=0.0;
    win = recallGSS(map, inputs[k]);
    for (i = 0; i < map.size; i++) {
      for (j = 0;j < map.size; j++) {
        dx    = 1.0 * (i - win.i) / map.size;
        dy    = 1.0 * (j - win.j) / map.size;
        coeff = exp(-1 * (dx * dx + dy * dy) / (2 * sig * sig));
	normalise+=coeff;
        dist  = distance_L1(inputs[k], map.weights[i][j], map.nb_inputs)/(1.0*one);
        distortion += coeff * dist * dist;
      }
    }
    global_distortion+=distortion/normalise;
  }
  return global_distortion/inp;
}

double protodistortion_measure_GSS(Kohonen map, int** inputs,int inp, double sig) {
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

  /* we assume that the gaussian prototypes have been computed */
  // gaussian_prototypes(map,SIGMA_GAUSS);
  for (k = 0;k < inp; k++) {
    normalise=0.0;
    distortion=0.0;
    win = recallGSS(map, inputs[k]);
    for (i = 0; i < map.size; i++) {
      for (j = 0;j < map.size; j++) {
        dx    = 1.0 * (i - win.i) / map.size;
        dy    = 1.0 * (j - win.j) / map.size;
        coeff = exp(-1 * (dx * dx + dy * dy) / (2 * sig * sig));
	normalise+=coeff;
        dist  = distance_L1(inputs[k], map.gss_weights[i][j], map.nb_inputs)/(1.0*one);
        distortion += coeff * dist * dist;
      }
    }
    global_distortion+=distortion/normalise;
  }
  return global_distortion/inp;
}

double avg_quant_error_GSS(Kohonen map, int ** inputs,int inp){
  Winner win;
  int i;
  float error=0.0;

  /* we assume that the gaussian prototypes have been computed */
  // gaussian_prototypes(map,SIGMA_GAUSS);
  for (i = 0; i < inp; i++){
    win = recallGSS(map,inputs[i]);
    error += distance_L1(inputs[i], map.gss_weights[win.i][win.j],INS)/(1.0*one);
  }
  error /= inp;
  return error;
}

Winner recallGSS(Kohonen map, int *input) {
  /* computes the soft winner, i.e. the neuron which gaussian prototype is at minimum distance from the given input (integer or fixed point) */
  /* we assume that the gaussian prototypes have been computed */
  int min = distance_L1(input, map.gss_weights[0][0],map.nb_inputs);
  int min_i = 0, min_j = 0;
  int i,j;
  
  for (i = 0; i < map.size; i++) {
    for (j = 0; j < map.size; j++) {
      int dist = distance_L1(input, map.gss_weights[i][j], map.nb_inputs);
      map.dnf[i][j] = 0;
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
  printf("(GSS) aqe after %d learning iterations : %f\n", 
            epoch * NBITEREPOCH, aqe);
	return aqe;
}

double evaldistortionGSS(Kohonen map, int ** inputs, int inp,int epoch) {

  double aqe = distortion_measure_GSS(map, inputs, inp, SIGMA_GAUSS);
    printf("(GSS) distortion after %d learning iterations : %f\n",
           epoch * NBITEREPOCH, aqe);
    return aqe;
}

double evalprotodistortionGSS(Kohonen map, int ** inputs, int inp,int epoch) {

  double aqe = protodistortion_measure_GSS(map, inputs, inp, SIGMA_GAUSS);
    printf("(GSS) protodistortion after %d learning iterations : %f\n",
           epoch * NBITEREPOCH, aqe);
    return aqe;
}

