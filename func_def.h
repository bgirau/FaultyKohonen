/*
 *
 *
 *
 *
*/

#ifndef __FUNC_DEF_H__
#define __FUNC_DEF_H__

#include "pre_def.h"

float 		mysqrt(double x);

Kohonen 	copy(Kohonen map);

void 		freeMap(Kohonen map);

void 		faulty_weights(Kohonen map, int p);

void 		faulty_bits(Kohonen map, int N);

void 		faulty_bit(Kohonen map);

void 		reverse_faulty_bit(Kohonen map);

Kohonen 	init();

Kohonen		init_pos(void);

int 		distance(int *A, int *B, int n);

Winner 		recall(Kohonen map, int *input);

void 		printVALS(Kohonen map);

void 		printDNF(Kohonen map);

double		weightDNFbase(int i, int j, int i0, int j0);

double 		weightDNF(int i, int j, int i0, int j0, double s_e, double s_i);

void 		printDNFkernel(double s_e,double s_i);

void 		printDNFkernelbase(void);

void	 	updateDNF(Kohonen map,double sig_e,double sig_i);

void 		initVALS(Kohonen map,int *input);

WinnerDNF 	recallDNF(Kohonen map,int *input);

int 		*prototypeDNF(Kohonen map);

void 		gaussianlearnstep(Kohonen map,int *input,double sig,double eps);

void 		NFlearnstep(Kohonen map,int *input,double sig,double eps);

void 		gaussianlearnstep_threshold(Kohonen map,int *input,double sig,double eps);

void 		heavisidelearnstep(Kohonen map,int *input,int radius,double eps);

void 		NN1neuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp);

void 		NN5neuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp);

int 		NN5DNFclass(Kohonen map,int **in,int **crossvalid,int testbloc,int inp);

int 		DNFclass(Kohonen map,int **in,int **crossvalid,int testbloc,int inp);

void 		MLneuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp);

void 		neuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp);

void 		printneuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp);

void 		errorrateDNF(Kohonen map,int** inputs,int inp,int** classe,int it,int **crossvalid,int testbloc);

void 		errorrate(Kohonen map, int ** inputs, double * distortion, int epoch);

void 		learn(Kohonen map, int ** inputs, int epoch);

void 		learn_NF(Kohonen map, int ** inputs, int epoch);

void 		learn_FI(Kohonen map, int ** inputs, int epoch);

int 		noise();

void 		learn_NI(Kohonen map, int ** inputs, int epoch);

void 		learn_threshold(Kohonen map, int ** inputs, int epoch);

double distortion_measure(Kohonen map, int** inputs, double sig);


#endif //__FUNC_DEF_H__
