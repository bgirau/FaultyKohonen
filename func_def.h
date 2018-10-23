/*
 *
 *
 *
 *
*/

#ifndef __FUNC_DEF_H__
#define __FUNC_DEF_H__

#include "pre_def.h"

//int min_precision_weights;

float 		mysqrt(double x);

Kohonen 	copy(Kohonen map);

void 		freeMap(Kohonen map);

void 		faulty_weights(Kohonen map, int p);

void 		faulty_bits(Kohonen map, int N);

int 		faulty_bit(Kohonen map);

void 		reverse_faulty_bit(Kohonen map,int b);

Kohonen 	init();

//Kohonen		init_pos(void);

int 		distance_L1(int *A,int *B,int n);

double 		distance(int * A, int * B, int n);

Winner 		recall(Kohonen map, int *input);

Winner 		recall_faulty_seq(Kohonen map, int *input,int p);

void 		printVALS(Kohonen map);

void 		printDNF(Kohonen map);

double		weightDNFbase(int i, int j, int i0, int j0);

double 		weightDNFJeremy(int i, int j, int i0, int j0, double A,double a,double B,double b);

double*		weightDNFbase1D(int i0,int size,double A,double a);

double*		weightDNFJeremy1D(int i0,int size,double A,double a);

void 		printDNFkernelJeremy(double A,double a,double B,double b);

void 		printDNFkernelbase();

void	 	updateDNF(Kohonen map,double A,double a,double B,double b,double h);

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

double      errorrate(Kohonen map, int ** inputs,int inp, int epoch);
double      evaldistortion(Kohonen map, int ** inputs,int inp, int epoch);

double      distortion_measure(Kohonen map, int ** inputs,int inp, double sig,int p);

void 		learn(Kohonen map, int ** inputs, int epoch);

void 		learn_NF(Kohonen map, int ** inputs, int epoch);

void 		learn_FI(Kohonen map, int ** inputs, int epoch);

int 		noise();

void 		learn_NI(Kohonen map, int ** inputs, int epoch);

void 		learn_threshold(Kohonen map, int ** inputs, int epoch);

double 	avg_quant_error(Kohonen map, int ** inputs,int inp, int p);

#endif //__FUNC_DEF_H__
