/*
 * 
 *
02.06.2017 : test avec normal_dataset
 *
 *
*/

#ifndef __PRE_DEF_H__
#define __PRE_DEF_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIZE 32

/* CONSTANTES POUR DNF */
#define dt_tau_Jeremy 0.774292
#define A_Jeremy 0.805076
#define a_Jeremy 3.95191
#define B_Jeremy 0.494946
#define b_Jeremy 33.8058
#define h_Jeremy 0.514081
#define NBITERDNF 100
#define TAU_DNF 0.2

/* CONSTANTES D'EXPERIMENTATION */
#define INS 3
#define DISTRIB normal3d_dataset
#define DISTRIBNAME "gaussians3D"
#define OPTIMIZED_WEIGHTS 0 /* 1 if faults limited to fractional part and significant integer part */
#define INDIVIDUAL_WEIGHTS 1 /* 1 if faults limited to significant bits of each weight */
#define WITH_TECHS 1 /* 1 if testing all fault-tolerant learning techniques */
#define SEQUENTIAL 0 /* 1 to test faults in a sequential implementation */
#define NBMAPS 25
#define NBMAPINITS 5 /* how many different initial maps
			 NBMAPS=NBMAPINITS*number of random inputs 
			                   for each learning iteration 
		       */
#define MAXFAULTPERCENT 9
#define nb_experiments 5 /* number of faulty versions of the same map */

/* CONSTANTES POUR GSS */
#define SIGMA_GAUSS 3.95

/* CONSTANTES D'APPRENTISSAGE */
#define NBEPOCHLEARN 100
#define NBITEREPOCH 50
#define TEST_DENSITY 20 /* number of test patterns per dimension */
#define TEST2_DENSITY 20 /* number of test patterns per dimension for fault tolerance evaluation */
#define PI 3.14159
#define TAU 0.3
#define TAUMIN 0.07
#define MINDIST 0.00001
#define FI_LEVEL 1 /* level of fault injection (number of bits) during learning */
#define NI_LEVEL 0.01 /* level of noise injection during learning */

/* CONSTANTES DE CONVERSION EN VIRGULE FIXE */
#define precision 16 /* precision of weights/inputs/values, 1 bit for sign coding */
//#define fractional 12 /* size of the fractional part */
//#define one 4096 /* fixed point value for 1.0 */
#define precision_int 16384 /* precision of weights/inputs/fractional part of intermediate computations */
#define fractional 10 /* size of the fractional part */
#define one 1024 /* fixed point value for 1.0 */
#define REST -0.15
#define k_w 1.1
#define W_E 1
#define W_I 1.5
#define k_s 0.5
#define SIGMA_E 0.25
#define SIGMA_I 0.4
#define ALPHA 0.5
#define EPS_SQRT -0.000001

/* AUTRES CONSTANTES */
#define TESTDIV 5
#define NBCLASSES 3
#define VERBOSE 0

typedef struct kohonen {
  int size;
  int nb_inputs;
  int ***weights;
  int ***gss_weights;
  int ***dnf_weights;
  int **vals;
  int **dnf;
  int *FI; //FI_i,FI_j,FI_k,FI_b
} Kohonen;

typedef struct winner {
  int i;
  int j;
  int value;
} Winner;

typedef struct {
  float i;
  float j;
} WinnerDNF;


#endif	//__PRE_DEF_H__
