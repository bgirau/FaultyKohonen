/*
 * 
 *
 *
 *
*/

#ifndef __PRE_DEF_H__
#define __PRE_DEF_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define VERBOSE 0
#define PI 3.14159
#define SIZE 8
#define INS 4
#define NBCLASSES 3
#define TESTDIV 5
#define TAU 0.3
#define TAUMIN 0.07
#define MINDIST 0.00001
#define NBEPOCHLEARN 10
#define NBITEREPOCH 10
#define FI_LEVEL 1 /* level of fault injection (number of bits) during learning */
#define NI_LEVEL 0.01 /* level of noise injection during learning */
#define NBMAPS 1
#define MAXFAULTPERCENT 2
#define NBVALIDPARTITIONS 1
#define FILENAME "weights8x8-4.txt"
#define INPUTFILENAME "inputs-4.txt"
#define precision 16 /* precision of weights/inputs/values, 1 bit for sign coding */
#define fractional 10 /* size of the fractional part */
#define one 1024 /* fixed point value for 1.0 */
#define precision_int 16384 /* precision of weights/inputs/fractional part of intermediate computations */
#define nb_experiments 4 /* number of faulty versions of the same map */
#define REST -0.15
#define k_w 1.1
#define W_E 1
#define W_I 1.5
#define k_s 0.5
#define SIGMA_E 0.25
#define SIGMA_I 0.4
#define TAU_DNF 0.2
#define ALPHA 0.5
#define NBITERDNF 100
#define EPS_SQRT -0.000001

/* computation time for 1000 exp. and 1000 test. = 1.8s for 10 different percentages
   linear time / INS, expe., test.
   less than quadratic time / SIZE */

typedef struct kohonen {
  int size;
  int nb_inputs;
  int ***weights;
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