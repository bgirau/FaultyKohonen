/* 
 *	File: custom_rand.h
 *	
 * 	Contains random generator with uniform and normal distribution
 *
*/ 


#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "pcg_basic.h"

#ifndef __CUSTOM_RAND_H__
#define __CUSTOM_RAND_H__

#define PI 3.14159

/*
 *	Pass:
 * 		size - size of the output vector
 *	Return
 * 		poiner to the vector
*/

double * 	uniform_random(pcg32_random_t * rng, int size);

double * 	normal_random(pcg32_random_t * rng, double mu, double sigma, int size);

void 			print_distrib(double * array, int size);

double *  normal_dataset(pcg32_random_t * rng);

#endif //__CUSTOM_RAND_H__