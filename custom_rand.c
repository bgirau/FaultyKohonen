
#include "custom_rand.h"

void  init_random(pcg32_random_t * rng){
	int rounds = 5;
	pcg32_srandom_r(rng, time(NULL) ^ (intptr_t)&printf, 
										(intptr_t)&rounds);
}

double * uniform_random(pcg32_random_t * rng, int size){
	int i;
	double * random_seq = malloc(sizeof(double) * size);
	for (i = 0; i < size; i++){
		random_seq[i] = ldexp(pcg32_random_r(rng), -32);	
	}
	return random_seq;
}

void 	print_distrib(double * array, int size){
	int i;
	for (i = 0; i < size; i++) printf("%-f\n",array[i]);
	printf("\n");
}

double * normal_random(pcg32_random_t * rng, double mu, 
													double sigma, int size){
	double 	*	random_seq = malloc(sizeof(double) * size);
	double	* u = uniform_random(rng, size+1); 
	double 		Z;
	int 			i;
	for(i = 0; i < size; i++){
		Z = sqrt(-2 * log(u[i])) * cos(2*PI*u[i+1]);
		random_seq[i] = Z * sigma + mu;
	}
	return random_seq;
}

double *  normal_dataset(pcg32_random_t * rng){
	double * vector = malloc(sizeof(double) * 2);
	static int it_gss = 0;
	double var;

	switch(it_gss){
		case 0:{
			var = fabs(normal_random(rng, 0.35, 0.1, 1)[0]);
			vector[0]	=	(var > 1.0) ? 1.0 : var;
			var = fabs(normal_random(rng, 0.35, 0.1, 1)[0]);
			vector[1]	=	(var > 1.0) ? 1.0 : var;
			break;
		}
		case 1:{
			var = fabs(normal_random(rng, 0.5, 0.1, 1)[0]);
			vector[0]	=	(var > 1.0) ? 1.0 : var;
			var = fabs(normal_random(rng, 0.65, 0.1, 1)[0]);
			vector[1]	=	(var > 1.0) ? 1.0 : var;
			break;
		}
		case 2:{
			var	=	fabs(normal_random(rng, 0.65, 0.1, 1)[0]);
			vector[0]	=	(var > 1.0) ? 1.0 : var;
			var	=	fabs(normal_random(rng, 0.35, 0.1, 1)[0]);
			vector[1]	=	(var > 1.0) ? 1.0 : var;
			it_gss = -1;
			break;
		}
	}
	it_gss++;
	return vector;
}
