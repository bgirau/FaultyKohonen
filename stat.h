
#include "pre_def.h"
#include "func_def.h"

#ifndef __STAT_H__
#define __STAT_H__

typedef struct avg_std_dev 
{
	float avg;
	float stddev;
	float avgdist;
	float stddevdist;
}	Avgstddev;

void avgstddev_init(Avgstddev * a);

void print_avgstddev(Avgstddev stat);
/* Accumulate faults and addeddist	*/
void acc_avgstddev(Avgstddev * stat, float faults, float addeddist);
/* statistical inference */
void calc_avgstddev(Avgstddev * stat);

typedef struct main_avg_std_dev
{
	float * mainavg;
	float * mainavgdist;
    float * mainstddev;
    float * mainstddevdist;
}	Mainavgstddev;
/* memory allocation for Mainavgstddev fields */
void Mainavgstddev_init(Mainavgstddev * mains, int size);

/* add Avgstddev to Mainavgstddev */ 
void add_avgstddev_mainavgstddev(Mainavgstddev * mains, Avgstddev stat, int pos);

/* print Mainavgstddev */ 
void print_mainavgstddev(Mainavgstddev mains, int size);

/* 2d array dynamic allocation */
int ** malloc_2darray(int height, int width);

double ** malloc_2darray_f(int height, int width);



#endif //__STAT_H__

