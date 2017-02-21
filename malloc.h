
#include "pre_def.h"
#include "func_def.h"

#ifndef __STAT_H__
#define __STAT_H__

/* array dynamic allocation */

double ** calloc_2darray(int dim1, int dim2);

int ** malloc_2darray(int height, int width);

double ** malloc_2darray_f(int height, int width);

int *** malloc_3darray(int dim1, int dim2, int dim3);

double ***  malloc_3darray_f(int dim1, int dim2, int dim3);

float **** malloc_4darray_f(int dim1, int dim2, int dim3, int dim4);

#endif //__STAT_H__

