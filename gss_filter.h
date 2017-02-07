
#include "pre_def.h"

#ifndef __GSS_FILTER_H__
#define __GSS_FILTER_H__

float * gauss_kernel(int map_size, float std_dev, int node);

void 		print_kernel(float *kernel, int map_size);


#endif //__GSS_FILTER_H__