
#include "pre_def.h"

#ifndef __GSS_FILTER_H__
#define __GSS_FILTER_H__

Kohonen 	gauss_filter(Kohonen map, float std_dev);

double 	**	gauss_kernel(int map_size, double std_dev, int x_node, int y_node);

void 		print_kernel(double **kernel, int map_size);


#endif //__GSS_FILTER_H__