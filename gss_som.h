/*
 *	File: gss_som.h
 *	Contains description of class assign with gaussian filtering and 
 *  applying of gaussian filtering during recall phase
*/

#include "pre_def.h"
#include "func_def.h"
#include "gss_filter.h"

#ifndef __GSS_SOM_H__
#define __GSS_SOM_H__

float gauss_distance(Kohonen map, int *B, int n, float std_dev, int x_node, int y_node);

void gaussian_prototypes(Kohonen map,float std_dev);

Winner recallGSS(Kohonen map, int *input);

double distortion_measure_GSS(Kohonen map, int** inputs, int inp,double sig);

double protodistortion_measure_GSS(Kohonen map, int** inputs, int inp,double sig);

double errorrateGSS(Kohonen map, int ** inputs,int inp, int epoch);

double avg_quant_error_GSS(Kohonen map, int ** inputs,int inp);

double evaldistortionGSS(Kohonen map, int ** inputs, int inp,int epoch);

double evalprotodistortionGSS(Kohonen map, int ** inputs, int inp,int epoch);

#endif //__GSS_SOM_H__
