
#include "pre_def.h"
#include "func_def.h"
#include "gss_som.h"
#include <pthread.h>

#ifndef __THREADS_H__
#define __THREADS_H__

#define 	NUM_THREADS	5

typedef struct evaluations
{
	double 	** 	distortion;			// distortion measurement
	double	**	quantization;				// average quantization error
	double 	** 	distortion_gss;	// distortion measurement	with gaussian filtering
	double	**	quantization_gss;		// average quantization error with gaussian filtering
} Evaluations;

typedef struct evaluations_faulty
{
	double 	*** 	distortion;			// distortion measurement
	double	***		quantization;				// average quantization error
	double 	*** 	distortion_gss;	// distortion measurement	with gaussian filtering
	double	***		quantization_gss;		// average quantization error with gaussian filtering
	double 	*** 	distortion_faulty;			// distortion measurement
	double	***		quantization_faulty;				// average quantization error
	double 	*** 	distortion_gss_faulty;	// distortion measurement	with gaussian filtering
	double	***		quantization_gss_faulty;		// average quantization error with gaussian filtering
} Evaluations_faulty;

typedef struct statistics
{
	// quantization
	double 	*	avg;							// mean
	double	*	stddev;					// standard deviation
	double 	*	avg_gss;					// mean gaussian filter
	double	*	stddev_gss;			// standard deviation gaussian filter
	double 	*	avg_faulty;							// mean
	double	*	stddev_faulty;					// standard deviation
	double 	*	avg_gss_faulty;					// mean gaussian filter
	double	*	stddev_gss_faulty;			// standard deviation gaussian filter
	// distortion
	double 	*	avgdist;							// mean
	double	*	stddevdist;					// standard deviation
	double 	*	avgdist_gss;					// mean gaussian filter
	double	*	stddevdist_gss;			// standard deviation gaussian filter
	double 	*	avgdist_faulty;							// mean
	double	*	stddevdist_faulty;					// standard deviation
	double 	*	avgdist_gss_faulty;					// mean gaussian filter
	double	*	stddevdist_gss_faulty;			// standard deviation gaussian filter

}	Statistics;

typedef	struct input_args
{
	char 				*		name;
	Kohonen 		*		map;
	int 				***	train_set;
	int 				**	valid_set;
	int 				**	test_set;
	Evaluations 				*		qlt_train;
	Evaluations 			 	*		qlt_valid;
	Evaluations_faulty 	*		qlt_test;	
	Statistics 					*		stat;
	char 				* 	status;
} Input_args;

typedef struct monitor_arg{
    char ** th1;
    char ** th2;
    char ** th3;
    char ** th4;
    char ** th5;
} Monitor;

void 		init_evaluations(Evaluations * ev);

void		init_evaluations_faulty(Evaluations_faulty * ev);

void 		init_statistics(Statistics * st);

void	*	learning_thread(void * args);

void    *   monitor_thread(void * args);

#endif //__THREADS_H__
