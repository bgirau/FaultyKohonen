
#include "func_def.h"
#include "gss_filter.h"
#include "gss_som.h"
#include "stat.h"
#include "time.h"
#include "custom_rand.h"

int main(){
  
  pcg32_random_t rng;
  init_random(&rng);

	srand(time(NULL));
	clock_t 	start = clock();
	int  p,i,j,e,k,m;

  Kohonen *map      = malloc(NBMAPS*sizeof(Kohonen));
  Kohonen *map_th   = malloc(NBMAPS*sizeof(Kohonen));
  Kohonen *map_FI   = malloc(NBMAPS*sizeof(Kohonen));
  Kohonen *map_NI   = malloc(NBMAPS*sizeof(Kohonen));
  Kohonen *map_NF   = malloc(NBMAPS*sizeof(Kohonen));
  Kohonen *mapinit  = malloc(NBMAPS*sizeof(Kohonen));

  /*  
      calculate distortion for each map for each epoche
      NBEPOCHLEARN+1 - # of distortions measure + measure before learning
  */
  double  ** distortion     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_th  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_FI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NF  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** distortion_gss     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_th_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_FI_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NI_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NF_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** distortion_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_th_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_FI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NF_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** distortion_gss_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_th_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_FI_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NI_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** distortion_NF_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  
  for (i = 0; i < NBMAPS; i++) mapinit[i] = init();

  int ** in = (int **) malloc_2darray(NBITEREPOCH, 2);
	int ** test = (int **) malloc_2darray(SIZE*SIZE*100, 2);

	for(i = 0; i < SIZE*SIZE*100; i++) {
  	for(k = 0; k < INS; k++) {
			test[i][k] = (int) ((1.0 * one) * normal_dataset(&rng)[k]);
    }
  }

  for (m = 0; m < NBMAPS; m++) {

  	printf("\n**************\
  		******************\n learning map number %d : \n\n",m);
  	map[m]		=	copy(mapinit[m]);
  	map_th[m]	=	copy(mapinit[m]);
  	map_FI[m]	=	copy(mapinit[m]);
  	map_NI[m]	=	copy(mapinit[m]);
  	map_NF[m]	=	copy(mapinit[m]);

    for(j = 0; j < NBEPOCHLEARN; j++){
      // generate new random values
      for(i = 0; i < NBITEREPOCH; i++) {
        for(k = 0; k < INS; k++) {
         in[i][k] = (int) ((1.0 * one) * normal_dataset(&rng)[k]);
        }
      }
      if (j == 0) {
      	printf("****************\nBefore learning\n");
        distortion[m][0] = errorrate(map[m], in, 0) / (one*1.0);
	      distortion_th[m][0] = distortion[m][0];
	      distortion_FI[m][0] = distortion[m][0];
	      distortion_NI[m][0] = distortion[m][0];
	      distortion_NF[m][0] = distortion[m][0];
	      distortion_gss[m][0] =  errorrateGSS(map[m], in, 0) / (one*1.0);
        distortion_th_gss[m][0] = distortion_gss[m][0];
        distortion_FI_gss[m][0] = distortion_gss[m][0];
        distortion_NI_gss[m][0] = distortion_gss[m][0];
        distortion_NF_gss[m][0] = distortion_gss[m][0];

        distortion_test[m][0] = errorrate(map[m], test, 0) / (one*1.0);
	      distortion_th_test[m][0] = distortion_test[m][0];
	      distortion_FI_test[m][0] = distortion_test[m][0];
	      distortion_NI_test[m][0] = distortion_test[m][0];
	      distortion_NF_test[m][0] = distortion_test[m][0];
        distortion_gss_test[m][0] = errorrateGSS(map[m], test, 0) / (one*1.0);
        distortion_th_gss_test[m][0] = distortion_gss_test[m][0];
        distortion_FI_gss_test[m][0] = distortion_gss_test[m][0];
        distortion_NI_gss_test[m][0] = distortion_gss_test[m][0];
        distortion_NF_gss_test[m][0] = distortion_gss_test[m][0];	      
      }
			learn(map[m], in, j);
    	printf("****************\nAfter standard learning\n");
      distortion[m][j+1] = errorrate(map[m], in,  j+1)/(one*1.0);
      distortion_gss[m][j+1] = errorrateGSS(map[m], in,  j+1)/(one*1.0);
      distortion_test[m][j+1] = errorrate(map[m], test, j+1)/(one*1.0);
      distortion_gss_test[m][j+1] = errorrateGSS(map[m], in,  j+1)/(one*1.0);
    	
    	learn_threshold(map_th[m], in, j);
    	printf("****************\nAfter thresholded learning\n");
      distortion_th[m][j+1] = errorrate(map_th[m], in, j+1)/(one*1.0);
      distortion_th_gss[m][j+1] = errorrateGSS(map_th[m], in, j+1)/(one*1.0);
      distortion_th_test[m][j+1] = errorrate(map_th[m], test, j+1)/(one*1.0);
      distortion_th_gss_test[m][j+1] = errorrateGSS(map_th[m], test, j+1)/(one*1.0);
    	
    	learn_FI(map_FI[m], in, j);
    	printf("****************\nAfter fault injection learning\n");
      distortion_FI[m][j+1] = errorrate(map_FI[m], in, j+1)/(one*1.0);
      distortion_FI_gss[m][j+1] = errorrateGSS(map_FI[m], in, j+1)/(one*1.0);
      distortion_FI_test[m][j+1] = errorrate(map_FI[m], test, j+1)/(one*1.0);
      distortion_FI_gss_test[m][j+1] = errorrateGSS(map_FI[m], test, j+1)/(one*1.0);
    	
    	learn_NI(map_NI[m], in, j);
    	printf("****************\nAfter noise injection learning\n");
      distortion_NI[m][j+1] = errorrate(map_NI[m], in, j+1)/(one*1.0);
      distortion_NI_gss[m][j+1] = errorrateGSS(map_NI[m], in, j+1)/(one*1.0);
      distortion_NI_test[m][j+1] = errorrate(map_NI[m], test, j+1)/(one*1.0);
      distortion_NI_gss_test[m][j+1] = errorrateGSS(map_NI[m], test, j+1)/(one*1.0);
    	
    	learn_NF(map_NF[m], in, j);
    	printf("****************\nAfter NF driven learning\n");
      distortion_NF[m][j+1] = errorrate(map_NF[m], in, j+1)/(one*1.0);
      distortion_NF_gss[m][j+1] = errorrateGSS(map_NF[m], in, j+1)/(one*1.0);
      distortion_NF_test[m][j+1] = errorrate(map_NF[m], test, j+1)/(one*1.0);
      distortion_NF_gss_test[m][j+1] = errorrateGSS(map_NF[m], test, j+1)/(one*1.0);
  	} // end loop over learn epoches
  } // end learn loop through all maps

  double *** distortion2_test 		= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_th_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_FI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_NI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_NF_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** distortion2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_th_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_FI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NF_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** distortion2_test_gss     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_th_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_FI_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NI_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NF_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** distortion2_test_faulty_gss     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_th_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_FI_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NI_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NF_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double * avg = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NF = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NF = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avg_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NF_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NF_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avg_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_th_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_th_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_FI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_FI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NF_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NF_gss = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avg_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_th_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_th_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_FI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_FI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NF_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NF_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));

  double tt = nb_experiments * NBMAPS;

	int ** test2 = (int **) malloc_2darray(SIZE*SIZE*1000, 2);

	for(i = 0; i < SIZE*SIZE*1000; i++) {
  	for(k = 0; k < INS; k++) {
			test2[i][k] = (int) ((1.0 * one) * normal_dataset(&rng)[k]);
    }
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
  	for (e = 0; e < nb_experiments; e++) {
			for (m = 0; m < NBMAPS; m++) {
				Kohonen map2    = copy(map[m]);
  			Kohonen map2_th = copy(map_th[m]);
				Kohonen map2_FI = copy(map_FI[m]);
  			Kohonen map2_NI = copy(map_NI[m]);
  			Kohonen map2_NF = copy(map_NF[m]);
				// introduction of faults in the copies of the pre-learned maps

				faulty_weights(map2, p);
				faulty_weights(map2_th, p);
				faulty_weights(map2_FI, p);
				faulty_weights(map2_NI, p);
				faulty_weights(map2_NF, p);

				distortion2_test[p][e][m] 	 = distortion_measure(map[m], test2, 1.0) / (1.0 * one);
				distortion2_th_test[p][e][m] = distortion_measure(map_th[m], test2, 1.0) / (1.0 * one);
				distortion2_FI_test[p][e][m] = distortion_measure(map_FI[m], test2, 1.0) / (1.0 * one);
				distortion2_NI_test[p][e][m] = distortion_measure(map_NI[m], test2, 1.0) / (1.0 * one);
				distortion2_NF_test[p][e][m] = distortion_measure(map_NF[m], test2, 1.0) / (1.0 * one);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2, test2, 1.0) / (1.0 * one);
        distortion2_th_test_faulty[p][e][m] = distortion_measure(map2_th, test2, 1.0) / (1.0 * one);
        distortion2_FI_test_faulty[p][e][m] = distortion_measure(map2_FI, test2, 1.0) / (1.0 * one);
        distortion2_NI_test_faulty[p][e][m] = distortion_measure(map2_NI, test2, 1.0) / (1.0 * one);
        distortion2_NF_test_faulty[p][e][m] = distortion_measure(map2_NF, test2, 1.0) / (1.0 * one);

        distortion2_test_gss[p][e][m]    = distortion_measure_GSS(map[m], test2, 1.0) / (1.0 * one);
        distortion2_th_test_gss[p][e][m] = distortion_measure_GSS(map_th[m], test2, 1.0) / (1.0 * one);
        distortion2_FI_test_gss[p][e][m] = distortion_measure_GSS(map_FI[m], test2, 1.0) / (1.0 * one);
        distortion2_NI_test_gss[p][e][m] = distortion_measure_GSS(map_NI[m], test2, 1.0) / (1.0 * one);
        distortion2_NF_test_gss[p][e][m] = distortion_measure_GSS(map_NF[m], test2, 1.0) / (1.0 * one);

        distortion2_test_faulty_gss[p][e][m]    = distortion_measure_GSS(map2, test2, 1.0) / (1.0 * one);
        distortion2_th_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_th, test2, 1.0) / (1.0 * one);
        distortion2_FI_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_FI, test2, 1.0) / (1.0 * one);
        distortion2_NI_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_NI, test2, 1.0) / (1.0 * one);
        distortion2_NF_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_NF, test2, 1.0) / (1.0 * one);

        avg[p] += distortion2_test[p][e][m];
        avg_th[p] += distortion2_th_test[p][e][m];
        avg_FI[p] += distortion2_FI_test[p][e][m];
        avg_NI[p] += distortion2_NI_test[p][e][m];
        avg_NF[p] += distortion2_NF_test[p][e][m];

        avg_faulty[p] += distortion2_test_faulty[p][e][m];
        avg_th_faulty[p] += distortion2_th_test_faulty[p][e][m];
        avg_FI_faulty[p] += distortion2_FI_test_faulty[p][e][m];
        avg_NI_faulty[p] += distortion2_NI_test_faulty[p][e][m];
        avg_NF_faulty[p] += distortion2_NF_test_faulty[p][e][m];

        avg_gss[p] += distortion2_test_gss[p][e][m];
        avg_th_gss[p] += distortion2_th_test_gss[p][e][m];
        avg_FI_gss[p] += distortion2_FI_test_gss[p][e][m];
        avg_NI_gss[p] += distortion2_NI_test_gss[p][e][m];
        avg_NF_gss[p] += distortion2_NF_test_gss[p][e][m];

        avg_faulty_gss[p] += distortion2_test_faulty_gss[p][e][m];
        avg_th_faulty_gss[p] += distortion2_th_test_faulty_gss[p][e][m];
        avg_FI_faulty_gss[p] += distortion2_FI_test_faulty_gss[p][e][m];
        avg_NI_faulty_gss[p] += distortion2_NI_test_faulty_gss[p][e][m];
        avg_NF_faulty_gss[p] += distortion2_NF_test_faulty_gss[p][e][m];

		  } // end loop on map initializations
		} // end loop on experiments (faulty versions)
    avg[p] /= tt;
    avg_th[p] /= tt;
    avg_FI[p] /= tt;
    avg_NI[p] /= tt;
    avg_NF[p] /= tt;
    avg_faulty[p] /= tt;
    avg_th_faulty[p]  /= tt;
    avg_FI_faulty[p]  /= tt;
    avg_NI_faulty[p]  /= tt;
    avg_NF_faulty[p]  /= tt;
    avg_gss[p]  /= tt;
    avg_th_gss[p] /= tt;
    avg_FI_gss[p] /= tt;
    avg_NI_gss[p] /= tt;
    avg_NF_gss[p] /= tt;
    avg_faulty_gss[p] /= tt;
    avg_th_faulty_gss[p]  /= tt;
    avg_FI_faulty_gss[p]  /= tt;
    avg_NI_faulty_gss[p]  /= tt;
    avg_NF_faulty_gss[p]  /= tt;
    
	}

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
        stddev[p] += (distortion2_test[p][e][m] - avg[p]) * (distortion2_test[p][e][m] - avg[p]);
        stddev_th[p] += (distortion2_th_test[p][e][m] - avg_th[p]) * (distortion2_th_test[p][e][m] - avg_th[p]);
        stddev_FI[p] += (distortion2_FI_test[p][e][m] - avg_FI[p]) * (distortion2_FI_test[p][e][m] - avg_FI[p]);
        stddev_NI[p] += (distortion2_NI_test[p][e][m] - avg_NI[p]) * (distortion2_NI_test[p][e][m] - avg_NI[p]);
        stddev_NF[p] += (distortion2_NF_test[p][e][m] - avg_NF[p]) * (distortion2_NF_test[p][e][m] - avg_NF[p]);

        stddev_faulty[p] += (distortion2_test_faulty[p][e][m] - avg_faulty[p]) * (distortion2_test_faulty[p][e][m] - avg_faulty[p]);
        stddev_th_faulty[p] += (distortion2_th_test_faulty[p][e][m] - avg_th_faulty[p]) * (distortion2_th_test_faulty[p][e][m] - avg_th_faulty[p]);
        stddev_FI_faulty[p] += (distortion2_FI_test_faulty[p][e][m] - avg_FI_faulty[p]) * (distortion2_FI_test_faulty[p][e][m] - avg_FI_faulty[p]);
        stddev_NI_faulty[p] += (distortion2_NI_test_faulty[p][e][m] - avg_NI_faulty[p]) * (distortion2_NI_test_faulty[p][e][m] - avg_NI_faulty[p]);
        stddev_NF_faulty[p] += (distortion2_NF_test_faulty[p][e][m] - avg_NF_faulty[p]) * (distortion2_NF_test_faulty[p][e][m] - avg_NF_faulty[p]);

        stddev_gss[p] += (distortion2_test_gss[p][e][m] - avg_gss[p]) * (distortion2_test_gss[p][e][m] - avg_gss[p]);
        stddev_th_gss[p] += (distortion2_th_test_gss[p][e][m] - avg_th_gss[p]) * (distortion2_th_test_gss[p][e][m] - avg_th_gss[p]);
        stddev_FI_gss[p] += (distortion2_FI_test_gss[p][e][m] - avg_FI_gss[p]) * (distortion2_FI_test_gss[p][e][m] - avg_FI_gss[p]);
        stddev_NI_gss[p] += (distortion2_NI_test_gss[p][e][m] - avg_NI_gss[p]) * (distortion2_NI_test_gss[p][e][m] - avg_NI_gss[p]);
        stddev_NF_gss[p] += (distortion2_NF_test_gss[p][e][m] - avg_NF_gss[p]) * (distortion2_NF_test_gss[p][e][m] - avg_NF_gss[p]);

        stddev_faulty_gss[p] += (distortion2_test_faulty_gss[p][e][m] - avg_faulty_gss[p]) * (distortion2_test_faulty_gss[p][e][m] - avg_faulty_gss[p]);
        stddev_th_faulty_gss[p] += (distortion2_th_test_faulty_gss[p][e][m] - avg_th_faulty_gss[p]) * (distortion2_th_test_faulty_gss[p][e][m] - avg_th_faulty_gss[p]);
        stddev_FI_faulty_gss[p] += (distortion2_FI_test_faulty_gss[p][e][m] - avg_FI_faulty_gss[p]) * (distortion2_FI_test_faulty_gss[p][e][m] - avg_FI_faulty_gss[p]);
        stddev_NI_faulty_gss[p] += (distortion2_NI_test_faulty_gss[p][e][m] - avg_NI_faulty_gss[p]) * (distortion2_NI_test_faulty_gss[p][e][m] - avg_NI_faulty_gss[p]);
        stddev_NF_faulty_gss[p] += (distortion2_NF_test_faulty_gss[p][e][m] - avg_NF_faulty_gss[p]) * (distortion2_NF_test_faulty_gss[p][e][m] - avg_NF_faulty_gss[p]);
      }
    }
    stddev[p] = mysqrt(stddev[p]/(tt-1));
    stddev_th[p] = mysqrt(stddev_th[p]/(tt-1));
    stddev_FI[p] = mysqrt(stddev_FI[p]/(tt-1));
    stddev_NI[p] = mysqrt(stddev_NI[p]/(tt-1)); 
    stddev_NF[p] = mysqrt(stddev_NF[p]/(tt-1));

    stddev_faulty[p] = mysqrt(stddev_faulty[p]/(tt-1)); 
    stddev_th_faulty[p] = mysqrt(stddev_th_faulty[p]/(tt-1));
    stddev_FI_faulty[p] = mysqrt(stddev_FI_faulty[p]/(tt-1));
    stddev_NI_faulty[p] = mysqrt(stddev_NI_faulty[p]/(tt-1));
    stddev_NF_faulty[p] = mysqrt(stddev_NF_faulty[p]/(tt-1));

    stddev_gss[p] = mysqrt(stddev_gss[p]/(tt-1));
    stddev_th_gss[p] = mysqrt(stddev_th_gss[p]/(tt-1));
    stddev_FI_gss[p] = mysqrt(stddev_FI_gss[p]/(tt-1));
    stddev_NI_gss[p] = mysqrt(stddev_NI_gss[p]/(tt-1)); 
    stddev_NF_gss[p] = mysqrt(stddev_NF_gss[p]/(tt-1));

    stddev_faulty_gss[p] = mysqrt(stddev_faulty_gss[p]/(tt-1)); 
    stddev_th_faulty_gss[p] = mysqrt(stddev_th_faulty_gss[p]/(tt-1));
    stddev_FI_faulty_gss[p] = mysqrt(stddev_FI_faulty_gss[p]/(tt-1));
    stddev_NI_faulty_gss[p] = mysqrt(stddev_NI_faulty_gss[p]/(tt-1));
    stddev_NF_faulty_gss[p] = mysqrt(stddev_NF_faulty_gss[p]/(tt-1));
  }

	FILE 	*	fp;

	// fp = fopen ("learn.json", "w+");
  // fprintf(fp, "{\n\t\"This file contains distortion measurements. Each measurement is calculated for each learning epoch on a LEARN set\": {");
 //  fprintf(fp, "{\n\t\t\"Map number\": {");
	// for(m = 0; m < NBMAPS; m++){
	// 	fprintf(fp, "\n\t\t\t\"%d\": { \n", m+1);
	// 	// fprintf(fp, "Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
	// 	for(j = 0; j < NBEPOCHLEARN; j++){
	// 		fprintf(fp, "\t\t\t\t\"%d\": [%-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f],\n", j,
 //        distortion[m][j], distortion_th[m][j], distortion_FI[m][j], distortion_NI[m][j], distortion_NF[m][j],
 //        distortion_gss[m][j], distortion_th_gss[m][j], distortion_FI_gss[m][j], 
 //        distortion_NI_gss[m][j], distortion_NF_gss[m][j]);
	// 	}
 //    fprintf(fp, "\t\t\t\t\"%d\": [%-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f]\n\t\t\t}", j,
 //        distortion[m][j], distortion_th[m][j], distortion_FI[m][j], distortion_NI[m][j], distortion_NF[m][j],
 //        distortion_gss[m][j], distortion_th_gss[m][j], distortion_FI_gss[m][j], 
 //        distortion_NI_gss[m][j], distortion_NF_gss[m][j]);
	// }
 //  fprintf(fp, "\n\t\t}\n\t}");
  fp = fopen ("learn.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", m, j,
        distortion[m][j], distortion_th[m][j], distortion_FI[m][j], distortion_NI[m][j], distortion_NF[m][j],
        distortion_gss[m][j], distortion_th_gss[m][j], distortion_FI_gss[m][j], 
        distortion_NI_gss[m][j], distortion_NF_gss[m][j]);
    }
  }
  fclose(fp);

  // fp = fopen ("test_during_learn.json", "w+");
  // fprintf(fp, "{\n\t\"This file contains distortion measurements. Each measurement is calculated for each learning epoch on a TEST set\": {");
  // fprintf(fp, "{\n\t\t\"Map number\": {");
  // for(m = 0; m < NBMAPS; m++){
  //   fprintf(fp, "\n\t\t\t\"%d\": { \n", m+1);
  //   for(j = 0; j < NBEPOCHLEARN; j++){
  //     fprintf(fp, "\t\t\t\t\"%d\": [%-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f],\n", j,
  //       distortion_test[m][j], distortion_th_test[m][j], distortion_FI_test[m][j], distortion_NI_test[m][j],
  //       distortion_NF_test[m][j], distortion_gss_test[m][j], distortion_th_gss_test[m][j],
  //       distortion_FI_gss_test[m][j], distortion_NI_gss_test[m][j], distortion_NF_gss_test[m][j]);
  //   }
  //   fprintf(fp, "\t\t\t\t\"%d\": [%-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f]\n\t\t\t}", j,
  //       distortion_test[m][j], distortion_th_test[m][j], distortion_FI_test[m][j], distortion_NI_test[m][j],
  //       distortion_NF_test[m][j], distortion_gss_test[m][j], distortion_th_gss_test[m][j],
  //       distortion_FI_gss_test[m][j], distortion_NI_gss_test[m][j], distortion_NF_gss_test[m][j]);
  // }
  // fprintf(fp, "\n\t\t}\n\t}");
  fp = fopen ("test_during_learn.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", m, j,
        distortion_test[m][j], distortion_th_test[m][j], distortion_FI_test[m][j], distortion_NI_test[m][j],
        distortion_NF_test[m][j], distortion_gss_test[m][j], distortion_th_gss_test[m][j],
        distortion_FI_gss_test[m][j], distortion_NI_gss_test[m][j], distortion_NF_gss_test[m][j]);
    }
  }
  fclose(fp);

  // fp = fopen ("test.json", "w+");
  // fprintf(fp, "{\n\t\"This file contains distortion measurements. Each measurement is calculated on a TEST set\": {");
  // fprintf(fp, "{\n\t\t\"Percentage faults\": {");
  // for (p = 0; p < MAXFAULTPERCENT; p++) {
  //   fprintf(fp, "\n\t\t\t\"%d\": { \n", p);
  //   fprintf(fp, "\t\t\t\t\"Experiment #\": {");
  //   for (e = 0; e < nb_experiments; e++) {
  //     fprintf(fp, "\n\t\t\t\t\t\"%d\": { \n", e+1);
  //     fprintf(fp, "\t\t\t\t\t\t\"Map number\": {\n");
  //     for (m = 0; m < NBMAPS-1; m++) {
  //       fprintf(fp, "\t\t\t\t\t\t\t\"%d\": [%-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f],\n", m+1,
  //         distortion2_test[p][e][m], distortion2_th_test[p][e][m], distortion2_FI_test[p][e][m], distortion2_NI_test[p][e][m], 
  //         distortion2_NF_test[p][e][m],
  //         distortion2_test_faulty[p][e][m], distortion2_th_test_faulty[p][e][m], distortion2_FI_test_faulty[p][e][m], 
  //         distortion2_NI_test_faulty[p][e][m], distortion2_NF_test_faulty[p][e][m],
  //         distortion2_test_gss[p][e][m], distortion2_th_test_gss[p][e][m], distortion2_FI_test_gss[p][e][m], 
  //         distortion2_NI_test_gss[p][e][m], distortion2_NF_test_gss[p][e][m],
  //         distortion2_test_faulty_gss[p][e][m], distortion2_th_test_faulty_gss[p][e][m], 
  //         distortion2_FI_test_faulty_gss[p][e][m], distortion2_NI_test_faulty_gss[p][e][m], distortion2_NF_test_faulty_gss[p][e][m]);
  //     }
  //     fprintf(fp, "\t\t\t\t\t\t\t\"%d\": [%-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f, %-f]\n\t\t\t\t\t\t}", m+1,
  //         distortion2_test[p][e][m], distortion2_th_test[p][e][m], distortion2_FI_test[p][e][m], distortion2_NI_test[p][e][m], 
  //         distortion2_NF_test[p][e][m],
  //         distortion2_test_faulty[p][e][m], distortion2_th_test_faulty[p][e][m], distortion2_FI_test_faulty[p][e][m], 
  //         distortion2_NI_test_faulty[p][e][m], distortion2_NF_test_faulty[p][e][m],
  //         distortion2_test_gss[p][e][m], distortion2_th_test_gss[p][e][m], distortion2_FI_test_gss[p][e][m], 
  //         distortion2_NI_test_gss[p][e][m], distortion2_NF_test_gss[p][e][m],
  //         distortion2_test_faulty_gss[p][e][m], distortion2_th_test_faulty_gss[p][e][m], 
  //         distortion2_FI_test_faulty_gss[p][e][m], distortion2_NI_test_faulty_gss[p][e][m], distortion2_NF_test_faulty_gss[p][e][m]);
  //     fprintf(fp, "\n\t\t\t\t\t}");
  //   }
  //   fprintf(fp, "\n\t\t\t\t}\n\t\t\t}");
  // }
  // fprintf(fp, "\n\t\t}\n\t}");
  fp = fopen ("test.txt", "w+");
  fprintf(fp, "Percentage_faults;Experiment_number;Map_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p, e+1, m+1,
          distortion2_test[p][e][m], distortion2_th_test[p][e][m], distortion2_FI_test[p][e][m], distortion2_NI_test[p][e][m], 
          distortion2_NF_test[p][e][m],
          distortion2_test_faulty[p][e][m], distortion2_th_test_faulty[p][e][m], distortion2_FI_test_faulty[p][e][m], 
          distortion2_NI_test_faulty[p][e][m], distortion2_NF_test_faulty[p][e][m],
          distortion2_test_gss[p][e][m], distortion2_th_test_gss[p][e][m], distortion2_FI_test_gss[p][e][m], 
          distortion2_NI_test_gss[p][e][m], distortion2_NF_test_gss[p][e][m],
          distortion2_test_faulty_gss[p][e][m], distortion2_th_test_faulty_gss[p][e][m], 
          distortion2_FI_test_faulty_gss[p][e][m], distortion2_NI_test_faulty_gss[p][e][m], distortion2_NF_test_faulty_gss[p][e][m]);
       }
    }
  }
  fclose(fp);

  // fp = fopen ("statistics.json", "w+");
  // fprintf(fp, "{\n\t\"This file contains average values and standard deviations of distortion measurements \": {");
  // fprintf(fp, "{\n\t\t\"Percentage faults\": {");
  // for (p = 0; p < MAXFAULTPERCENT; p++) {
  //   fprintf(fp, "\n\t\t\t\"%d\": {", p);
  //   fprintf(fp, "\n\t\t\t\t\"Standard\": [[%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f]]", 
  //     avg[p], stddev[p], avg_th[p], stddev_th[p], avg_FI[p], stddev_FI[p], avg_NI[p], stddev_NI[p], avg_NF[p], stddev_NF[p]);
  //   fprintf(fp, "\n\t\t\t\t\"Standard_faulty\": [[%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f]]", 
  //     avg_faulty[p], stddev_faulty[p], avg_th_faulty[p], stddev_th_faulty[p], avg_FI_faulty[p], stddev_FI_faulty[p], 
  //     avg_NI_faulty[p], stddev_NI_faulty[p], avg_NF_faulty[p], stddev_NF_faulty[p]);
  //   fprintf(fp, "\n\t\t\t\t\"GSS\": [[%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f]]", 
  //     avg_gss[p], stddev_gss[p], avg_th_gss[p], stddev_th_gss[p], avg_FI_gss[p], stddev_FI_gss[p], 
  //     avg_NI_gss[p], stddev_NI_gss[p], avg_NF_gss[p], stddev_NF_gss[p]);
  //   fprintf(fp, "\n\t\t\t\t\"GSS_faulty\": [[%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f], [%-f, %-f]]\n\t\t\t}", 
  //     avg_faulty_gss[p], stddev_faulty_gss[p], avg_th_faulty_gss[p], stddev_th_faulty_gss[p], avg_FI_faulty_gss[p], stddev_FI_faulty_gss[p], 
  //     avg_NI_faulty_gss[p], stddev_NI_faulty_gss[p], avg_NF_faulty_gss[p], stddev_NF_faulty_gss[p]);
  //   fprintf(fp, "\t\t\t");
  // }
  // fprintf(fp, "\n\t\t}\n\t}");

  fp = fopen ("statistics.txt", "w+");
  fprintf(fp, "Percentage_faults;Standard_avg;Standard_std;Threshold_avg;Threshold_std;FaultInjection_avg;FaultInjection_std;\
    NoiseInjetion_avg;NoiseInjetion_std;NeuralField_avg;NeuralField_std;Standard_GSS_avg;Standard_GSS_std;Threshold_GSS_avg;Threshold_GSS_std;\
    FaultInjection_GSS_avg;FaultInjection_GSS_std;NoiseInjetion_GSS_avg;NoiseInjetion_GSS_std;NeuralField_GSS_avg;NeuralField_GSS_std\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p, 
      avg_faulty[p], stddev_faulty[p], avg_th_faulty[p], stddev_th_faulty[p], avg_FI_faulty[p], stddev_FI_faulty[p], 
      avg_NI_faulty[p], stddev_NI_faulty[p], avg_NF_faulty[p], stddev_NF_faulty[p],
      avg_faulty_gss[p], stddev_faulty_gss[p], avg_th_faulty_gss[p], stddev_th_faulty_gss[p], avg_FI_faulty_gss[p], stddev_FI_faulty_gss[p], 
      avg_NI_faulty_gss[p], stddev_NI_faulty_gss[p], avg_NF_faulty_gss[p], stddev_NF_faulty_gss[p]);
  }
  fclose(fp);

	for (m = 0; m < NBMAPS; m++) {
		freeMap(map[m]);
		freeMap(map_th[m]);
		freeMap(map_FI[m]); 
		freeMap(map_NI[m]);
		freeMap(map_NF[m]);
  }

  clock_t stop = clock();
  double elapsed = (double) (stop - start) / 1000.0;
  printf("Time elapsed in ms: %f\n", elapsed);
  exit(1);
}