
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

  /*  calculate distortion for each map for each epoche
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
  
  for (i=0;i<NBMAPS;i++) mapinit[i] = init();

  int ** in = (int **) malloc_2darray(NBITEREPOCH, 2);
	int ** test = (int **) malloc_2darray(SIZE*SIZE*100, 2);

	for(i = 0; i < SIZE*SIZE*100; i++) {
  	for(k = 0; k < INS; k++) {
			test[i][k] = (int) ((1.0 * one) * normal_dataset(&rng)[k]);
    }
  }

  for (m=0;m<NBMAPS;m++) {

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
      if(j == 0){
      	printf("****************\nBefore learning\n");
	      errorrate(map[m], in, distortion[m], 0);
	      distortion_th[m] = distortion[m];
	      distortion_FI[m] = distortion[m];
	      distortion_NI[m] = distortion[m];
	      distortion_NF[m] = distortion[m];
	      errorrateGSS(map[m], in, distortion_gss[m], 0);
	      distortion_th_gss[m] = distortion_gss[m];
	      distortion_FI_gss[m] = distortion_gss[m];
	      distortion_NI_gss[m] = distortion_gss[m];
	      distortion_NF_gss[m] = distortion_gss[m];
	      
	      errorrate(map[m], test, distortion_test[m], 0);
	      distortion_th_test[m] = distortion_test[m];
	      distortion_FI_test[m] = distortion_test[m];
	      distortion_NI_test[m] = distortion_test[m];
	      distortion_NF_test[m] = distortion_test[m];
	      errorrateGSS(map[m], test, distortion_gss_test[m], 0);
	      distortion_th_gss_test[m] = distortion_gss_test[m];
	      distortion_FI_gss_test[m] = distortion_gss_test[m];
	      distortion_NI_gss_test[m] = distortion_gss_test[m];
	      distortion_NF_gss_test[m] = distortion_gss_test[m];
	      
      }
			learn(map[m], in, j);
    	printf("****************\nAfter standard learning\n");
    	errorrate(map[m], in, distortion[m], j+1);
			errorrateGSS(map[m], in, distortion_gss[m], j+1);
			errorrate(map[m], test, distortion_test[m], j+1);
			errorrateGSS(map[m], test, distortion_gss_test[m], j+1);
    	
    	learn_threshold(map_th[m], in, j);
    	printf("****************\nAfter thresholded learning\n");
    	errorrate(map_th[m], in, distortion_th[m], j+1);
			errorrateGSS(map_th[m], in, distortion_th_gss[m], j+1);
			errorrate(map_th[m], test, distortion_th_test[m], j+1);
			errorrateGSS(map_th[m], test, distortion_th_gss_test[m], j+1);
    	
    	learn_FI(map_FI[m], in, j);
    	printf("****************\nAfter fault injection learning\n");
    	errorrate(map_FI[m], in, distortion_FI[m], j+1);
			errorrateGSS(map_FI[m], in, distortion_FI_gss[m], j+1);
			errorrate(map_FI[m], test, distortion_FI_test[m], j+1);
			errorrateGSS(map_FI[m], test, distortion_FI_gss_test[m], j+1);
    	
    	learn_NI(map_NI[m], in, j);
    	printf("****************\nAfter noise injection learning\n");
    	errorrate(map_NI[m], in, distortion_NI[m], j+1);
			errorrateGSS(map_NI[m], in, distortion_NI_gss[m], j+1);
			errorrate(map_NI[m], test, distortion_NI_test[m], j+1);
			errorrateGSS(map_NI[m], test, distortion_NI_gss_test[m], j+1);
    	
    	learn_NF(map_NF[m], in, j);
    	printf("****************\nAfter NF driven learning\n");
    	errorrate(map_NF[m], in, distortion_NF[m], j+1);
			errorrateGSS(map_NF[m], in, distortion_NF_gss[m], j+1);
			errorrate(map_NF[m], test, distortion_NF_test[m], j+1);
			errorrateGSS(map_NF[m], test, distortion_NF_gss_test[m], j+1);
  	} // end loop over learn epoches
  } // end learn loop through all maps

  double *** distortion2_test 		= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_th_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_FI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_NI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
	double *** distortion2_NF_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

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

				distortion2_test[p][e][m] 	 = distortion_measure(map2, test2, 1.0);
				distortion2_th_test[p][e][m] = distortion_measure(map2_th, test2, 1.0);
				distortion2_FI_test[p][e][m] = distortion_measure(map2_FI, test2, 1.0);
				distortion2_NI_test[p][e][m] = distortion_measure(map2_NI, test2, 1.0);
				distortion2_NF_test[p][e][m] = distortion_measure(map2_NF, test2, 1.0);
		  } // end loop on map initializations
		} // end loop on experiments (faulty versions)
	}
	for (m = 0; m < NBMAPS; m++) {
		freeMap(map[m]);
		freeMap(map_th[m]);
		freeMap(map_FI[m]);
		freeMap(map_NI[m]);
		freeMap(map_NF[m]);
  }
  clock_t stop = clock();
  double elapsed = (double) (stop - start) / 1000.0;
  printf("Time elapsed in ms: %f", elapsed);
  exit(1);
}