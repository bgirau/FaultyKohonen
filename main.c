
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
  double  ** quantization     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_th  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_FI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NF  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** quantization_gss     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_th_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_FI_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NI_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NF_gss  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** quantization_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_th_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_FI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NF_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** quantization_gss_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_th_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_FI_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NI_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  double  ** quantization_NF_gss_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  
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
  int ** test = (int **) malloc_2darray(SIZE*SIZE*TEST_DENSITY, 2);
  
  for(i = 0; i < SIZE*SIZE*TEST_DENSITY; i++) {
    for(k = 0; k < INS; k++) {
      test[i][k] = (int) ((1.0 * one) * uniform_dataset(&rng)[k]);
    }
  }

  /*
  for(i = 0; i < SIZE*SIZE*100; i++) {
    printf("%g %g\n",test[i][0]/(1.0*one),test[i][1]/(1.0*one));
  }
  */
  
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
	  in[i][k] = (int) ((1.0 * one) * uniform_dataset(&rng)[k]);
        }
      }
      if (j == 0) {
      	printf("****************\nBefore learning\n");
        quantization[m][0] = errorrate(map[m], in,NBITEREPOCH, 0);
	quantization_th[m][0] = quantization[m][0];
	quantization_FI[m][0] = quantization[m][0];
	quantization_NI[m][0] = quantization[m][0];
	quantization_NF[m][0] = quantization[m][0];
	quantization_gss[m][0] =  errorrateGSS(map[m], in,NBITEREPOCH, 0);
        quantization_th_gss[m][0] = quantization_gss[m][0];
        quantization_FI_gss[m][0] = quantization_gss[m][0];
        quantization_NI_gss[m][0] = quantization_gss[m][0];
        quantization_NF_gss[m][0] = quantization_gss[m][0];

        quantization_test[m][0] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
	quantization_th_test[m][0] = quantization_test[m][0];
	quantization_FI_test[m][0] = quantization_test[m][0];
	quantization_NI_test[m][0] = quantization_test[m][0];
	quantization_NF_test[m][0] = quantization_test[m][0];
        quantization_gss_test[m][0] = errorrateGSS(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
        quantization_th_gss_test[m][0] = quantization_gss_test[m][0];
        quantization_FI_gss_test[m][0] = quantization_gss_test[m][0];
        quantization_NI_gss_test[m][0] = quantization_gss_test[m][0];
        quantization_NF_gss_test[m][0] = quantization_gss_test[m][0];
	
        distortion[m][0] = evaldistortion(map[m], in,NBITEREPOCH, 0);
	distortion_th[m][0] = distortion[m][0];
	distortion_FI[m][0] = distortion[m][0];
	distortion_NI[m][0] = distortion[m][0];
	distortion_NF[m][0] = distortion[m][0];
	distortion_gss[m][0] =  evaldistortionGSS(map[m], in,NBITEREPOCH, 0);
        distortion_th_gss[m][0] = distortion_gss[m][0];
        distortion_FI_gss[m][0] = distortion_gss[m][0];
        distortion_NI_gss[m][0] = distortion_gss[m][0];
        distortion_NF_gss[m][0] = distortion_gss[m][0];

        distortion_test[m][0] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
	distortion_th_test[m][0] = distortion_test[m][0];
	distortion_FI_test[m][0] = distortion_test[m][0];
	distortion_NI_test[m][0] = distortion_test[m][0];
	distortion_NF_test[m][0] = distortion_test[m][0];
        distortion_gss_test[m][0] = evaldistortionGSS(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
        distortion_th_gss_test[m][0] = distortion_gss_test[m][0];
        distortion_FI_gss_test[m][0] = distortion_gss_test[m][0];
        distortion_NI_gss_test[m][0] = distortion_gss_test[m][0];
        distortion_NF_gss_test[m][0] = distortion_gss_test[m][0];	      
      }
      learn(map[m], in, j);
      printf("****************\nAfter standard learning\n");
      quantization[m][j+1] = errorrate(map[m], in,NBITEREPOCH,  j+1);
      quantization_gss[m][j+1] = errorrateGSS(map[m], in,NBITEREPOCH,  j+1);
      quantization_test[m][j+1] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      quantization_gss_test[m][j+1] = errorrateGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
      distortion[m][j+1] = evaldistortion(map[m], in,NBITEREPOCH,  j+1);
      distortion_gss[m][j+1] = evaldistortionGSS(map[m], in,NBITEREPOCH,  j+1);
      distortion_test[m][j+1] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      distortion_gss_test[m][j+1] = evaldistortionGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
    	
      learn_threshold(map_th[m], in, j);
      printf("****************\nAfter thresholded learning\n");
      quantization_th[m][j+1] = errorrate(map[m], in,NBITEREPOCH,  j+1);
      quantization_th_gss[m][j+1] = errorrateGSS(map[m], in,NBITEREPOCH,  j+1);
      quantization_th_test[m][j+1] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      quantization_th_gss_test[m][j+1] = errorrateGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
      distortion_th[m][j+1] = evaldistortion(map[m], in,NBITEREPOCH,  j+1);
      distortion_th_gss[m][j+1] = evaldistortionGSS(map[m], in,NBITEREPOCH,  j+1);
      distortion_th_test[m][j+1] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      distortion_th_gss_test[m][j+1] = evaldistortionGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
    	
      learn_FI(map_FI[m], in, j);
      printf("****************\nAfter fault injection learning\n");
      quantization_FI[m][j+1] = errorrate(map[m], in,NBITEREPOCH,  j+1);
      quantization_FI_gss[m][j+1] = errorrateGSS(map[m], in,NBITEREPOCH,  j+1);
      quantization_FI_test[m][j+1] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      quantization_FI_gss_test[m][j+1] = errorrateGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
      distortion_FI[m][j+1] = evaldistortion(map[m], in,NBITEREPOCH,  j+1);
      distortion_FI_gss[m][j+1] = evaldistortionGSS(map[m], in,NBITEREPOCH,  j+1);
      distortion_FI_test[m][j+1] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      distortion_FI_gss_test[m][j+1] = evaldistortionGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);

      learn_NI(map_NI[m], in, j);
      printf("****************\nAfter noise injection learning\n");
      quantization_NI[m][j+1] = errorrate(map[m], in,NBITEREPOCH,  j+1);
      quantization_NI_gss[m][j+1] = errorrateGSS(map[m], in,NBITEREPOCH,  j+1);
      quantization_NI_test[m][j+1] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      quantization_NI_gss_test[m][j+1] = errorrateGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
      distortion_NI[m][j+1] = evaldistortion(map[m], in,NBITEREPOCH,  j+1);
      distortion_NI_gss[m][j+1] = evaldistortionGSS(map[m], in,NBITEREPOCH,  j+1);
      distortion_NI_test[m][j+1] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      distortion_NI_gss_test[m][j+1] = evaldistortionGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
    	
      learn_NF(map_NF[m], in, j);
      printf("****************\nAfter NF driven learning\n");
      quantization_NF[m][j+1] = errorrate(map[m], in,NBITEREPOCH,  j+1);
      quantization_NF_gss[m][j+1] = errorrateGSS(map[m], in,NBITEREPOCH,  j+1);
      quantization_NF_test[m][j+1] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      quantization_NF_gss_test[m][j+1] = errorrateGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);
      distortion_NF[m][j+1] = evaldistortion(map[m], in,NBITEREPOCH,  j+1);
      distortion_NF_gss[m][j+1] = evaldistortionGSS(map[m], in,NBITEREPOCH,  j+1);
      distortion_NF_test[m][j+1] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      distortion_NF_gss_test[m][j+1] = evaldistortionGSS(map[m], test,SIZE*SIZE*TEST_DENSITY,  j+1);

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

  double *** quantization2_test 		= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_th_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_FI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NF_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_th_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_FI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NF_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test_gss     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_th_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_FI_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NI_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NF_test_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test_faulty_gss     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_th_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_FI_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NI_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NF_test_faulty_gss  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

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

  double * avgdist = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NF = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NF = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NF_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NF_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_th_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_th_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_FI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_FI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NI_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NF_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NF_gss = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_th_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_th_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_FI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_FI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NI_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NF_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NF_faulty_gss = calloc(MAXFAULTPERCENT, sizeof(double));

  double tt = nb_experiments * NBMAPS;

  int ** test2 = (int **) malloc_2darray(SIZE*SIZE*TEST2_DENSITY, 2);

  for(i = 0; i < SIZE*SIZE*TEST2_DENSITY; i++) {
    for(k = 0; k < INS; k++) {
      test2[i][k] = (int) ((1.0 * one) * uniform_dataset(&rng)[k]);
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

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_th_test[p][e][m] = avg_quant_error(map_th[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_FI_test[p][e][m] = avg_quant_error(map_FI[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_NI_test[p][e][m] = avg_quant_error(map_NI[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_NF_test[p][e][m] = avg_quant_error(map_NF[m], test2,SIZE*SIZE*TEST2_DENSITY);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_th_test_faulty[p][e][m] = avg_quant_error(map2_th, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_FI_test_faulty[p][e][m] = avg_quant_error(map2_FI, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NI_test_faulty[p][e][m] = avg_quant_error(map2_NI, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NF_test_faulty[p][e][m] = avg_quant_error(map2_NF, test2,SIZE*SIZE*TEST2_DENSITY);

        quantization2_test_gss[p][e][m]    = avg_quant_error_GSS(map[m], test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_th_test_gss[p][e][m] = avg_quant_error_GSS(map_th[m], test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_FI_test_gss[p][e][m] = avg_quant_error_GSS(map_FI[m], test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NI_test_gss[p][e][m] = avg_quant_error_GSS(map_NI[m], test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NF_test_gss[p][e][m] = avg_quant_error_GSS(map_NF[m], test2,SIZE*SIZE*TEST2_DENSITY);

        quantization2_test_faulty_gss[p][e][m]    = avg_quant_error_GSS(map2, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_th_test_faulty_gss[p][e][m] = avg_quant_error_GSS(map2_th, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_FI_test_faulty_gss[p][e][m] = avg_quant_error_GSS(map2_FI, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NI_test_faulty_gss[p][e][m] = avg_quant_error_GSS(map2_NI, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NF_test_faulty_gss[p][e][m] = avg_quant_error_GSS(map2_NF, test2,SIZE*SIZE*TEST2_DENSITY);

        avg[p] += quantization2_test[p][e][m];
        avg_th[p] += quantization2_th_test[p][e][m];
        avg_FI[p] += quantization2_FI_test[p][e][m];
        avg_NI[p] += quantization2_NI_test[p][e][m];
        avg_NF[p] += quantization2_NF_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];
        avg_th_faulty[p] += quantization2_th_test_faulty[p][e][m];
        avg_FI_faulty[p] += quantization2_FI_test_faulty[p][e][m];
        avg_NI_faulty[p] += quantization2_NI_test_faulty[p][e][m];
        avg_NF_faulty[p] += quantization2_NF_test_faulty[p][e][m];

        avg_gss[p] += quantization2_test_gss[p][e][m];
        avg_th_gss[p] += quantization2_th_test_gss[p][e][m];
        avg_FI_gss[p] += quantization2_FI_test_gss[p][e][m];
        avg_NI_gss[p] += quantization2_NI_test_gss[p][e][m];
        avg_NF_gss[p] += quantization2_NF_test_gss[p][e][m];

        avg_faulty_gss[p] += quantization2_test_faulty_gss[p][e][m];
        avg_th_faulty_gss[p] += quantization2_th_test_faulty_gss[p][e][m];
        avg_FI_faulty_gss[p] += quantization2_FI_test_faulty_gss[p][e][m];
        avg_NI_faulty_gss[p] += quantization2_NI_test_faulty_gss[p][e][m];
        avg_NF_faulty_gss[p] += quantization2_NF_test_faulty_gss[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_th_test[p][e][m] = distortion_measure(map_th[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_FI_test[p][e][m] = distortion_measure(map_FI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_NI_test[p][e][m] = distortion_measure(map_NI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_NF_test[p][e][m] = distortion_measure(map_NF[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_th_test_faulty[p][e][m] = distortion_measure(map2_th,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_FI_test_faulty[p][e][m] = distortion_measure(map2_FI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NI_test_faulty[p][e][m] = distortion_measure(map2_NI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NF_test_faulty[p][e][m] = distortion_measure(map2_NF,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        distortion2_test_gss[p][e][m]    = distortion_measure_GSS(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_th_test_gss[p][e][m] = distortion_measure_GSS(map_th[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_FI_test_gss[p][e][m] = distortion_measure_GSS(map_FI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NI_test_gss[p][e][m] = distortion_measure_GSS(map_NI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NF_test_gss[p][e][m] = distortion_measure_GSS(map_NF[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        distortion2_test_faulty_gss[p][e][m]    = distortion_measure_GSS(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_th_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_th,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_FI_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_FI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NI_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_NI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NF_test_faulty_gss[p][e][m] = distortion_measure_GSS(map2_NF,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        avgdist[p] += distortion2_test[p][e][m];
        avgdist_th[p] += distortion2_th_test[p][e][m];
        avgdist_FI[p] += distortion2_FI_test[p][e][m];
        avgdist_NI[p] += distortion2_NI_test[p][e][m];
        avgdist_NF[p] += distortion2_NF_test[p][e][m];

        avgdist_faulty[p] += distortion2_test_faulty[p][e][m];
        avgdist_th_faulty[p] += distortion2_th_test_faulty[p][e][m];
        avgdist_FI_faulty[p] += distortion2_FI_test_faulty[p][e][m];
        avgdist_NI_faulty[p] += distortion2_NI_test_faulty[p][e][m];
        avgdist_NF_faulty[p] += distortion2_NF_test_faulty[p][e][m];

        avgdist_gss[p] += distortion2_test_gss[p][e][m];
        avgdist_th_gss[p] += distortion2_th_test_gss[p][e][m];
        avgdist_FI_gss[p] += distortion2_FI_test_gss[p][e][m];
        avgdist_NI_gss[p] += distortion2_NI_test_gss[p][e][m];
        avgdist_NF_gss[p] += distortion2_NF_test_gss[p][e][m];

        avgdist_faulty_gss[p] += distortion2_test_faulty_gss[p][e][m];
        avgdist_th_faulty_gss[p] += distortion2_th_test_faulty_gss[p][e][m];
        avgdist_FI_faulty_gss[p] += distortion2_FI_test_faulty_gss[p][e][m];
        avgdist_NI_faulty_gss[p] += distortion2_NI_test_faulty_gss[p][e][m];
        avgdist_NF_faulty_gss[p] += distortion2_NF_test_faulty_gss[p][e][m];

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
    
    avgdist[p] /= tt;
    avgdist_th[p] /= tt;
    avgdist_FI[p] /= tt;
    avgdist_NI[p] /= tt;
    avgdist_NF[p] /= tt;
    avgdist_faulty[p] /= tt;
    avgdist_th_faulty[p]  /= tt;
    avgdist_FI_faulty[p]  /= tt;
    avgdist_NI_faulty[p]  /= tt;
    avgdist_NF_faulty[p]  /= tt;
    avgdist_gss[p]  /= tt;
    avgdist_th_gss[p] /= tt;
    avgdist_FI_gss[p] /= tt;
    avgdist_NI_gss[p] /= tt;
    avgdist_NF_gss[p] /= tt;
    avgdist_faulty_gss[p] /= tt;
    avgdist_th_faulty_gss[p]  /= tt;
    avgdist_FI_faulty_gss[p]  /= tt;
    avgdist_NI_faulty_gss[p]  /= tt;
    avgdist_NF_faulty_gss[p]  /= tt;
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
        stddev[p] += (quantization2_test[p][e][m] - avg[p]) * (quantization2_test[p][e][m] - avg[p]);
        stddev_th[p] += (quantization2_th_test[p][e][m] - avg_th[p]) * (quantization2_th_test[p][e][m] - avg_th[p]);
        stddev_FI[p] += (quantization2_FI_test[p][e][m] - avg_FI[p]) * (quantization2_FI_test[p][e][m] - avg_FI[p]);
        stddev_NI[p] += (quantization2_NI_test[p][e][m] - avg_NI[p]) * (quantization2_NI_test[p][e][m] - avg_NI[p]);
        stddev_NF[p] += (quantization2_NF_test[p][e][m] - avg_NF[p]) * (quantization2_NF_test[p][e][m] - avg_NF[p]);

        stddev_faulty[p] += (quantization2_test_faulty[p][e][m] - avg_faulty[p]) * (quantization2_test_faulty[p][e][m] - avg_faulty[p]);
        stddev_th_faulty[p] += (quantization2_th_test_faulty[p][e][m] - avg_th_faulty[p]) * (quantization2_th_test_faulty[p][e][m] - avg_th_faulty[p]);
        stddev_FI_faulty[p] += (quantization2_FI_test_faulty[p][e][m] - avg_FI_faulty[p]) * (quantization2_FI_test_faulty[p][e][m] - avg_FI_faulty[p]);
        stddev_NI_faulty[p] += (quantization2_NI_test_faulty[p][e][m] - avg_NI_faulty[p]) * (quantization2_NI_test_faulty[p][e][m] - avg_NI_faulty[p]);
        stddev_NF_faulty[p] += (quantization2_NF_test_faulty[p][e][m] - avg_NF_faulty[p]) * (quantization2_NF_test_faulty[p][e][m] - avg_NF_faulty[p]);

        stddev_gss[p] += (quantization2_test_gss[p][e][m] - avg_gss[p]) * (quantization2_test_gss[p][e][m] - avg_gss[p]);
        stddev_th_gss[p] += (quantization2_th_test_gss[p][e][m] - avg_th_gss[p]) * (quantization2_th_test_gss[p][e][m] - avg_th_gss[p]);
        stddev_FI_gss[p] += (quantization2_FI_test_gss[p][e][m] - avg_FI_gss[p]) * (quantization2_FI_test_gss[p][e][m] - avg_FI_gss[p]);
        stddev_NI_gss[p] += (quantization2_NI_test_gss[p][e][m] - avg_NI_gss[p]) * (quantization2_NI_test_gss[p][e][m] - avg_NI_gss[p]);
        stddev_NF_gss[p] += (quantization2_NF_test_gss[p][e][m] - avg_NF_gss[p]) * (quantization2_NF_test_gss[p][e][m] - avg_NF_gss[p]);

        stddev_faulty_gss[p] += (quantization2_test_faulty_gss[p][e][m] - avg_faulty_gss[p]) * (quantization2_test_faulty_gss[p][e][m] - avg_faulty_gss[p]);
        stddev_th_faulty_gss[p] += (quantization2_th_test_faulty_gss[p][e][m] - avg_th_faulty_gss[p]) * (quantization2_th_test_faulty_gss[p][e][m] - avg_th_faulty_gss[p]);
        stddev_FI_faulty_gss[p] += (quantization2_FI_test_faulty_gss[p][e][m] - avg_FI_faulty_gss[p]) * (quantization2_FI_test_faulty_gss[p][e][m] - avg_FI_faulty_gss[p]);
        stddev_NI_faulty_gss[p] += (quantization2_NI_test_faulty_gss[p][e][m] - avg_NI_faulty_gss[p]) * (quantization2_NI_test_faulty_gss[p][e][m] - avg_NI_faulty_gss[p]);
        stddev_NF_faulty_gss[p] += (quantization2_NF_test_faulty_gss[p][e][m] - avg_NF_faulty_gss[p]) * (quantization2_NF_test_faulty_gss[p][e][m] - avg_NF_faulty_gss[p]);

	stddevdist[p] += (distortion2_test[p][e][m] - avgdist[p]) * (distortion2_test[p][e][m] - avgdist[p]);
        stddevdist_th[p] += (distortion2_th_test[p][e][m] - avgdist_th[p]) * (distortion2_th_test[p][e][m] - avgdist_th[p]);
        stddevdist_FI[p] += (distortion2_FI_test[p][e][m] - avgdist_FI[p]) * (distortion2_FI_test[p][e][m] - avgdist_FI[p]);
        stddevdist_NI[p] += (distortion2_NI_test[p][e][m] - avgdist_NI[p]) * (distortion2_NI_test[p][e][m] - avgdist_NI[p]);
        stddevdist_NF[p] += (distortion2_NF_test[p][e][m] - avgdist_NF[p]) * (distortion2_NF_test[p][e][m] - avgdist_NF[p]);

        stddevdist_faulty[p] += (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]) * (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]);
        stddevdist_th_faulty[p] += (distortion2_th_test_faulty[p][e][m] - avgdist_th_faulty[p]) * (distortion2_th_test_faulty[p][e][m] - avgdist_th_faulty[p]);
        stddevdist_FI_faulty[p] += (distortion2_FI_test_faulty[p][e][m] - avgdist_FI_faulty[p]) * (distortion2_FI_test_faulty[p][e][m] - avgdist_FI_faulty[p]);
        stddevdist_NI_faulty[p] += (distortion2_NI_test_faulty[p][e][m] - avgdist_NI_faulty[p]) * (distortion2_NI_test_faulty[p][e][m] - avgdist_NI_faulty[p]);
        stddevdist_NF_faulty[p] += (distortion2_NF_test_faulty[p][e][m] - avgdist_NF_faulty[p]) * (distortion2_NF_test_faulty[p][e][m] - avgdist_NF_faulty[p]);

        stddevdist_gss[p] += (distortion2_test_gss[p][e][m] - avgdist_gss[p]) * (distortion2_test_gss[p][e][m] - avgdist_gss[p]);
        stddevdist_th_gss[p] += (distortion2_th_test_gss[p][e][m] - avgdist_th_gss[p]) * (distortion2_th_test_gss[p][e][m] - avgdist_th_gss[p]);
        stddevdist_FI_gss[p] += (distortion2_FI_test_gss[p][e][m] - avgdist_FI_gss[p]) * (distortion2_FI_test_gss[p][e][m] - avgdist_FI_gss[p]);
        stddevdist_NI_gss[p] += (distortion2_NI_test_gss[p][e][m] - avgdist_NI_gss[p]) * (distortion2_NI_test_gss[p][e][m] - avgdist_NI_gss[p]);
        stddevdist_NF_gss[p] += (distortion2_NF_test_gss[p][e][m] - avgdist_NF_gss[p]) * (distortion2_NF_test_gss[p][e][m] - avgdist_NF_gss[p]);

        stddevdist_faulty_gss[p] += (distortion2_test_faulty_gss[p][e][m] - avgdist_faulty_gss[p]) * (distortion2_test_faulty_gss[p][e][m] - avgdist_faulty_gss[p]);
        stddevdist_th_faulty_gss[p] += (distortion2_th_test_faulty_gss[p][e][m] - avgdist_th_faulty_gss[p]) * (distortion2_th_test_faulty_gss[p][e][m] - avgdist_th_faulty_gss[p]);
        stddevdist_FI_faulty_gss[p] += (distortion2_FI_test_faulty_gss[p][e][m] - avgdist_FI_faulty_gss[p]) * (distortion2_FI_test_faulty_gss[p][e][m] - avgdist_FI_faulty_gss[p]);
        stddevdist_NI_faulty_gss[p] += (distortion2_NI_test_faulty_gss[p][e][m] - avgdist_NI_faulty_gss[p]) * (distortion2_NI_test_faulty_gss[p][e][m] - avgdist_NI_faulty_gss[p]);
        stddevdist_NF_faulty_gss[p] += (distortion2_NF_test_faulty_gss[p][e][m] - avgdist_NF_faulty_gss[p]) * (distortion2_NF_test_faulty_gss[p][e][m] - avgdist_NF_faulty_gss[p]);
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

    stddevdist[p] = mysqrt(stddevdist[p]/(tt-1));
    stddevdist_th[p] = mysqrt(stddevdist_th[p]/(tt-1));
    stddevdist_FI[p] = mysqrt(stddevdist_FI[p]/(tt-1));
    stddevdist_NI[p] = mysqrt(stddevdist_NI[p]/(tt-1)); 
    stddevdist_NF[p] = mysqrt(stddevdist_NF[p]/(tt-1));

    stddevdist_faulty[p] = mysqrt(stddevdist_faulty[p]/(tt-1)); 
    stddevdist_th_faulty[p] = mysqrt(stddevdist_th_faulty[p]/(tt-1));
    stddevdist_FI_faulty[p] = mysqrt(stddevdist_FI_faulty[p]/(tt-1));
    stddevdist_NI_faulty[p] = mysqrt(stddevdist_NI_faulty[p]/(tt-1));
    stddevdist_NF_faulty[p] = mysqrt(stddevdist_NF_faulty[p]/(tt-1));

    stddevdist_gss[p] = mysqrt(stddevdist_gss[p]/(tt-1));
    stddevdist_th_gss[p] = mysqrt(stddevdist_th_gss[p]/(tt-1));
    stddevdist_FI_gss[p] = mysqrt(stddevdist_FI_gss[p]/(tt-1));
    stddevdist_NI_gss[p] = mysqrt(stddevdist_NI_gss[p]/(tt-1)); 
    stddevdist_NF_gss[p] = mysqrt(stddevdist_NF_gss[p]/(tt-1));

    stddevdist_faulty_gss[p] = mysqrt(stddevdist_faulty_gss[p]/(tt-1)); 
    stddevdist_th_faulty_gss[p] = mysqrt(stddevdist_th_faulty_gss[p]/(tt-1));
    stddevdist_FI_faulty_gss[p] = mysqrt(stddevdist_FI_faulty_gss[p]/(tt-1));
    stddevdist_NI_faulty_gss[p] = mysqrt(stddevdist_NI_faulty_gss[p]/(tt-1));
    stddevdist_NF_faulty_gss[p] = mysqrt(stddevdist_NF_faulty_gss[p]/(tt-1));
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
  fp = fopen ("learn_quantization.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", m, j,
	      quantization[m][j], quantization_th[m][j], quantization_FI[m][j], quantization_NI[m][j], quantization_NF[m][j],
	      quantization_gss[m][j], quantization_th_gss[m][j], quantization_FI_gss[m][j], 
	      quantization_NI_gss[m][j], quantization_NF_gss[m][j]);
    }
  }
  fclose(fp);

  fp = fopen ("learn_distortion.txt", "w+");
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
  fp = fopen ("test_quantization_during_learn.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", m, j,
	      quantization_test[m][j], quantization_th_test[m][j], quantization_FI_test[m][j], quantization_NI_test[m][j],
	      quantization_NF_test[m][j], quantization_gss_test[m][j], quantization_th_gss_test[m][j],
	      quantization_FI_gss_test[m][j], quantization_NI_gss_test[m][j], quantization_NF_gss_test[m][j]);
    }
  }
  fclose(fp);

  fp = fopen ("test_distortion_during_learn.txt", "w+");
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
  fp = fopen ("test_distortion.txt", "w+");
  fprintf(fp, "Percentage_faults;Experiment_number;Map_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p, e+1, m+1,
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

  fp = fopen ("test_quantization.txt", "w+");
  fprintf(fp, "Percentage_faults;Experiment_number;Map_number;Standard;Threshold;FaultInjection;NoiseInjetion;NeuralField;Standard_GSS;Threshold_GSS;FaultInjection_GSS;NoiseInjetion_GSS;NeuralField_GSS\n");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p, e+1, m+1,
		quantization2_test[p][e][m], quantization2_th_test[p][e][m], quantization2_FI_test[p][e][m], quantization2_NI_test[p][e][m], 
		quantization2_NF_test[p][e][m],
		quantization2_test_faulty[p][e][m], quantization2_th_test_faulty[p][e][m], quantization2_FI_test_faulty[p][e][m], 
		quantization2_NI_test_faulty[p][e][m], quantization2_NF_test_faulty[p][e][m],
		quantization2_test_gss[p][e][m], quantization2_th_test_gss[p][e][m], quantization2_FI_test_gss[p][e][m], 
		quantization2_NI_test_gss[p][e][m], quantization2_NF_test_gss[p][e][m],
		quantization2_test_faulty_gss[p][e][m], quantization2_th_test_faulty_gss[p][e][m], 
		quantization2_FI_test_faulty_gss[p][e][m], quantization2_NI_test_faulty_gss[p][e][m], quantization2_NF_test_faulty_gss[p][e][m]);
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

  fp = fopen ("statistics_quantization.txt", "w+");
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

  fp = fopen ("statistics_distortion.txt", "w+");
  fprintf(fp, "Percentage_faults;Standard_avgdist;Standard_std;Threshold_avgdist;Threshold_std;FaultInjection_avgdist;FaultInjection_std;\
    NoiseInjetion_avgdist;NoiseInjetion_std;NeuralField_avgdist;NeuralField_std;Standard_GSS_avgdist;Standard_GSS_std;Threshold_GSS_avgdist;Threshold_GSS_std;\
    FaultInjection_GSS_avgdist;FaultInjection_GSS_std;NoiseInjetion_GSS_avgdist;NoiseInjetion_GSS_std;NeuralField_GSS_avgdist;NeuralField_GSS_std\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p, 
	    avgdist_faulty[p], stddevdist_faulty[p], avgdist_th_faulty[p], stddevdist_th_faulty[p], avgdist_FI_faulty[p], stddevdist_FI_faulty[p], 
	    avgdist_NI_faulty[p], stddevdist_NI_faulty[p], avgdist_NF_faulty[p], stddevdist_NF_faulty[p],
	    avgdist_faulty_gss[p], stddevdist_faulty_gss[p], avgdist_th_faulty_gss[p], stddevdist_th_faulty_gss[p], avgdist_FI_faulty_gss[p], stddevdist_FI_faulty_gss[p], 
	    avgdist_NI_faulty_gss[p], stddevdist_NI_faulty_gss[p], avgdist_NF_faulty_gss[p], stddevdist_NF_faulty_gss[p]);
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
