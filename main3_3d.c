
#include "func_def.h"


#include "stat.h"
#include "time.h"
#include "custom_rand.h"
#include "gss_som.h"

Kohonen *map;
Kohonen *map_th;
Kohonen *map_FI;
Kohonen *map_NI;
pcg32_random_t rng;

/*void precision_faulty_weights() {
  
  }*/

void fault_tolerance(int ep) {
  double *** distortion2_test 		= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_th_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_FI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** distortion2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_th_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_FI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** distortion2_NI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_th_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_FI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NI_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_th_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_FI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
  double *** quantization2_NI_test_faulty  = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double * avg = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NI = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avg_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avg_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_th = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_FI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NI = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NI = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_th_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_FI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * avgdist_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_NI_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  int p,e,m,i,k;
  double tt = nb_experiments * NBMAPS;

  int ** test2 = (int **) malloc_2darray(SIZE*SIZE*TEST2_DENSITY, INS);
  FILE 	*	fp;
  

  for(i = 0; i < SIZE*SIZE*TEST2_DENSITY; i++) {
    double *v= DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      test2[i][k] = (int) ((1.0 * one) *v[k]);
    }
    free(v);
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
	Kohonen map2    = copy(map[m]);
	Kohonen map2_th = copy(map_th[m]);
	Kohonen map2_FI = copy(map_FI[m]);
	Kohonen map2_NI = copy(map_NI[m]);

	// introduction of faults in the copies of the pre-learned maps
	faulty_weights(map2, p);
	faulty_weights(map2_th, p);
	faulty_weights(map2_FI, p);
	faulty_weights(map2_NI, p);

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_th_test[p][e][m] = avg_quant_error(map_th[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_FI_test[p][e][m] = avg_quant_error(map_FI[m], test2,SIZE*SIZE*TEST2_DENSITY);
	quantization2_NI_test[p][e][m] = avg_quant_error(map_NI[m], test2,SIZE*SIZE*TEST2_DENSITY);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_th_test_faulty[p][e][m] = avg_quant_error(map2_th, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_FI_test_faulty[p][e][m] = avg_quant_error(map2_FI, test2,SIZE*SIZE*TEST2_DENSITY);
        quantization2_NI_test_faulty[p][e][m] = avg_quant_error(map2_NI, test2,SIZE*SIZE*TEST2_DENSITY);

        avg[p] += quantization2_test[p][e][m];
        avg_th[p] += quantization2_th_test[p][e][m];
        avg_FI[p] += quantization2_FI_test[p][e][m];
        avg_NI[p] += quantization2_NI_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];
        avg_th_faulty[p] += quantization2_th_test_faulty[p][e][m];
        avg_FI_faulty[p] += quantization2_FI_test_faulty[p][e][m];
        avg_NI_faulty[p] += quantization2_NI_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_th_test[p][e][m] = distortion_measure(map_th[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_FI_test[p][e][m] = distortion_measure(map_FI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
	distortion2_NI_test[p][e][m] = distortion_measure(map_NI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_th_test_faulty[p][e][m] = distortion_measure(map2_th,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_FI_test_faulty[p][e][m] = distortion_measure(map2_FI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);
        distortion2_NI_test_faulty[p][e][m] = distortion_measure(map2_NI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        avgdist[p] += distortion2_test[p][e][m];
        avgdist_th[p] += distortion2_th_test[p][e][m];
        avgdist_FI[p] += distortion2_FI_test[p][e][m];
        avgdist_NI[p] += distortion2_NI_test[p][e][m];

        avgdist_faulty[p] += distortion2_test_faulty[p][e][m];
        avgdist_th_faulty[p] += distortion2_th_test_faulty[p][e][m];
        avgdist_FI_faulty[p] += distortion2_FI_test_faulty[p][e][m];
        avgdist_NI_faulty[p] += distortion2_NI_test_faulty[p][e][m];

      } // end loop on map initializations
    } // end loop on experiments (faulty versions)

    avg[p] /= tt;
    avg_th[p] /= tt;
    avg_FI[p] /= tt;
    avg_NI[p] /= tt;

    avg_faulty[p] /= tt;
    avg_th_faulty[p]  /= tt;
    avg_FI_faulty[p]  /= tt;
    avg_NI_faulty[p]  /= tt;
    
    avgdist[p] /= tt;
    avgdist_th[p] /= tt;
    avgdist_FI[p] /= tt;
    avgdist_NI[p] /= tt;

    avgdist_faulty[p] /= tt;
    avgdist_th_faulty[p]  /= tt;
    avgdist_FI_faulty[p]  /= tt;
    avgdist_NI_faulty[p]  /= tt;
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
        stddev[p] += (quantization2_test[p][e][m] - avg[p]) * (quantization2_test[p][e][m] - avg[p]);
        stddev_th[p] += (quantization2_th_test[p][e][m] - avg_th[p]) * (quantization2_th_test[p][e][m] - avg_th[p]);
        stddev_FI[p] += (quantization2_FI_test[p][e][m] - avg_FI[p]) * (quantization2_FI_test[p][e][m] - avg_FI[p]);
        stddev_NI[p] += (quantization2_NI_test[p][e][m] - avg_NI[p]) * (quantization2_NI_test[p][e][m] - avg_NI[p]);

        stddev_faulty[p] += (quantization2_test_faulty[p][e][m] - avg_faulty[p]) * (quantization2_test_faulty[p][e][m] - avg_faulty[p]);
        stddev_th_faulty[p] += (quantization2_th_test_faulty[p][e][m] - avg_th_faulty[p]) * (quantization2_th_test_faulty[p][e][m] - avg_th_faulty[p]);
        stddev_FI_faulty[p] += (quantization2_FI_test_faulty[p][e][m] - avg_FI_faulty[p]) * (quantization2_FI_test_faulty[p][e][m] - avg_FI_faulty[p]);
        stddev_NI_faulty[p] += (quantization2_NI_test_faulty[p][e][m] - avg_NI_faulty[p]) * (quantization2_NI_test_faulty[p][e][m] - avg_NI_faulty[p]);

	stddevdist[p] += (distortion2_test[p][e][m] - avgdist[p]) * (distortion2_test[p][e][m] - avgdist[p]);
        stddevdist_th[p] += (distortion2_th_test[p][e][m] - avgdist_th[p]) * (distortion2_th_test[p][e][m] - avgdist_th[p]);
        stddevdist_FI[p] += (distortion2_FI_test[p][e][m] - avgdist_FI[p]) * (distortion2_FI_test[p][e][m] - avgdist_FI[p]);
        stddevdist_NI[p] += (distortion2_NI_test[p][e][m] - avgdist_NI[p]) * (distortion2_NI_test[p][e][m] - avgdist_NI[p]);

        stddevdist_faulty[p] += (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]) * (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]);
        stddevdist_th_faulty[p] += (distortion2_th_test_faulty[p][e][m] - avgdist_th_faulty[p]) * (distortion2_th_test_faulty[p][e][m] - avgdist_th_faulty[p]);
        stddevdist_FI_faulty[p] += (distortion2_FI_test_faulty[p][e][m] - avgdist_FI_faulty[p]) * (distortion2_FI_test_faulty[p][e][m] - avgdist_FI_faulty[p]);
        stddevdist_NI_faulty[p] += (distortion2_NI_test_faulty[p][e][m] - avgdist_NI_faulty[p]) * (distortion2_NI_test_faulty[p][e][m] - avgdist_NI_faulty[p]);
      }
    }
    stddev[p] = mysqrt(stddev[p]/(tt-1));
    stddev_th[p] = mysqrt(stddev_th[p]/(tt-1));
    stddev_FI[p] = mysqrt(stddev_FI[p]/(tt-1));
    stddev_NI[p] = mysqrt(stddev_NI[p]/(tt-1)); 

    stddev_faulty[p] = mysqrt(stddev_faulty[p]/(tt-1)); 
    stddev_th_faulty[p] = mysqrt(stddev_th_faulty[p]/(tt-1));
    stddev_FI_faulty[p] = mysqrt(stddev_FI_faulty[p]/(tt-1));
    stddev_NI_faulty[p] = mysqrt(stddev_NI_faulty[p]/(tt-1));

    stddevdist[p] = mysqrt(stddevdist[p]/(tt-1));
    stddevdist_th[p] = mysqrt(stddevdist_th[p]/(tt-1));
    stddevdist_FI[p] = mysqrt(stddevdist_FI[p]/(tt-1));
    stddevdist_NI[p] = mysqrt(stddevdist_NI[p]/(tt-1)); 

    stddevdist_faulty[p] = mysqrt(stddevdist_faulty[p]/(tt-1)); 
    stddevdist_th_faulty[p] = mysqrt(stddevdist_th_faulty[p]/(tt-1));
    stddevdist_FI_faulty[p] = mysqrt(stddevdist_FI_faulty[p]/(tt-1));
    stddevdist_NI_faulty[p] = mysqrt(stddevdist_NI_faulty[p]/(tt-1));

  }

  fp = fopen ("test_distortion_faulty.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Experiment_number;Map_number;Standard;Threshold;FaultInjection;NoiseInjetion;Standard_faulty;Threshold_faulty;FaultInjection_faulty;NoiseInjection_faulty;Standard_tolerance;Threshold_tolerance;FaultInjection_tolerance;NoiseInjection_tolerance\n");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", ep, p, e+1, m+1,
		distortion2_test[p][e][m], distortion2_th_test[p][e][m], distortion2_FI_test[p][e][m], distortion2_NI_test[p][e][m],
		distortion2_test_faulty[p][e][m], distortion2_th_test_faulty[p][e][m], distortion2_FI_test_faulty[p][e][m], 
		distortion2_NI_test_faulty[p][e][m],
		distortion2_test_faulty[p][e][m]/distortion2_test[p][e][m], distortion2_th_test_faulty[p][e][m]/distortion2_th_test[p][e][m],
		distortion2_FI_test_faulty[p][e][m]/distortion2_FI_test[p][e][m], 
		distortion2_NI_test_faulty[p][e][m]/distortion2_NI_test[p][e][m]);
      }
    }
  }
  fclose(fp);

  fp = fopen ("test_quantization_faulty.txt", "a+");
  fprintf(fp, "Epoch; Percentage_faults;Experiment_number;Map_number;Standard;Threshold;FaultInjection;NoiseInjection;Standard_faulty;Threshold_faulty;FaultInjection_faulty;NoiseInjection_faulty\n;Standard_tolerance;Threshold_tolerance;FaultInjection_tolerance;NoiseInjection_tolerance");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", ep, p, e+1, m+1,
		quantization2_test[p][e][m], quantization2_th_test[p][e][m], quantization2_FI_test[p][e][m], quantization2_NI_test[p][e][m], 
		quantization2_test_faulty[p][e][m], quantization2_th_test_faulty[p][e][m], quantization2_FI_test_faulty[p][e][m], 
		quantization2_NI_test_faulty[p][e][m],
		quantization2_test_faulty[p][e][m]/quantization2_test[p][e][m], quantization2_th_test_faulty[p][e][m]/quantization2_th_test[p][e][m],
		quantization2_FI_test_faulty[p][e][m]/quantization2_FI_test[p][e][m], 
		quantization2_NI_test_faulty[p][e][m]/quantization2_NI_test[p][e][m]);
      }
    }
  }
  fclose(fp);

  fp = fopen ("statistics_quantization.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg;Threshold_avg;Threshold_std;Threshold_ratio_avg;FaultInjection_avg;FaultInjection_std;FaultInjection_ratio_avg;NoiseInjection_avg;NoiseInjection_std;NoiseInjection_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", ep, p, 
	    avg_faulty[p], stddev_faulty[p], avg_faulty[p]/avg[p], avg_th_faulty[p], stddev_th_faulty[p], avg_th_faulty[p]/avg_th[p], avg_FI_faulty[p], stddev_FI_faulty[p], avg_FI_faulty[p]/avg_FI[p], 
	    avg_NI_faulty[p], stddev_NI_faulty[p], avg_NI_faulty[p]/avg_NI[p]);
  }
  fclose(fp);

  fp = fopen ("statistics_distortion.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg;Threshold_avg;Threshold_std;Threshold_ratio_avg;FaultInjection_avg;FaultInjection_std;FaultInjection_ratio_avg;NoiseInjection_avg;NoiseInjection_std;NoiseInjection_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", ep, p, 
	    avgdist_faulty[p], stddevdist_faulty[p], avgdist_faulty[p]/avgdist[p], avgdist_th_faulty[p], stddevdist_th_faulty[p], avgdist_th_faulty[p]/avgdist_th[p], avgdist_FI_faulty[p], stddevdist_FI_faulty[p], avgdist_FI_faulty[p]/avgdist_FI[p], 
	    avgdist_NI_faulty[p], stddevdist_NI_faulty[p], avgdist_NI_faulty[p]/avgdist_NI[p]);
  }
  fclose(fp);
  free_3darray(distortion2_test 		,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_th_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_FI_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_NI_test 	,MAXFAULTPERCENT,nb_experiments);

  free_3darray(distortion2_test_faulty     ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_th_test_faulty  ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_FI_test_faulty  ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_NI_test_faulty  ,MAXFAULTPERCENT,nb_experiments);

  free_3darray(quantization2_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_th_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_FI_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_NI_test 	,MAXFAULTPERCENT,nb_experiments);

  free_3darray(quantization2_test_faulty     ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_th_test_faulty  ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_FI_test_faulty  ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_NI_test_faulty  ,MAXFAULTPERCENT,nb_experiments);

  free(avg);
  free(stddev);
  free(avg_th);
  free(stddev_th);
  free(avg_FI);
  free(stddev_FI);
  free(avg_NI);
  free(stddev_NI);

  free(avg_faulty);
  free(stddev_faulty);
  free(avg_th_faulty);
  free(stddev_th_faulty);
  free(avg_FI_faulty);
  free(stddev_FI_faulty);
  free(avg_NI_faulty);
  free(stddev_NI_faulty);

  free(avgdist);
  free(stddevdist);
  free(avgdist_th);
  free(stddevdist_th);
  free(avgdist_FI);
  free(stddevdist_FI);
  free(avgdist_NI);
  free(stddevdist_NI);

  free(avgdist_faulty);
  free(stddevdist_faulty);
  free(avgdist_th_faulty);
  free(stddevdist_th_faulty);
  free(avgdist_FI_faulty);
  free(stddevdist_FI_faulty);
  free(avgdist_NI_faulty);
  free(stddevdist_NI_faulty);

}

void fault_tolerance_eco(int ep) {
  double *** distortion2_test 		= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** distortion2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double * avg = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avg_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  int p,e,m,i,k;
  double tt = nb_experiments * NBMAPS;

  int ** test2 = (int **) malloc_2darray(SIZE*SIZE*TEST2_DENSITY, INS);
  FILE 	*	fp;
  

  for(i = 0; i < SIZE*SIZE*TEST2_DENSITY; i++) {
    double *v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      test2[i][k] = (int) ((1.0 * one) * v[k]);
    }
    free(v);
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
	Kohonen map2    = copy(map[m]);

	// introduction of faults in the copies of the pre-learned maps
	faulty_weights(map2, p);

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,SIZE*SIZE*TEST2_DENSITY);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,SIZE*SIZE*TEST2_DENSITY);

        avg[p] += quantization2_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        avgdist[p] += distortion2_test[p][e][m];

        avgdist_faulty[p] += distortion2_test_faulty[p][e][m];

      } // end loop on map initializations
    } // end loop on experiments (faulty versions)

    avg[p] /= tt;

    avg_faulty[p] /= tt;
    
    avgdist[p] /= tt;

    avgdist_faulty[p] /= tt;
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
        stddev[p] += (quantization2_test[p][e][m] - avg[p]) * (quantization2_test[p][e][m] - avg[p]);

        stddev_faulty[p] += (quantization2_test_faulty[p][e][m] - avg_faulty[p]) * (quantization2_test_faulty[p][e][m] - avg_faulty[p]);

	stddevdist[p] += (distortion2_test[p][e][m] - avgdist[p]) * (distortion2_test[p][e][m] - avgdist[p]);

        stddevdist_faulty[p] += (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]) * (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]);
      }
    }
    stddev[p] = mysqrt(stddev[p]/(tt-1));

    stddev_faulty[p] = mysqrt(stddev_faulty[p]/(tt-1)); 

    stddevdist[p] = mysqrt(stddevdist[p]/(tt-1));

    stddevdist_faulty[p] = mysqrt(stddevdist_faulty[p]/(tt-1)); 

  }

  fp = fopen ("test_distortion_faulty.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Experiment_number;Map_number;Standard;Standard_faulty;Standard_tolerance\n");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f\n", ep, p, e+1, m+1,
		distortion2_test[p][e][m],
		distortion2_test_faulty[p][e][m],
		distortion2_test_faulty[p][e][m]/distortion2_test[p][e][m]);
      }
    }
  }
  fclose(fp);

  fp = fopen ("test_quantization_faulty.txt", "a+");
  fprintf(fp, "Epoch; Percentage_faults;Experiment_number;Map_number;Standard;Standard_faulty;Standard_tolerance");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f\n", ep, p, e+1, m+1,
		quantization2_test[p][e][m], 
		quantization2_test_faulty[p][e][m],
		quantization2_test_faulty[p][e][m]/quantization2_test[p][e][m]);
      }
    }
  }
  fclose(fp);

  fp = fopen ("statistics_quantization.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f\n", ep, p, 
	    avg_faulty[p], stddev_faulty[p], avg_faulty[p]/avg[p]);
  }
  fclose(fp);

  fp = fopen ("statistics_distortion.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avgdist;Standard_std;Standard_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f\n", ep, p, 
	    avgdist_faulty[p], stddevdist_faulty[p], avgdist_faulty[p]/avgdist[p]);
  }
  fclose(fp);

  free_3darray(distortion2_test 		,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_test_faulty     ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_test_faulty     ,MAXFAULTPERCENT,nb_experiments);

  free(avg);
  free(stddev);
  free(avg_faulty);
  free(stddev_faulty);
  free(avgdist);
  free(stddevdist);
  free(avgdist_faulty);
  free(stddevdist_faulty);

}

void fault_tolerance_gss(int ep) {
  double *** distortion2_test 		= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** distortion2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test 	= malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double *** quantization2_test_faulty     = malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);

  double * avg = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avg_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddev_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist = calloc(MAXFAULTPERCENT, sizeof(double));

  double * avgdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
  double * stddevdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));

  int p,e,m,i,k;
  double tt = nb_experiments * NBMAPS;

  int ** test2 = (int **) malloc_2darray(SIZE*SIZE*TEST2_DENSITY, INS);
  FILE 	*	fp;
  

  for(i = 0; i < SIZE*SIZE*TEST2_DENSITY; i++) {
    double*v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      test2[i][k] = (int) ((1.0 * one) * v[k]);
    }
    free(v);
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
	Kohonen map2    = copy(map[m]);

	// introduction of faults in the copies of the pre-learned maps
	faulty_weights(map2, p);
	gaussian_prototypes(map[m],SIGMA_GAUSS);
	gaussian_prototypes(map2,SIGMA_GAUSS);

	quantization2_test[p][e][m] 	 = avg_quant_error_GSS(map[m], test2,SIZE*SIZE*TEST2_DENSITY);

        quantization2_test_faulty[p][e][m]    = avg_quant_error_GSS(map2, test2,SIZE*SIZE*TEST2_DENSITY);

        avg[p] += quantization2_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = protodistortion_measure_GSS(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        distortion2_test_faulty[p][e][m]    = protodistortion_measure_GSS(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS);

        avgdist[p] += distortion2_test[p][e][m];

        avgdist_faulty[p] += distortion2_test_faulty[p][e][m];

      } // end loop on map initializations
    } // end loop on experiments (faulty versions)

    avg[p] /= tt;

    avg_faulty[p] /= tt;
    
    avgdist[p] /= tt;

    avgdist_faulty[p] /= tt;
  }

  for (p = 0; p < MAXFAULTPERCENT; p++) {
    for (e = 0; e < nb_experiments; e++) {
      for (m = 0; m < NBMAPS; m++) {
        stddev[p] += (quantization2_test[p][e][m] - avg[p]) * (quantization2_test[p][e][m] - avg[p]);

        stddev_faulty[p] += (quantization2_test_faulty[p][e][m] - avg_faulty[p]) * (quantization2_test_faulty[p][e][m] - avg_faulty[p]);

	stddevdist[p] += (distortion2_test[p][e][m] - avgdist[p]) * (distortion2_test[p][e][m] - avgdist[p]);

        stddevdist_faulty[p] += (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]) * (distortion2_test_faulty[p][e][m] - avgdist_faulty[p]);
      }
    }
    stddev[p] = mysqrt(stddev[p]/(tt-1));

    stddev_faulty[p] = mysqrt(stddev_faulty[p]/(tt-1)); 

    stddevdist[p] = mysqrt(stddevdist[p]/(tt-1));

    stddevdist_faulty[p] = mysqrt(stddevdist_faulty[p]/(tt-1)); 

  }

  fp = fopen ("test_protodistortionGSS_faulty.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Experiment_number;Map_number;Standard;Standard_faulty;Standard_tolerance\n");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f\n", ep, p, e+1, m+1,
		distortion2_test[p][e][m],
		distortion2_test_faulty[p][e][m],
		distortion2_test_faulty[p][e][m]/distortion2_test[p][e][m]);
      }
    }
  }
  fclose(fp);

  fp = fopen ("test_quantizationGSS_faulty.txt", "a+");
  fprintf(fp, "Epoch; Percentage_faults;Experiment_number;Map_number;Standard;Standard_faulty;Standard_tolerance");
  for (p = 0; p < MAXFAULTPERCENT; p++){
    for (e = 0; e < nb_experiments; e++){
      for (m = 0; m < NBMAPS; m++){
        fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f\n", ep, p, e+1, m+1,
		quantization2_test[p][e][m], 
		quantization2_test_faulty[p][e][m],
		quantization2_test_faulty[p][e][m]/quantization2_test[p][e][m]);
      }
    }
  }
  fclose(fp);

  fp = fopen ("statistics_quantization_GSS.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f\n", ep, p, 
	    avg_faulty[p], stddev_faulty[p], avg_faulty[p]/avg[p]);
  }
  fclose(fp);

  fp = fopen ("statistics_protodistortionGSS.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avgdist;Standard_std;Standard_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f\n", ep, p, 
	    avgdist_faulty[p], stddevdist_faulty[p], avgdist_faulty[p]/avgdist[p]);
  }
  fclose(fp);

  
  free_3darray(distortion2_test 		,MAXFAULTPERCENT,nb_experiments);
  free_3darray(distortion2_test_faulty     ,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_test 	,MAXFAULTPERCENT,nb_experiments);
  free_3darray(quantization2_test_faulty     ,MAXFAULTPERCENT,nb_experiments);

  free(avg);
  free(stddev);
  free(avg_faulty);
  free(stddev_faulty);
  free(avgdist);
  free(stddevdist);
  free(avgdist_faulty);
  free(stddevdist_faulty);

}

int main(){
  
  init_random(&rng);

  srand(time(NULL));
  clock_t 	start = clock();
  int  p,i,j,e,k,m;

  map = malloc(NBMAPS*sizeof(Kohonen));
  Kohonen *mapinit  = malloc(NBMAPS*sizeof(Kohonen));

  /*  
      calculate distortion for each map for each epoche
      NBEPOCHLEARN+1 - # of distortions measure + measure before learning
  */
  double  ** quantization     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** quantization_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** distortion     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  double  ** distortion_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  
  for (i = 0; i < NBMAPS; i++) mapinit[i] = init();
  
  int ** in = (int **) malloc_2darray(NBITEREPOCH, INS);
  int ** test = (int **) malloc_2darray(SIZE*SIZE*TEST_DENSITY, INS);
  
  for(i = 0; i < NBITEREPOCH; i++) {
    double *v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      in[i][k] = (int) ((1.0 * one) * v[k]);
    }
    free(v);
  }
  
  for(i = 0; i < SIZE*SIZE*TEST_DENSITY; i++) {
    double *v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      test[i][k] = (int) ((1.0 * one) * v[k]);
    }
    free(v);
  }
  
  /*
    for(i = 0; i < SIZE*SIZE*100; i++) {
    printf("%g %g\n",test[i][0]/(1.0*one),test[i][1]/(1.0*one));
    }
  */
    
  for (m = 0; m < NBMAPS; m++) {
    
    printf("\n**************\
  		******************\n map number %d : \n\n",m);
    map[m]		=	copy(mapinit[m]);
    
    printf("****************\nBefore learning\n");
    printf("learn:\n");
    quantization[m][0] = errorrate(map[m], in,NBITEREPOCH, 0);

    printf("test:\n");
    quantization_test[m][0] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
    
    printf("learn:\n");
    distortion[m][0] = evaldistortion(map[m], in,NBITEREPOCH, 0);
    
    printf("test:\n");
    distortion_test[m][0] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);

  }

  fault_tolerance_eco(0);
  fault_tolerance_gss(0);

  for(j = 0; j < NBEPOCHLEARN; j++){
    // generate new random values
    for(i = 0; i < NBITEREPOCH; i++) {
      double *v=DISTRIB(&rng);
      for(k = 0; k < INS; k++) {
	in[i][k] = (int) ((1.0 * one) * v[k]);
      }
      free(v);
    }

    for (m = 0; m < NBMAPS; m++) {
      printf("\n**************\
  		******************\n map number %d, epoch %d : \n\n",m,j);
      
      learn(map[m], in, j);
      printf("****************\nAfter standard learning\n");
      printf("learn:\n");
      quantization[m][j+1] = errorrate(map[m], in,NBITEREPOCH,  j+1);

      printf("test:\n");
      quantization_test[m][j+1] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);

      printf("learn:\n");
      distortion[m][j+1] = evaldistortion(map[m], in,NBITEREPOCH,  j+1);

      printf("test:\n");
      distortion_test[m][j+1] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
    } // end learn loop through all maps
    
    fault_tolerance_eco(j+1);
    fault_tolerance_gss(j+1);
  } // end loop over learn epoches
  
  FILE 	*	fp;
  
  fp = fopen ("learn_quantization.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      quantization[m][j]);
      
      
    }
  }
  fclose(fp);
  
  fp = fopen ("learn_distortion.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      distortion[m][j]);
    }
  }
  fclose(fp);
  
  fp = fopen ("test_quantization.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      quantization_test[m][j]);
    }
  }
  fclose(fp);

  fp = fopen ("test_distortion.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      distortion_test[m][j]);
    }
  }
  fclose(fp);


  for (m = 0; m < NBMAPS; m++) {
    freeMap(map[m]);
  }

  clock_t stop = clock();
  double elapsed = (double) (stop - start) / 1000.0;
  printf("Time elapsed in ms: %f\n", elapsed);
  exit(1);
}
