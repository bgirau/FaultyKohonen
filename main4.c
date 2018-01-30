/* version spécifique à la préparation de l'article long suite à WSOM, basée sur main3_distrib.c :
   - pas de GSS, pas de NF
   - pas de génération des fichiers exhaustifs test_... et learn_...
   - simulation de fautes d'une version séquentielle
*/


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

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,SIZE*SIZE*TEST2_DENSITY,p);
	quantization2_th_test[p][e][m] = avg_quant_error(map_th[m], test2,SIZE*SIZE*TEST2_DENSITY,p);
	quantization2_FI_test[p][e][m] = avg_quant_error(map_FI[m], test2,SIZE*SIZE*TEST2_DENSITY,p);
	quantization2_NI_test[p][e][m] = avg_quant_error(map_NI[m], test2,SIZE*SIZE*TEST2_DENSITY,p);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,SIZE*SIZE*TEST2_DENSITY,p);
        quantization2_th_test_faulty[p][e][m] = avg_quant_error(map2_th, test2,SIZE*SIZE*TEST2_DENSITY,p);
        quantization2_FI_test_faulty[p][e][m] = avg_quant_error(map2_FI, test2,SIZE*SIZE*TEST2_DENSITY,p);
        quantization2_NI_test_faulty[p][e][m] = avg_quant_error(map2_NI, test2,SIZE*SIZE*TEST2_DENSITY,p);

        avg[p] += quantization2_test[p][e][m];
        avg_th[p] += quantization2_th_test[p][e][m];
        avg_FI[p] += quantization2_FI_test[p][e][m];
        avg_NI[p] += quantization2_NI_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];
        avg_th_faulty[p] += quantization2_th_test_faulty[p][e][m];
        avg_FI_faulty[p] += quantization2_FI_test_faulty[p][e][m];
        avg_NI_faulty[p] += quantization2_NI_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);
	distortion2_th_test[p][e][m] = distortion_measure(map_th[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);
	distortion2_FI_test[p][e][m] = distortion_measure(map_FI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);
	distortion2_NI_test[p][e][m] = distortion_measure(map_NI[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);
        distortion2_th_test_faulty[p][e][m] = distortion_measure(map2_th,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);
        distortion2_FI_test_faulty[p][e][m] = distortion_measure(map2_FI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);
        distortion2_NI_test_faulty[p][e][m] = distortion_measure(map2_NI,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);

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

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,SIZE*SIZE*TEST2_DENSITY,p);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,SIZE*SIZE*TEST2_DENSITY,p);

        avg[p] += quantization2_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,SIZE*SIZE*TEST2_DENSITY,SIGMA_GAUSS,p);

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

  fp = fopen ("statistics_quantization.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg_p;Standard_ratio_avg_0\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", ep, p, 
	    avg_faulty[p], stddev_faulty[p], avg_faulty[p]/avg[p], avg_faulty[p]/avg[0]);
  }
  fclose(fp);

  fp = fopen ("statistics_distortion.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avgdist;Standard_std;Standard_ratio_avg_p;Standard_ratio_avg_0\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", ep, p, 
	    avgdist_faulty[p], stddevdist_faulty[p], avgdist_faulty[p]/avgdist[p], avgdist_faulty[p]/avgdist[0]);
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


  Kohonen *mapinit;

  /*  
      calculate distortion for each map for each epoche
      NBEPOCHLEARN+1 - # of distortions measure + measure before learning
  */
  double  ** quantization;
  double  ** quantization_th;
  double  ** quantization_FI;
  double  ** quantization_NI;
  
  double  ** quantization_test;
  double  ** quantization_th_test;
  double  ** quantization_FI_test;
  double  ** quantization_NI_test;

  double  ** distortion;
  double  ** distortion_th;
  double  ** distortion_FI;
  double  ** distortion_NI;

  double  ** distortion_test;
  double  ** distortion_th_test;
  double  ** distortion_FI_test;
  double  ** distortion_NI_test;

  map = malloc(NBMAPS*sizeof(Kohonen));
  mapinit  = malloc(NBMAPS*sizeof(Kohonen));
  quantization     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  quantization_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  distortion     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  distortion_test     = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);

  if (WITH_TECHS) {
    map_th   = malloc(NBMAPS*sizeof(Kohonen));
    map_FI   = malloc(NBMAPS*sizeof(Kohonen));
    map_NI   = malloc(NBMAPS*sizeof(Kohonen));
    
    quantization_th  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    quantization_FI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    quantization_NI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    
    quantization_th_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    quantization_FI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    quantization_NI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    
    distortion_th  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    distortion_FI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    distortion_NI  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    
    distortion_th_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    distortion_FI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
    distortion_NI_test  = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN+1);
  }
  
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
    if (WITH_TECHS) {
      map_th[m]	=	copy(mapinit[m]);
      map_FI[m]	=	copy(mapinit[m]);
      map_NI[m]	=	copy(mapinit[m]);
    }
    
    printf("****************\nBefore learning\n");
    printf("learn:\n");
    quantization[m][0] = errorrate(map[m], in,NBITEREPOCH, 0);
    if (WITH_TECHS) {
	quantization_th[m][0] = quantization[m][0];
	quantization_FI[m][0] = quantization[m][0];
	quantization_NI[m][0] = quantization[m][0];
    }

    printf("test:\n");
    quantization_test[m][0] = errorrate(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
    if (WITH_TECHS) {
	quantization_th_test[m][0] = quantization_test[m][0];
	quantization_FI_test[m][0] = quantization_test[m][0];
	quantization_NI_test[m][0] = quantization_test[m][0];
    }
    
    printf("learn:\n");
    distortion[m][0] = evaldistortion(map[m], in,NBITEREPOCH, 0);
    if (WITH_TECHS) {
	distortion_th[m][0] = distortion[m][0];
	distortion_FI[m][0] = distortion[m][0];
	distortion_NI[m][0] = distortion[m][0];
    }
    
    printf("test:\n");
    distortion_test[m][0] = evaldistortion(map[m], test,SIZE*SIZE*TEST_DENSITY, 0);
    if (WITH_TECHS) {
	distortion_th_test[m][0] = distortion_test[m][0];
	distortion_FI_test[m][0] = distortion_test[m][0];
	distortion_NI_test[m][0] = distortion_test[m][0];
    }

  }

  if (WITH_TECHS)
    fault_tolerance(0);
  else
    fault_tolerance_eco(0);

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

      if (WITH_TECHS) {  	
	learn_threshold(map_th[m], in, j);
	printf("****************\nAfter thresholded learning\n");
	printf("learn:\n");
	quantization_th[m][j+1] = errorrate(map_th[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	quantization_th_test[m][j+1] = errorrate(map_th[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
	
	printf("learn:\n");
	distortion_th[m][j+1] = evaldistortion(map_th[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	distortion_th_test[m][j+1] = evaldistortion(map_th[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
	
    	
	learn_FI(map_FI[m], in, j);
	printf("****************\nAfter fault injection learning\n");
	printf("learn:\n");
	quantization_FI[m][j+1] = errorrate(map_FI[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	quantization_FI_test[m][j+1] = errorrate(map_FI[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
	
	printf("learn:\n");
	distortion_FI[m][j+1] = evaldistortion(map_FI[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	distortion_FI_test[m][j+1] = evaldistortion(map_FI[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
	
	
	learn_NI(map_NI[m], in, j);
	printf("****************\nAfter noise injection learning\n");
	printf("learn:\n");
	quantization_NI[m][j+1] = errorrate(map_NI[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	quantization_NI_test[m][j+1] = errorrate(map_NI[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
	
	printf("learn:\n");
	distortion_NI[m][j+1] = evaldistortion(map_NI[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	distortion_NI_test[m][j+1] = evaldistortion(map_NI[m], test,SIZE*SIZE*TEST_DENSITY, j+1);
      }
    } // end learn loop through all maps

    if (WITH_TECHS)
      fault_tolerance(j+1);
    else
      fault_tolerance_eco(j+1);
  } // end loop over learn epoches
  
  FILE 	*	fp;
  
  fp = fopen ("learn_quantization.txt", "w+");
  if (WITH_TECHS)
    fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS)
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", m, j,
		quantization[m][j], quantization_th[m][j], quantization_FI[m][j], quantization_NI[m][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", m, j,
		quantization[m][j]);
    }
  }
  fclose(fp);
  
  fp = fopen ("learn_distortion.txt", "w+");
  if (WITH_TECHS)
    fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS)
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", m, j,
		distortion[m][j], distortion_th[m][j], distortion_FI[m][j], distortion_NI[m][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", m, j,
		distortion[m][j]);
    }
  }
  fclose(fp);
  
  fp = fopen ("test_quantization.txt", "w+");
  if (WITH_TECHS)
    fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS)
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", m, j,
		quantization_test[m][j], quantization_th_test[m][j], quantization_FI_test[m][j], quantization_NI_test[m][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", m, j,
		quantization_test[m][j]);
    }
  }
  fclose(fp);

  fp = fopen ("test_distortion.txt", "w+");
  if (WITH_TECHS)
    fprintf(fp, "Map_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS) 
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", m, j,
		distortion_test[m][j], distortion_th_test[m][j], distortion_FI_test[m][j], distortion_NI_test[m][j]);
      else
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
