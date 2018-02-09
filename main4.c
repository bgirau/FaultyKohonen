/* version spécifique à la préparation de l'article long suite à WSOM, basée sur main3_distrib.c :
   - pas de GSS, pas de NF
   - pas de génération des fichiers exhaustifs test_... et learn_...
   - simulation de fautes d'une version séquentielle
*/


#include <string.h>

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

char* changeDir() {
  char *filename;
  char arg[50];
  filename=(char*)malloc(200*sizeof(char));
  char cmd[200];
  strcpy(filename,"Results_NoGSS_NoDNF_");
  strcat(filename,DISTRIBNAME);
  strcat(filename,"_");
  sprintf(arg,"%dx%dneurons_",SIZE,SIZE);
  strcat(filename,arg);
  if (OPTIMIZED_WEIGHTS) strcat(filename,"OWS_");
  else if (INDIVIDUAL_WEIGHTS) strcat(filename,"IWS_"); else strcat(filename,"SWS_");
  if (SEQUENTIAL) strcat(filename,"SEQ_"); else strcat(filename,"PAR_");
  sprintf(arg,"%dmaps_",NBMAPS);
  strcat(filename,arg);
  sprintf(arg,"%dinits_",NBMAPINITS);
  strcat(filename,arg);
  sprintf(arg,"%dx%diter_",NBEPOCHLEARN,NBITEREPOCH);
  strcat(filename,arg);
  sprintf(arg,"dens%d_",TEST_DENSITY);
  strcat(filename,arg);
  if (WITH_TECHS) strcat(filename,"alltechs"); else strcat(filename,"notech");
  strcpy(cmd,"mkdir ");
  strcat(cmd,filename);
  system(cmd);
  strcpy(cmd,"cp pre_def.h ");
  strcat(cmd,filename);
  system(cmd);
  return filename;
}

void fault_tolerance(int ep,char *path) {
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

  int density2=1;
  for (i=0;i<INS;i++) density2 *= TEST2_DENSITY;
  int ** test2 = (int **) malloc_2darray(density2, INS);
  FILE 	*	fp;
  

  for(i = 0; i < density2; i++) {
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

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,density2,p);
	quantization2_th_test[p][e][m] = avg_quant_error(map_th[m], test2,density2,p);
	quantization2_FI_test[p][e][m] = avg_quant_error(map_FI[m], test2,density2,p);
	quantization2_NI_test[p][e][m] = avg_quant_error(map_NI[m], test2,density2,p);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,density2,p);
        quantization2_th_test_faulty[p][e][m] = avg_quant_error(map2_th, test2,density2,p);
        quantization2_FI_test_faulty[p][e][m] = avg_quant_error(map2_FI, test2,density2,p);
        quantization2_NI_test_faulty[p][e][m] = avg_quant_error(map2_NI, test2,density2,p);

        avg[p] += quantization2_test[p][e][m];
        avg_th[p] += quantization2_th_test[p][e][m];
        avg_FI[p] += quantization2_FI_test[p][e][m];
        avg_NI[p] += quantization2_NI_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];
        avg_th_faulty[p] += quantization2_th_test_faulty[p][e][m];
        avg_FI_faulty[p] += quantization2_FI_test_faulty[p][e][m];
        avg_NI_faulty[p] += quantization2_NI_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,density2,SIGMA_GAUSS,p);
	distortion2_th_test[p][e][m] = distortion_measure(map_th[m],test2,density2,SIGMA_GAUSS,p);
	distortion2_FI_test[p][e][m] = distortion_measure(map_FI[m],test2,density2,SIGMA_GAUSS,p);
	distortion2_NI_test[p][e][m] = distortion_measure(map_NI[m],test2,density2,SIGMA_GAUSS,p);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,density2,SIGMA_GAUSS,p);
        distortion2_th_test_faulty[p][e][m] = distortion_measure(map2_th,test2,density2,SIGMA_GAUSS,p);
        distortion2_FI_test_faulty[p][e][m] = distortion_measure(map2_FI,test2,density2,SIGMA_GAUSS,p);
        distortion2_NI_test_faulty[p][e][m] = distortion_measure(map2_NI,test2,density2,SIGMA_GAUSS,p);

        avgdist[p] += distortion2_test[p][e][m];
        avgdist_th[p] += distortion2_th_test[p][e][m];
        avgdist_FI[p] += distortion2_FI_test[p][e][m];
        avgdist_NI[p] += distortion2_NI_test[p][e][m];

        avgdist_faulty[p] += distortion2_test_faulty[p][e][m];
        avgdist_th_faulty[p] += distortion2_th_test_faulty[p][e][m];
        avgdist_FI_faulty[p] += distortion2_FI_test_faulty[p][e][m];
        avgdist_NI_faulty[p] += distortion2_NI_test_faulty[p][e][m];

	freeMap(map2);
	freeMap(map2_th);
	freeMap(map2_FI);
	freeMap(map2_NI);

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
  char filename[300];
  strcpy(filename,path);
  strcat(filename,"/statistics_quantization.txt");
  fp = fopen (filename, "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg;Threshold_avg;Threshold_std;Threshold_ratio_avg;FaultInjection_avg;FaultInjection_std;FaultInjection_ratio_avg;NoiseInjection_avg;NoiseInjection_std;NoiseInjection_ratio_avg\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", ep, p, 
	    avg_faulty[p], stddev_faulty[p], avg_faulty[p]/avg[p], avg_th_faulty[p], stddev_th_faulty[p], avg_th_faulty[p]/avg_th[p], avg_FI_faulty[p], stddev_FI_faulty[p], avg_FI_faulty[p]/avg_FI[p], 
	    avg_NI_faulty[p], stddev_NI_faulty[p], avg_NI_faulty[p]/avg_NI[p]);
  }
  fclose(fp);

  strcpy(filename,path);
  strcat(filename,"/statistics_distortion.txt");
  fp = fopen (filename, "a+");
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

  free_2darray(test2,density2);
}

void fault_tolerance_eco(int ep,char *path) {
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

  int density2=1;
  for (i=0;i<INS;i++) density2 *= TEST2_DENSITY;
  int ** test2 = (int **) malloc_2darray(density2, INS);
  FILE 	*	fp;
  

  for(i = 0; i < density2; i++) {
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

	quantization2_test[p][e][m] 	 = avg_quant_error(map[m], test2,density2,p);

        quantization2_test_faulty[p][e][m]    = avg_quant_error(map2, test2,density2,p);

        avg[p] += quantization2_test[p][e][m];

        avg_faulty[p] += quantization2_test_faulty[p][e][m];

	distortion2_test[p][e][m] 	 = distortion_measure(map[m],test2,density2,SIGMA_GAUSS,p);

        distortion2_test_faulty[p][e][m]    = distortion_measure(map2,test2,density2,SIGMA_GAUSS,p);

        avgdist[p] += distortion2_test[p][e][m];

        avgdist_faulty[p] += distortion2_test_faulty[p][e][m];

	freeMap(map2);

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

  char filename[300];
  strcpy(filename,path);
  strcat(filename,"statistics_quantization.txt");
  fp = fopen ("/statistics_quantization.txt", "a+");
  fprintf(fp, "Epoch;Percentage_faults;Standard_avg;Standard_std;Standard_ratio_avg_p;Standard_ratio_avg_0\n");
  for (p = 0; p < MAXFAULTPERCENT; p++) {
    fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", ep, p, 
	    avg_faulty[p], stddev_faulty[p], avg_faulty[p]/avg[p], avg_faulty[p]/avg[0]);
  }
  fclose(fp);

  strcpy(filename,path);
  strcat(filename,"/statistics_distortion.txt");
  fp = fopen (filename, "a+");
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

  free_2darray(test2,density2);

}

int main(){

  char *path=changeDir();
  
  init_random(&rng);

  srand(time(NULL));
  clock_t 	start = clock();
  int  p,i,j,e,k,m,mi,pat;


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
  mapinit  = malloc(NBMAPINITS*sizeof(Kohonen));
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
  
  for (i = 0; i < NBMAPINITS; i++) mapinit[i] = init();
  
  int ** in = (int **) malloc_2darray(NBITEREPOCH, INS);
  int density=1;
  for (i=0;i<INS;i++) density *= TEST_DENSITY;
  int ** test = (int **) malloc_2darray(density, INS);
  
  for(i = 0; i < NBITEREPOCH; i++) {
    double *v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      in[i][k] = (int) ((1.0 * one) * v[k]);
    }
    free(v);
  }
  
  for(i = 0; i < density; i++) {
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
    map[m]		=	copy(mapinit[m%NBMAPINITS]);
    if (WITH_TECHS) {
      map_th[m]	=	copy(mapinit[m%NBMAPINITS]);
      map_FI[m]	=	copy(mapinit[m%NBMAPINITS]);
      map_NI[m]	=	copy(mapinit[m%NBMAPINITS]);
    }
  }
  for (mi = 0; mi < NBMAPINITS; mi++) {
    printf("****************\nBefore learning\n");
    printf("learn:\n");
    quantization[mi][0] = errorrate(map[mi], in,NBITEREPOCH, 0);
    if (WITH_TECHS) {
	quantization_th[mi][0] = quantization[mi][0];
	quantization_FI[mi][0] = quantization[mi][0];
	quantization_NI[mi][0] = quantization[mi][0];
    }

    printf("test:\n");
    quantization_test[mi][0] = errorrate(map[mi], test,density, 0);
    if (WITH_TECHS) {
	quantization_th_test[mi][0] = quantization_test[mi][0];
	quantization_FI_test[mi][0] = quantization_test[mi][0];
	quantization_NI_test[mi][0] = quantization_test[mi][0];
    }
    
    printf("learn:\n");
    distortion[mi][0] = evaldistortion(map[mi], in,NBITEREPOCH, 0);
    if (WITH_TECHS) {
	distortion_th[mi][0] = distortion[mi][0];
	distortion_FI[mi][0] = distortion[mi][0];
	distortion_NI[mi][0] = distortion[mi][0];
    }
    
    printf("test:\n");
    distortion_test[mi][0] = evaldistortion(map[mi], test,density, 0);
    if (WITH_TECHS) {
	distortion_th_test[mi][0] = distortion_test[mi][0];
	distortion_FI_test[mi][0] = distortion_test[mi][0];
	distortion_NI_test[mi][0] = distortion_test[mi][0];
    }

  }

  if (WITH_TECHS)
    fault_tolerance(0,path);
  else
    fault_tolerance_eco(0,path);

  for(j = 0; j < NBEPOCHLEARN; j++){
    for (mi = 0; mi < NBMAPINITS; mi++) {
      quantization[mi][j+1] = 0;
      quantization_test[mi][j+1] = 0;
      distortion[mi][j+1] = 0;
      distortion_test[mi][j+1] = 0;
      if (WITH_TECHS) {  	
	quantization_th[mi][j+1] = 0;
	quantization_th_test[mi][j+1] = 0;
	distortion_th[mi][j+1] = 0;
	distortion_th_test[mi][j+1] = 0;
	quantization_FI[mi][j+1] = 0;
	quantization_FI_test[mi][j+1] = 0;
	distortion_FI[mi][j+1] = 0;
	distortion_FI_test[mi][j+1] = 0;
	quantization_NI[mi][j+1] = 0;
	quantization_NI_test[mi][j+1] = 0;
	distortion_NI[mi][j+1] = 0;
	distortion_NI_test[mi][j+1] = 0;
      }	
    }
    for (pat=0;pat<NBMAPS/NBMAPINITS;pat++) {
      // generate new random values
      for(i = 0; i < NBITEREPOCH; i++) {
	double *v=DISTRIB(&rng);
	for(k = 0; k < INS; k++) {
	  in[i][k] = (int) ((1.0 * one) * v[k]);
	}
	free(v);
      }
      
      for (mi = 0; mi < NBMAPINITS; mi++) {
	m=mi+pat*NBMAPINITS;
	printf("\n**************\
  		******************\n map number %d, epoch %d : \n\n",m,j);
	
	learn(map[m], in, j);
	printf("****************\nAfter standard learning\n");
	printf("learn:\n");

	quantization[mi][j+1] += errorrate(map[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	quantization_test[mi][j+1] += errorrate(map[m], test,density, j+1);
	
	printf("learn:\n");
	distortion[mi][j+1] += evaldistortion(map[m], in,NBITEREPOCH,  j+1);
	
	printf("test:\n");
	distortion_test[mi][j+1] += evaldistortion(map[m], test,density, j+1);
	
	if (WITH_TECHS) {  	
	  learn_threshold(map_th[m], in, j);
	  printf("****************\nAfter thresholded learning\n");
	  printf("learn:\n");
	  quantization_th[mi][j+1] += errorrate(map_th[m], in,NBITEREPOCH,  j+1);
	  
	  printf("test:\n");
	  quantization_th_test[mi][j+1] += errorrate(map_th[m], test,density, j+1);
	  
	  printf("learn:\n");
	  distortion_th[m][j+1] += evaldistortion(map_th[m], in,NBITEREPOCH,  j+1);
	  
	  printf("test:\n");
	  distortion_th_test[mi][j+1] += evaldistortion(map_th[m], test,density, j+1);
	  
	  
	  learn_FI(map_FI[m], in, j);
	  printf("****************\nAfter fault injection learning\n");
	  printf("learn:\n");
	  quantization_FI[mi][j+1] += errorrate(map_FI[m], in,NBITEREPOCH,  j+1);
	  
	  printf("test:\n");
	  quantization_FI_test[mi][j+1] += errorrate(map_FI[m], test,density, j+1);
	  
	  printf("learn:\n");
	  distortion_FI[mi][j+1] += evaldistortion(map_FI[m], in,NBITEREPOCH,  j+1);
	  
	  printf("test:\n");
	  distortion_FI_test[mi][j+1] += evaldistortion(map_FI[m], test,density, j+1);
	  
	  
	  learn_NI(map_NI[m], in, j);
	  printf("****************\nAfter noise injection learning\n");
	  printf("learn:\n");
	  quantization_NI[mi][j+1] += errorrate(map_NI[m], in,NBITEREPOCH,  j+1);
	  
	  printf("test:\n");
	  quantization_NI_test[mi][j+1] += errorrate(map_NI[m], test,density, j+1);
	  
	  printf("learn:\n");
	  distortion_NI[mi][j+1] += evaldistortion(map_NI[m], in,NBITEREPOCH,  j+1);
	  
	  printf("test:\n");
	  distortion_NI_test[mi][j+1] += evaldistortion(map_NI[m], test,density, j+1);
	}
      } // end loop through mapinits
    } // end learn loop with different patterns
    for (mi = 0; mi < NBMAPINITS; mi++) {
      int divise=NBMAPS/NBMAPINITS;
      quantization[mi][j+1] /= divise;
      quantization_test[mi][j+1] /= divise;
      distortion[mi][j+1] /= divise;
      distortion_test[mi][j+1] /= divise;
      if (WITH_TECHS) {  	
	quantization_th[mi][j+1] /= divise;
	quantization_th_test[mi][j+1] /= divise;
	distortion_th[m][j+1] /= divise;
	distortion_th_test[mi][j+1] /= divise;
	quantization_FI[mi][j+1] /= divise;
	quantization_FI_test[mi][j+1] /= divise;
	distortion_FI[mi][j+1] /= divise;
	distortion_FI_test[mi][j+1] /= divise;
	quantization_NI[mi][j+1] /= divise;
	quantization_NI_test[mi][j+1] /= divise;
	distortion_NI[mi][j+1] /= divise;
	distortion_NI_test[mi][j+1] /= divise;
      }	
    }
    
    if (WITH_TECHS)
      fault_tolerance(j+1,path);
    else
      fault_tolerance_eco(j+1,path);
  } // end loop over learn epoches
  
  FILE 	*	fp;

  char filename[300];
  strcpy(filename,path);
  strcat(filename,"/learn_quantization.txt");

  fp = fopen (filename, "w+");
  if (WITH_TECHS)
    fprintf(fp, "MapInit_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "MapInit_number;Epoch_number;Standard\n");
  for (mi = 0; mi < NBMAPINITS; mi++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS)
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", mi, j,
		quantization[mi][j], quantization_th[mi][j], quantization_FI[mi][j], quantization_NI[mi][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", mi, j,
		quantization[mi][j]);
    }
  }
  fclose(fp);
  
  strcpy(filename,path);
  strcat(filename,"/learn_distortion.txt");

  fp = fopen (filename, "w+");
  if (WITH_TECHS)
    fprintf(fp, "MapInit_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "MapInit_number;Epoch_number;Standard\n");
  for (mi = 0; mi < NBMAPINITS; mi++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS)
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", mi, j,
		distortion[mi][j], distortion_th[mi][j], distortion_FI[mi][j], distortion_NI[mi][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", mi, j,
		distortion[mi][j]);
    }
  }
  fclose(fp);
  
  strcpy(filename,path);
  strcat(filename,"/test_quantization.txt");

  fp = fopen (filename, "w+");
  if (WITH_TECHS)
    fprintf(fp, "MapInit_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "MapInit_number;Epoch_number;Standard\n");
  for (mi = 0; mi < NBMAPINITS; mi++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS)
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", mi, j,
		quantization_test[mi][j], quantization_th_test[mi][j], quantization_FI_test[mi][j], quantization_NI_test[mi][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", mi, j,
		quantization_test[mi][j]);
    }
  }
  fclose(fp);

  strcpy(filename,path);
  strcat(filename,"/test_distortion.txt");

  fp = fopen (filename, "w+");
  if (WITH_TECHS)
    fprintf(fp, "MapInit_number;Epoch_number;Standard;Threshold;FaultInjection;NoiseInjetion\n");
  else
    fprintf(fp, "MapInit_number;Epoch_number;Standard\n");
  for (mi = 0; mi < NBMAPINITS; mi++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      if (WITH_TECHS) 
	fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", mi, j,
		distortion_test[mi][j], distortion_th_test[mi][j], distortion_FI_test[mi][j], distortion_NI_test[mi][j]);
      else
	fprintf(fp, "%-d; %-d; %-f\n", mi, j,
		distortion_test[mi][j]);
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
