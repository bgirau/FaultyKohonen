
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

void generateTikzPrototypes(Kohonen ma,int m,int ep,int err) {
  int i,j,k;
  char *name;
  FILE *fp;
  
  gaussian_prototypes(ma,SIGMA_GAUSS);
  name=(char*)malloc(100*sizeof(char));
  sprintf(name,"prototypes_map%d_iter%d_err%d.txt",m,ep,err);
  fp = fopen (name, "a+");
  if (INS==2) {
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\node (n_%d_%d) at (%f,%f){};\n",i,j,ma.weights[i][j][0]*1.0/one,ma.weights[i][j][1]*1.0/one);
      }
    }
    fprintf(fp,"\\begin{scope}[on background layer]\n");
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size-1;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i,j+1);
      }
    }
    for (i=0;i<ma.size-1;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i+1,j);
      }
    }
    fprintf(fp,"\\end{scope}\n");
    fclose(fp);
    sprintf(name,"prototypesGSS_map%d_iter%d_err%d.txt",m,ep,err);
    fp = fopen (name, "a+");
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\node (n_%d_%d) at (%f,%f){};\n",i,j,ma.gss_weights[i][j][0]*1.0/one,ma.gss_weights[i][j][1]*1.0/one);
      }
    }
    fprintf(fp,"\\begin{scope}[on background layer]\n");
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size-1;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i,j+1);
      }
    }
    for (i=0;i<ma.size-1;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i+1,j);
      }
    }
    fprintf(fp,"\\end{scope}\n");
  } else {
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\node (n_%d_%d) at (%f,%f){};\n",i,j,ma.weights[i][j][0]*1.0/one-0.7*ma.weights[i][j][1]*1.0/one,ma.weights[i][j][2]*1.0/one-0.7*ma.weights[i][j][1]*1.0/one);
      }
    }
    fprintf(fp,"\\begin{scope}[on background layer]\n");
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size-1;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i,j+1);
      }
    }
    for (i=0;i<ma.size-1;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i+1,j);
      }
    }
    fprintf(fp,"\\end{scope}\n");
    fclose(fp);
    sprintf(name,"prototypesGSS_map%d_iter%d_err%d.txt",m,ep,err);
    fp = fopen (name, "a+");
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\node (n_%d_%d) at (%f,%f){};\n",i,j,ma.gss_weights[i][j][0]*1.0/one-0.7*ma.gss_weights[i][j][1]*1.0/one,ma.gss_weights[i][j][2]*1.0/one-0.7*ma.gss_weights[i][j][1]*1.0/one);
      }
    }
    fprintf(fp,"\\begin{scope}[on background layer]\n");
    for (i=0;i<ma.size;i++) {
      for (j=0;j<ma.size-1;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i,j+1);
      }
    }
    for (i=0;i<ma.size-1;i++) {
      for (j=0;j<ma.size;j++) {
	fprintf(fp, "\\draw (n_%d_%d) -- (n_%d_%d);\n",i,j,i+1,j);
      }
    }
    fprintf(fp,"\\end{scope}\n");
  }
  fclose(fp);
  free(name);
}

void study_gss(int ep) {
  int m,p;

  for (m = 0; m < NBMAPS; m++) {
    generateTikzPrototypes(map[m],m,ep,0);
    
    for (p = 1; p < MAXFAULTPERCENT; p++) {
      Kohonen map2    = copy(map[m]);
      faulty_weights(map2, p);
      generateTikzPrototypes(map2,m,ep,p);
    }

  } // end loop on map initializations
}

int main(){
  
  init_random(&rng);

  srand(time(NULL));
  clock_t 	start = clock();
  int  p,i,j,e,k,m;
  FILE *fp;

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
  
  int ** in = (int **) malloc_2darray(NBITEREPOCH, 2);
  int ** test = (int **) malloc_2darray(SIZE*SIZE*TEST_DENSITY, 2);
  
  for(i = 0; i < NBITEREPOCH; i++) {
    double *v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      in[i][k] = (int) ((1.0 * one) * v[k]);
    }
  }
  
  for(i = 0; i < SIZE*SIZE*TEST_DENSITY; i++) {
    double *v=DISTRIB(&rng);
    for(k = 0; k < INS; k++) {
      test[i][k] = (int) ((1.0 * one) * v[k]);
    }
  }
  char	*name=(char*)malloc(100*sizeof(char));
  sprintf(name,"test_data.txt");
  fp = fopen (name, "a+");
  for(i = 0; i < SIZE*SIZE*TEST_DENSITY; i++) {
    fprintf(fp, "\\node (i_%d) at (%f,%f){};\n",i,test[i][0]*1.0/one,test[i][1]*1.0/one);
  }
  fclose(fp);
  free(name);

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

  study_gss(0);

  for(j = 0; j < NBEPOCHLEARN; j++){
    // generate new random values
    for(i = 0; i < NBITEREPOCH; i++) {
      double *v=DISTRIB(&rng);
      for(k = 0; k < INS; k++) {
	in[i][k] = (int) ((1.0 * one) * v[k]);
      }
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
    
    study_gss(j+1);
  } // end loop over learn epoches
  
  
  fp = fopen ("learn_quantization_study.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      quantization[m][j]);
      
      
    }
  }
  fclose(fp);
  
  fp = fopen ("learn_distortion_study.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      distortion[m][j]);
    }
  }
  fclose(fp);
  
  fp = fopen ("test_quantization_study.txt", "w+");
  fprintf(fp, "Map_number;Epoch_number;Standard\n");
  for (m = 0; m < NBMAPS; m++){
    for (j = 0; j < NBEPOCHLEARN; j++){
      fprintf(fp, "%-d; %-d; %-f\n", m, j,
	      quantization_test[m][j]);
    }
  }
  fclose(fp);

  fp = fopen ("test_distortion_study.txt", "w+");
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
