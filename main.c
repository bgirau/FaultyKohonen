
#include "func_def.h"
#include "malloc.h"
#include "custom_rand.h"
#include "threads.h"


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

  Evaluations eval_train, eval_train_th, eval_train_FI, 
              eval_train_NI, eval_train_NF;

  Evaluations eval_valid, eval_valid_th, eval_valid_FI, 
              eval_valid_NI, eval_valid_NF;

  init_evaluations(&eval_train); init_evaluations(&eval_valid);
  init_evaluations(&eval_train_th); init_evaluations(&eval_valid_th);
  init_evaluations(&eval_train_FI); init_evaluations(&eval_valid_FI);
  init_evaluations(&eval_train_NI); init_evaluations(&eval_valid_NI);
  init_evaluations(&eval_train_NF); init_evaluations(&eval_valid_NF);

  Evaluations_faulty eval_test, eval_test_th, eval_test_FI, eval_test_NI, eval_test_NF;

  init_evaluations_faulty(&eval_test);
  init_evaluations_faulty(&eval_test_th);
  init_evaluations_faulty(&eval_test_FI);
  init_evaluations_faulty(&eval_test_NI);
  init_evaluations_faulty(&eval_test_NF);

  Statistics stat, stat_th, stat_FI, stat_NI, stat_NF;

  init_statistics(&stat);
  init_statistics(&stat_th);
  init_statistics(&stat_FI);
  init_statistics(&stat_NI);
  init_statistics(&stat_NF);

  int *** in = malloc_3darray(NBEPOCHLEARN,NBITEREPOCH, 2);
  int ** test = malloc_2darray(SIZE * SIZE * TEST_DENSITY, 2);
  int ** test2 = malloc_2darray(SIZE*SIZE*TEST2_DENSITY, 2);
  
  for(j = 0; j < NBEPOCHLEARN; j++){
    // generate new random values
    for(i = 0; i < NBITEREPOCH; i++) {
      for(k = 0; k < INS; k++) {
        in[j][i][k] = (int) ((1.0 * one) * uniform_dataset(&rng)[k]);
      }
    }
  }

  for(i = 0; i < SIZE*SIZE*TEST_DENSITY; i++) {
    for(k = 0; k < INS; k++) {
      test[i][k] = (int) ((1.0 * one) * uniform_dataset(&rng)[k]);
    }
  }

  for(i = 0; i < SIZE*SIZE*TEST2_DENSITY; i++) {
    for(k = 0; k < INS; k++) {
      test2[i][k] = (int) ((1.0 * one) * uniform_dataset(&rng)[k]);
    }
  }

  for (i = 0; i < NBMAPS; i++) mapinit[i] = init();

  for (m = 0; m < NBMAPS; m++) {
    map[m]    = copy(mapinit[m]);
    map_th[m] = copy(mapinit[m]);
    map_FI[m] = copy(mapinit[m]);
    map_NI[m] = copy(mapinit[m]);
    map_NF[m] = copy(mapinit[m]);
  }

  pthread_t threads[NUM_THREADS];
  pthread_t mon_thrd;

  Input_args standard;

  standard.name = "Standard";
  standard.map  = map;
  standard.train_set = in;
  standard.valid_set = test;
  standard.test_set = test2;
  standard.qlt_train = &eval_train;
  standard.qlt_valid = &eval_valid;
  standard.qlt_test = &eval_test;
  standard.stat = &stat;
  standard.status = "Init";

  Input_args th;

  th.name = "Threshold";
  th.map  = map_th;
  th.train_set = in;
  th.valid_set = test;
  th.test_set = test2;
  th.qlt_train = &eval_train_th;
  th.qlt_valid = &eval_valid_th;
  th.qlt_test = &eval_test_th;
  th.stat = &stat_th;
  th.status = "Init";

  Input_args FI;

  FI.name = "FaultInj";
  FI.map  = map_FI;
  FI.train_set = in;
  FI.valid_set = test;
  FI.test_set = test2;
  FI.qlt_train = &eval_train_FI;
  FI.qlt_valid = &eval_valid_FI;
  FI.qlt_test = &eval_test_FI;
  FI.stat = &stat_FI;
  FI.status = "Init";

  Input_args NI;

  NI.name = "NoiseInj";
  NI.map  = map_NI;
  NI.train_set = in;
  NI.valid_set = test;
  NI.test_set = test2;
  NI.qlt_train = &eval_train_NI;
  NI.qlt_valid = &eval_valid_NI;
  NI.qlt_test = &eval_test_NI;
  NI.stat = &stat_NI;
  NI.status = "Init";

  Input_args NF;

  NF.name = "NeuralField";
  NF.map  = map_NF;
  NF.train_set = in;
  NF.valid_set = test;
  NF.test_set = test2;
  NF.qlt_train = &eval_train_NF;
  NF.qlt_valid = &eval_valid_NF;
  NF.qlt_test = &eval_test_NF;
  NF.stat = &stat_NF;
  NF.status = "Init";

  Monitor   mon;
  mon.th1 = &standard.status;
  mon.th2 = &th.status;
  mon.th3 = &FI.status;
  mon.th4 = &NI.status;
  mon.th5 = &NF.status;

  if (pthread_create(&threads[0], NULL, &learning_thread, (void *)&standard) != 0){
    printf("Can not create thread\n");
    exit(1);
  }
  if (pthread_create(&threads[1], NULL, &learning_thread, (void *)&th) != 0){
    printf("Can not create thread\n");
    exit(1);
  }
  if (pthread_create(&threads[2], NULL, &learning_thread, (void *)&FI) != 0){
    printf("Can not create thread\n");
    exit(1);
  }
  if (pthread_create(&threads[3], NULL, &learning_thread, (void *)&NI) != 0){
    printf("Can not create thread\n");
    exit(1);  
  }
  if (pthread_create(&threads[4], NULL, &learning_thread, (void *)&NF) != 0){
    printf("Can not create thread\n");
    exit(1);
  }
  if (pthread_create(&mon_thrd, NULL, &monitor_thread, (void *)&mon) != 0){
    printf("Can not create thread\n");
    exit(1);
  }

  int thd;
  for(thd = 0; thd < NUM_THREADS; thd++){
    pthread_join(threads[thd], NULL);
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
  printf("Time elapsed in ms: %f\n", elapsed);
}
