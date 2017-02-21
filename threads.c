
#include "threads.h"
#include "malloc.h"
#include "string.h"
#include "unistd.h"

void init_evaluations(Evaluations *ev) {
    ev->distortion = malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
    ev->quantization = malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
    ev->distortion_gss = malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
    ev->quantization_gss = malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
}

void init_evaluations_faulty(Evaluations_faulty *ev) {
    ev->distortion = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->distortion_gss = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization_gss = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->distortion_faulty = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization_faulty = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->distortion_gss_faulty = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization_gss_faulty = malloc_4darray_f(NBEPOCHLEARN+1, MAXFAULTPERCENT, nb_experiments, NBMAPS);
}

void init_statistics(Statistics *st) {
    st->avg = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddev = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avg_gss = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddev_gss = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avg_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddev_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avg_gss_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddev_gss_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avgdist = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddevdist = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avgdist_gss = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddevdist_gss = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avgdist_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddevdist_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->avgdist_gss_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
    st->stddevdist_gss_faulty = calloc_2darray(NBEPOCHLEARN+1, MAXFAULTPERCENT);
}

void *monitor_thread(void *args) {
    Monitor *mon = args;
    clock_t start = clock();
    clock_t stop;
    while (1) {
        printf("\033[H\033[J");
        printf("Thread 1 status: %s\t %-.3f%%\n", *mon->th1, *mon->th1_ready );
        printf("Thread 2 status: %s\t %-.3f%%\n", *mon->th2, *mon->th2_ready );
        printf("Thread 3 status: %s\t %-.3f%%\n", *mon->th3, *mon->th3_ready );
        printf("Thread 4 status: %s\t %-.3f%%\n", *mon->th4, *mon->th4_ready );
        stop = clock();
        printf("Time: %-f\n", (double) (stop-start)/1000.0);
//        printf("Thread 5 status: %s\t %-.3f%%\n", *mon->th5, *mon->th5_ready );
        sleep(2);
    }
    exit(1);
}

void *learning_thread(void *args) {
    Input_args *arg = args;
    int p, e, m, j, i;
    i = 0;
    clock_t start = clock();

    arg->ready = 0.0;
    double tt = nb_experiments * NBMAPS;
    char *temp_str = "Learning: ";
    char *tmp;
    tmp = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(tmp, temp_str);
    strcat(tmp, arg->name);
    arg->status = tmp;
    Kohonen map2;

    for ( m = 0; m < NBMAPS; m++ ) {
        // printf("****************\nBefore learning\n");
        arg->qlt_train->distortion[ m ][ 0 ] = distortion_measure(arg->map[ m ], arg->train_set[ 0 ], NBITEREPOCH,
                                                                  SIGMA_GAUSS);
//        arg->qlt_train->distortion_gss[ m ][ 0 ] = distortion_measure_GSS(arg->map[ m ], arg->train_set[ 0 ],
//                                                                          NBITEREPOCH, SIGMA_GAUSS);

        arg->qlt_train->quantization[ m ][ 0 ] = avg_quant_error(arg->map[ m ], arg->train_set[ 0 ], NBITEREPOCH);
//        arg->qlt_train->quantization_gss[ m ][ 0 ] = avg_quant_error_GSS(arg->map[ m ], arg->train_set[ 0 ],
//                                                                         NBITEREPOCH);

        arg->qlt_valid->distortion[ m ][ 0 ] = distortion_measure(arg->map[ m ], arg->valid_set, TEST_DENSITY,
                                                                  SIGMA_GAUSS);
//        arg->qlt_valid->distortion_gss[ m ][ 0 ] = distortion_measure_GSS(arg->map[ m ], arg->valid_set, TEST_DENSITY,
//                                                                          SIGMA_GAUSS);

        arg->qlt_valid->quantization[ m ][ 0 ] = avg_quant_error(arg->map[ m ], arg->valid_set, TEST_DENSITY);
//        arg->qlt_valid->quantization_gss[ m ][ 0 ] = avg_quant_error_GSS(arg->map[ m ], arg->valid_set, TEST_DENSITY);

        for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
            for ( e = 0; e < nb_experiments; e++ ) {
                map2 = copy(arg->map[ m ]);
                faulty_weights(map2, p);
                arg->qlt_test->quantization[ 0 ][ p ][ e ][ m ] = avg_quant_error(arg->map[ m ], arg->test_set, TEST2_DENSITY);
//                arg->qlt_test->quantization_gss[ 0 ][ p ][ e ][ m ] = avg_quant_error_GSS(arg->map[ m ], arg->test_set,
//                                                                                     TEST2_DENSITY);
                arg->qlt_test->quantization_faulty[ 0 ][ p ][ e ][ m ] = avg_quant_error(map2, arg->test_set, TEST2_DENSITY);
//                arg->qlt_test->quantization_gss_faulty[ 0 ][ p ][ e ][ m ] = avg_quant_error_GSS(map2, arg->test_set,
//                                                                                           TEST2_DENSITY);
                arg->qlt_test->distortion[ 0 ][ p ][ e ][ m ] = distortion_measure(arg->map[ m ], arg->test_set, TEST2_DENSITY,
                                                                                   SIGMA_GAUSS);
//                arg->qlt_test->distortion_gss[ 0 ][ p ][ e ][ m ] = distortion_measure_GSS(arg->map[ m ], arg->test_set,
//                                                                                           TEST2_DENSITY, SIGMA_GAUSS);
                arg->qlt_test->distortion_faulty[ 0 ][ p ][ e ][ m ] = distortion_measure(map2, arg->test_set, TEST2_DENSITY,
                                                                                          SIGMA_GAUSS);
//                arg->qlt_test->distortion_gss_faulty[0][ p ][ e ][ m ] = distortion_measure_GSS(map2, arg->test_set,
//                                                                                                  TEST2_DENSITY, SIGMA_GAUSS);
                freeMap(map2);
            }
        }

        for ( j = 0; j < NBEPOCHLEARN; j++ ) {
            if (strcmp(arg->name, "Standard") == 0) {
                learn(arg->map[ m ], arg->train_set[ j ], j);
            } else if (strcmp(arg->name, "Threshold") == 0) {
                learn_threshold(arg->map[ m ], arg->train_set[ j ], j);
            } else if (strcmp(arg->name, "FaultInj") == 0) {
                learn_FI(arg->map[ m ], arg->train_set[ j ], j);
            } else if (strcmp(arg->name, "NoiseInj") == 0) {
                learn_NI(arg->map[ m ], arg->train_set[ j ], j);
            } else if (strcmp(arg->name, "NeuralField") == 0) {
                learn_NF(arg->map[ m ], arg->train_set[ j ], j);
            } else {
                printf("Something goes wrong in threads!\n");
                exit(1);
            }

            // printf("****************\nAfter %s learning\n", arg->name);
            arg->qlt_train->distortion[ m ][ j+1 ] = distortion_measure(arg->map[ m ], arg->train_set[ j ], NBITEREPOCH,
                                                                      SIGMA_GAUSS);
//            arg->qlt_train->distortion_gss[ m ][ j+1 ] = distortion_measure_GSS(arg->map[ m ], arg->train_set[ j ],
//                                                                              NBITEREPOCH, SIGMA_GAUSS);

            arg->qlt_train->quantization[ m ][ j+1 ] = avg_quant_error(arg->map[ m ], arg->train_set[ j ], NBITEREPOCH);
//            arg->qlt_train->quantization_gss[ m ][ j+1 ] = avg_quant_error_GSS(arg->map[ m ], arg->train_set[ j ],
//                                                                             NBITEREPOCH);

            arg->qlt_valid->distortion[ m ][ j+1 ] = distortion_measure(arg->map[ m ], arg->valid_set, TEST_DENSITY,
                                                                      SIGMA_GAUSS);
//            arg->qlt_valid->distortion_gss[ m ][ j+1 ] = distortion_measure_GSS(arg->map[ m ], arg->valid_set,
//                                                                              SIZE*SIZE*TEST_DENSITY, SIGMA_GAUSS);

            arg->qlt_valid->quantization[ m ][ j+1 ] = avg_quant_error(arg->map[ m ], arg->valid_set, TEST_DENSITY);
//            arg->qlt_valid->quantization_gss[ m ][ j+1 ] = avg_quant_error_GSS(arg->map[ m ], arg->valid_set,
//                                                                             SIZE*SIZE*TEST_DENSITY);
            for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
                for ( e = 0; e < nb_experiments; e++ ) {
                    map2 = copy(arg->map[ m ]);
                    faulty_weights(map2, p);
                    arg->qlt_test->quantization[ j+1 ][ p ][ e ][ m ] = avg_quant_error(arg->map[ m ], arg->test_set, TEST2_DENSITY);
                    arg->qlt_test->quantization_faulty[ j+1 ][ p ][ e ][ m ] = avg_quant_error(map2, arg->test_set, TEST2_DENSITY);
//                    arg->qlt_test->quantization_gss[ j+1 ][ p ][ e ][ m ] = avg_quant_error_GSS(arg->map[ m ], arg->test_set,
//                                                                                         SIZE*SIZE*TEST2_DENSITY);
//                    arg->qlt_test->quantization_gss_faulty[j+1][ p ][ e ][ m ] = avg_quant_error_GSS(map2, arg->test_set,
//                                                                                               SIZE*SIZE*TEST2_DENSITY);
                    arg->qlt_test->distortion[ j+1 ][ p ][ e ][ m ] = distortion_measure(arg->map[ m ], arg->test_set, TEST2_DENSITY,
                                                                                         SIGMA_GAUSS);
//                    arg->qlt_test->distortion_gss[j+1][ p ][ e ][ m ] = distortion_measure_GSS(arg->map[ m ], arg->test_set,
//                                                                                          SIZE*SIZE*TEST2_DENSITY, SIGMA_GAUSS);
                    arg->qlt_test->distortion_faulty[ j+1 ][ p ][ e ][ m ] = distortion_measure(map2, arg->test_set, TEST2_DENSITY,
                                                                                                SIGMA_GAUSS);
//                    arg->qlt_test->distortion_gss_faulty[j+1][ p ][ e ][ m ] = distortion_measure_GSS(map2, arg->test_set,
//                                                                                                 SIZE*SIZE*TEST2_DENSITY, SIGMA_GAUSS);
                    freeMap(map2);
                }
            }
            i++;
            arg->ready = (float) (i * 100.0 / (1.0 * NBMAPS * NBEPOCHLEARN));
        }
    }
    free(arg->status);
    temp_str = "Evaluating : ";
    arg->status = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(arg->status, temp_str);
    strcat(arg->status, arg->name);

    i = 0;
    for ( j = 0; j < NBEPOCHLEARN+1; j++ ) {
        for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
            for ( e = 0; e < nb_experiments; e++ ) {
                for ( m = 0; m < NBMAPS; m++ ) {
                    arg->stat->avg[ j ][ p ] += arg->qlt_test->quantization[ j ][ p ][ e ][ m ];
//                    arg->stat->avg_gss[ j ][ p ] += arg->qlt_test->quantization_gss[ j ][ p ][ e ][ m ];
                    arg->stat->avg_faulty[ j ][ p ] += arg->qlt_test->quantization_faulty[ j ][ p ][ e ][ m ];
//                    arg->stat->avg_gss_faulty[ j ][ p ] += arg->qlt_test->quantization_gss_faulty[ j ][ p ][ e ][ m ];

                    arg->stat->avgdist[ j ][ p ] += arg->qlt_test->distortion[ j ][ p ][ e ][ m ];
//                    arg->stat->avgdist_gss[ j ][ p ] += arg->qlt_test->distortion_gss[ j ][ p ][ e ][ m ];
                    arg->stat->avgdist_faulty[ j ][ p ] += arg->qlt_test->distortion_faulty[ j ][ p ][ e ][ m ];
//                    arg->stat->avgdist_gss_faulty[ j ][ p ] += arg->qlt_test->distortion_gss_faulty[ j ][ p ][ e ][ m ];
                    i++;
                    arg->ready = (float) (i * 100.0 / (2.0 * NBMAPS * nb_experiments * MAXFAULTPERCENT * (NBEPOCHLEARN+1)));
                }
            }

            arg->stat->avg[ j ][ p ] /= tt;
            arg->stat->avg_gss[ j ][ p ] /= tt;
            arg->stat->avg_faulty[ j ][ p ] /= tt;
            arg->stat->avg_gss_faulty[ j ][ p ] /= tt;
            arg->stat->avgdist[ j ][ p ] /= tt;
            arg->stat->avgdist_gss[ j ][ p ] /= tt;
            arg->stat->avgdist_faulty[ j ][ p ] /= tt;
            arg->stat->avgdist_gss_faulty[ j ][ p ] /= tt;
        }

        for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
            for ( e = 0; e < nb_experiments; e++ ) {
                for ( m = 0; m < NBMAPS; m++ ) {
                    arg->stat->stddev[ j ][ p ] +=
                            (arg->qlt_test->quantization[ j ][ p ][ e ][ m ] - arg->stat->avg[ j ][ p ]) *
                            (arg->qlt_test->quantization[ j ][ p ][ e ][ m ] - arg->stat->avg[ j ][ p ]);
//                    arg->stat->stddev_gss[ j ][ p ] +=
//                            (arg->qlt_test->quantization_gss[ j ][ p ][ e ][ m ] - arg->stat->avg_gss[ j ][ p ]) *
//                            (arg->qlt_test->quantization_gss[ j ][ p ][ e ][ m ] - arg->stat->avg_gss[ j ][ p ]);
                    arg->stat->stddev_faulty[ j ][ p ] +=
                            (arg->qlt_test->quantization_faulty[ j ][ p ][ e ][ m ] - arg->stat->avg_faulty[ j ][ p ]) *
                            (arg->qlt_test->quantization_faulty[ j ][ p ][ e ][ m ] - arg->stat->avg_faulty[ j ][ p ]);
//                    arg->stat->stddev_gss_faulty[ j ][ p ] +=
//                            (arg->qlt_test->quantization_gss_faulty[ j ][ p ][ e ][ m ] -
//                             arg->stat->avg_gss_faulty[ j ][ p ]) *
//                            (arg->qlt_test->quantization_gss_faulty[ j ][ p ][ e ][ m ] -
//                             arg->stat->avg_gss_faulty[ j ][ p ]);

                    arg->stat->stddevdist[ j ][ p ] +=
                            (arg->qlt_test->distortion[ j ][ p ][ e ][ m ] - arg->stat->avgdist[ j ][ p ]) *
                            (arg->qlt_test->distortion[ j ][ p ][ e ][ m ] - arg->stat->avgdist[ j ][ p ]);
//                    arg->stat->stddevdist_gss[ j ][ p ] +=
//                            (arg->qlt_test->distortion_gss[ j ][ p ][ e ][ m ] - arg->stat->avgdist_gss[ j ][ p ]) *
//                            (arg->qlt_test->distortion_gss[ j ][ p ][ e ][ m ] - arg->stat->avgdist_gss[ j ][ p ]);
                    arg->stat->stddevdist_faulty[ j ][ p ] +=
                            (arg->qlt_test->distortion_faulty[ j ][ p ][ e ][ m ] -
                             arg->stat->avgdist_faulty[ j ][ p ]) *
                            (arg->qlt_test->distortion_faulty[ j ][ p ][ e ][ m ] -
                             arg->stat->avgdist_faulty[ j ][ p ]);
//                    arg->stat->stddevdist_gss_faulty[ j ][ p ] +=
//                            (arg->qlt_test->distortion_gss_faulty[ j ][ p ][ e ][ m ] -
//                             arg->stat->avgdist_gss_faulty[ j ][ p ]) *
//                            (arg->qlt_test->distortion_gss_faulty[ j ][ p ][ e ][ m ] -
//                             arg->stat->avgdist_gss_faulty[ j ][ p ]);
                    i++;
                    arg->ready = (float) (i * 100.0 / (2.0 * NBMAPS * nb_experiments * MAXFAULTPERCENT * (NBEPOCHLEARN+1)));
                }
            }
            arg->stat->stddev[ j ][ p ] = mysqrt(arg->stat->stddev[ j ][ p ] / (tt - 1));
//            arg->stat->stddev_gss[ j ][ p ] = mysqrt(arg->stat->stddev_gss[ j ][ p ] / (tt - 1));
            arg->stat->stddev_faulty[ j ][ p ] = mysqrt(arg->stat->stddev_faulty[ j ][ p ] / (tt - 1));
//            arg->stat->stddev_gss_faulty[ j ][ p ] = mysqrt(arg->stat->stddev_gss_faulty[ j ][ p ] / (tt - 1));

            arg->stat->stddevdist[ j ][ p ] = mysqrt(arg->stat->stddevdist[ j ][ p ] / (tt - 1));
//            arg->stat->stddevdist_gss[ j ][ p ] = mysqrt(arg->stat->stddevdist_gss[ j ][ p ] / (tt - 1));
            arg->stat->stddevdist_faulty[ j ][ p ] = mysqrt(arg->stat->stddevdist_faulty[ j ][ p ] / (tt - 1));
//            arg->stat->stddevdist_gss_faulty[ j ][ p ] = mysqrt(arg->stat->stddevdist_gss_faulty[ j ][ p ] / (tt - 1));
        }
    }
    free(arg->status);
    temp_str = "File output: ";
    arg->status = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(arg->status, temp_str);
    strcat(arg->status, arg->name);

   FILE *fp;
   char *temp = "n_learn_evaluting_";
   char *file_ext = ".txt";
   char *buffer = malloc(strlen(temp) + strlen(arg->name) + strlen(file_ext) + 1);
   strcpy(buffer, temp);
   strcat(buffer, arg->name);
   strcat(buffer, file_ext);
   fp = fopen(buffer, "w+");
   fprintf(fp, "MapNumber;EpochNumber;distortion;distortion_gss;quantization;quantization_gss\n");
   for ( m = 0; m < NBMAPS; m++ ) {
       for ( j = 0; j < NBEPOCHLEARN; j++ ) {
           fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", m, j,
                   arg->qlt_train->distortion[ m ][ j ],
                   arg->qlt_train->distortion_gss[ m ][ j ],
                   arg->qlt_train->quantization[ m ][ j ],
                   arg->qlt_train->quantization_gss[ m ][ j ]);
       }
   }
   fclose(fp);

   temp = "n_valid_evaluting_";
   free(buffer);
   buffer = malloc(strlen(temp) + strlen(arg->name) + strlen(file_ext) + 1);
   strcpy(buffer, temp);
   strcat(buffer, arg->name);
   strcat(buffer, file_ext);
   fp = fopen(buffer, "w+");
   fprintf(fp, "MapNumber;EpochNumber;distortion;distortion_gss;quantization;quantization_gss\n");
   for ( m = 0; m < NBMAPS; m++ ) {
       for ( j = 0; j < NBEPOCHLEARN; j++ ) {
           fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f\n", m, j,
                   arg->qlt_valid->distortion[ m ][ j ],
                   arg->qlt_valid->distortion_gss[ m ][ j ],
                   arg->qlt_valid->quantization[ m ][ j ],
                   arg->qlt_valid->quantization_gss[ m ][ j ]);
       }
   }
   fclose(fp);

   temp = "n_test_evaluting_";
   free(buffer);
   buffer = malloc(strlen(temp) + strlen(arg->name) + strlen(file_ext) + 1);
   strcpy(buffer, temp);
   strcat(buffer, arg->name);
   strcat(buffer, file_ext);
   fp = fopen(buffer, "w+");
   fprintf(fp,
           "Epoch;PercentageFaults;ExpNumber;MapNumber;quantization;quantization_gss;quantization_faulty;quantization_gss_faulty;distortion;distortion_gss;distortion_faulty;distortion_gss_faulty\n");
    for ( j = 0; j < NBEPOCHLEARN+1; j++ ) {
       for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
           for ( e = 0; e < nb_experiments; e++ ) {
               for ( m = 0; m < NBMAPS; m++ ) {
                   fprintf(fp, "%-d; %-d; %-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", j, p, e, m,
                           arg->qlt_test->quantization[ j ][ p ][ e ][ m ],
                           arg->qlt_test->quantization_gss[ j ][ p ][ e ][ m ],
                           arg->qlt_test->quantization_faulty[ j ][ p ][ e ][ m ],
                           arg->qlt_test->quantization_gss_faulty[ j ][ p ][ e ][ m ],
                           arg->qlt_test->distortion[ j ][ p ][ e ][ m ],
                           arg->qlt_test->distortion_gss[ j ][ p ][ e ][ m ],
                           arg->qlt_test->distortion_faulty[ j ][ p ][ e ][ m ],
                           arg->qlt_test->distortion_gss_faulty[ j ][ p ][ e ][ m ]);
               }
           }
       }
    }
   fclose(fp);

   temp = "n_statistics_";
   free(buffer);
   buffer = malloc(strlen(temp) + strlen(arg->name) + strlen(file_ext) + 1);
   strcpy(buffer, temp);
   strcat(buffer, arg->name);
   strcat(buffer, file_ext);
   fp = fopen(buffer, "w+");
   fprintf(fp,
           "Epoch;PercentageFaults;avg;avg_gss;avg_faulty;avg_gss_faulty;avgdist;avgdist_gss;avgdist_faulty;avgdist_gss_faulty;stddev;stddev_gss;stddev_faulty;stddev_gss_faulty;stddevdist;stddevdist_gss;stddevdist_faulty;stddevdist_gss_faulty\n");
    for ( j = 0; j < NBEPOCHLEARN+1; j++ ) {
        for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
            fprintf(fp, "%-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", j, p,
                    arg->stat->avg[ j ][ p ],
                    arg->stat->avg_gss[ j ][ p ],
                    arg->stat->avg_faulty[ j ][ p ],
                    arg->stat->avg_gss_faulty[ j ][ p ],
                    arg->stat->avgdist[ j ][ p ],
                    arg->stat->avgdist_gss[ j ][ p ],
                    arg->stat->avgdist_faulty[ j ][ p ],
                    arg->stat->avgdist_gss_faulty[ j ][ p ],
                    arg->stat->stddev[ j ][ p ],
                    arg->stat->stddev_gss[ j ][ p ],
                    arg->stat->stddev_faulty[ j ][ p ],
                    arg->stat->stddev_gss_faulty[ j ][ p ],
                    arg->stat->stddevdist[ j ][ p ],
                    arg->stat->stddevdist_gss[ j ][ p ],
                    arg->stat->stddevdist_faulty[ j ][ p ],
                    arg->stat->stddevdist_gss_faulty[ j ][ p ]);
        }
    }
   fclose(fp);



    free(arg->status);
    temp_str = "Complete: ";
    arg->status = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(arg->status, temp_str);
    strcat(arg->status, arg->name);


    clock_t stop = clock();
    float elapsed = (float) (stop - start) / 1000.0;
    arg->ready = elapsed;
    //printf("Time elapsed in ms: %f\n", elapsed);

    pthread_exit(NULL);
    return NULL;
}