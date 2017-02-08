
#include "threads.h"
#include "malloc.h"
#include "string.h"
#include "unistd.h"

void init_evaluations(Evaluations *ev) {
    ev->distortion = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
    ev->quantization = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
    ev->distortion_gss = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
    ev->quantization_gss = (double **) malloc_2darray_f(NBMAPS, NBEPOCHLEARN + 1);
}

void init_evaluations_faulty(Evaluations_faulty *ev) {
    ev->distortion = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->distortion_gss = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization_gss = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->distortion_faulty = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization_faulty = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->distortion_gss_faulty = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
    ev->quantization_gss_faulty = (double ***) malloc_3darray_f(MAXFAULTPERCENT, nb_experiments, NBMAPS);
}

void init_statistics(Statistics *st) {
    st->avg = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddev = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avg_gss = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddev_gss = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avg_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddev_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avg_gss_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddev_gss_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avgdist = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddevdist = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avgdist_gss = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddevdist_gss = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avgdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddevdist_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->avgdist_gss_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
    st->stddevdist_gss_faulty = calloc(MAXFAULTPERCENT, sizeof(double));
}

void *monitor_thread(void *args) {
    Monitor *mon = args;

    while (1) {
        printf("Thread 1 status: %s\n", *mon->th1);
        printf("Thread 2 status: %s\n", *mon->th2);
        printf("Thread 3 status: %s\n", *mon->th3);
        printf("Thread 4 status: %s\n", *mon->th4);
        printf("Thread 5 status: %s\n", *mon->th5);
        usleep(50000);
    }
    exit(1);
}

void *learning_thread(void *args) {
    Input_args *arg = args;

    clock_t start = clock();

    double tt = nb_experiments * NBMAPS;
    int p, e, m, j, i;
    char *temp_str = "Start learning: ";
    char *tmp;
    tmp = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(tmp, temp_str);
    strcat(tmp, arg->name);
    arg->status = tmp;

    for ( m = 0; m < NBMAPS; m++ ) {
        // printf("****************\nBefore learning\n");
        arg->qlt_train->distortion[ m ][ 0 ] = distortion_measure(arg->map[ m ], arg->train_set[ 0 ], NBITEREPOCH,
                                                                  SIGMA_GAUSS);
        arg->qlt_train->distortion_gss[ m ][ 0 ] = distortion_measure_GSS(arg->map[ m ], arg->train_set[ 0 ],
                                                                          NBITEREPOCH, SIGMA_GAUSS);

        arg->qlt_train->quantization[ m ][ 0 ] = avg_quant_error(arg->map[ m ], arg->train_set[ 0 ], NBITEREPOCH);
        arg->qlt_train->quantization_gss[ m ][ 0 ] = avg_quant_error_GSS(arg->map[ m ], arg->train_set[ 0 ],
                                                                         NBITEREPOCH);

        arg->qlt_valid->distortion[ m ][ 0 ] = distortion_measure(arg->map[ m ], arg->valid_set, NBITEREPOCH,
                                                                  SIGMA_GAUSS);
        arg->qlt_valid->distortion_gss[ m ][ 0 ] = distortion_measure_GSS(arg->map[ m ], arg->valid_set, NBITEREPOCH,
                                                                          SIGMA_GAUSS);

        arg->qlt_valid->quantization[ m ][ 0 ] = avg_quant_error(arg->map[ m ], arg->valid_set, NBITEREPOCH);
        arg->qlt_valid->quantization_gss[ m ][ 0 ] = avg_quant_error_GSS(arg->map[ m ], arg->valid_set, NBITEREPOCH);

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
            arg->qlt_train->distortion[ m ][ j ] = distortion_measure(arg->map[ m ], arg->train_set[ j ], NBITEREPOCH,
                                                                      SIGMA_GAUSS);
            arg->qlt_train->distortion_gss[ m ][ j ] = distortion_measure_GSS(arg->map[ m ], arg->train_set[ j ],
                                                                              NBITEREPOCH,
                                                                              SIGMA_GAUSS);

            arg->qlt_train->quantization[ m ][ j ] = avg_quant_error(arg->map[ m ], arg->train_set[ j ], NBITEREPOCH);
            arg->qlt_train->quantization_gss[ m ][ j ] = avg_quant_error_GSS(arg->map[ m ], arg->train_set[ j ],
                                                                             NBITEREPOCH);

            arg->qlt_valid->distortion[ m ][ j ] = distortion_measure(arg->map[ m ], arg->valid_set, NBITEREPOCH,
                                                                      SIGMA_GAUSS);
            arg->qlt_valid->distortion_gss[ m ][ j ] = distortion_measure_GSS(arg->map[ m ], arg->valid_set,
                                                                              NBITEREPOCH,
                                                                              SIGMA_GAUSS);

            arg->qlt_valid->quantization[ m ][ j ] = avg_quant_error(arg->map[ m ], arg->valid_set, NBITEREPOCH);
            arg->qlt_valid->quantization_gss[ m ][ j ] = avg_quant_error_GSS(arg->map[ m ], arg->valid_set,
                                                                             NBITEREPOCH);
        }
    }
    free(arg->status);
    temp_str = "Learning finished: ";
    arg->status = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(arg->status, temp_str);
    strcat(arg->status, arg->name);
    Kohonen map2;
    for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
        for ( e = 0; e < nb_experiments; e++ ) {
            for ( m = 0; m < NBMAPS; m++ ) {
                map2 = copy(arg->map[ m ]);

                // introduction of faults in the copies of the pre-learned maps
                faulty_weights(map2, p);

                arg->qlt_test->quantization[ p ][ e ][ m ] = avg_quant_error(arg->map[ m ], arg->test_set, NBITEREPOCH);
                arg->qlt_test->quantization_gss[ p ][ e ][ m ] = avg_quant_error_GSS(arg->map[ m ], arg->test_set,
                                                                                     NBITEREPOCH);
                arg->qlt_test->quantization_faulty[ p ][ e ][ m ] = avg_quant_error(map2, arg->test_set, NBITEREPOCH);
                arg->qlt_test->quantization_gss_faulty[ p ][ e ][ m ] = avg_quant_error_GSS(map2, arg->test_set,
                                                                                            NBITEREPOCH);

                arg->qlt_test->distortion[ p ][ e ][ m ] = distortion_measure(arg->map[ m ], arg->test_set, NBITEREPOCH,
                                                                              SIGMA_GAUSS);
                arg->qlt_test->distortion_gss[ p ][ e ][ m ] = distortion_measure_GSS(arg->map[ m ], arg->test_set,
                                                                                      NBITEREPOCH, SIGMA_GAUSS);
                arg->qlt_test->distortion_faulty[ p ][ e ][ m ] = distortion_measure(map2, arg->test_set, NBITEREPOCH,
                                                                                     SIGMA_GAUSS);
                arg->qlt_test->distortion_gss_faulty[ p ][ e ][ m ] = distortion_measure_GSS(map2, arg->test_set,
                                                                                             NBITEREPOCH, SIGMA_GAUSS);

                arg->stat->avg[ p ] += arg->qlt_test->quantization[ p ][ e ][ m ];
                arg->stat->avg_gss[ p ] += arg->qlt_test->quantization_gss[ p ][ e ][ m ];
                arg->stat->avg_faulty[ p ] += arg->qlt_test->quantization_faulty[ p ][ e ][ m ];
                arg->stat->avg_gss_faulty[ p ] += arg->qlt_test->quantization_gss_faulty[ p ][ e ][ m ];

                arg->stat->avgdist[ p ] += arg->qlt_test->distortion[ p ][ e ][ m ];
                arg->stat->avgdist_gss[ p ] += arg->qlt_test->distortion_gss[ p ][ e ][ m ];
                arg->stat->avgdist_faulty[ p ] += arg->qlt_test->distortion_faulty[ p ][ e ][ m ];
                arg->stat->avgdist_gss_faulty[ p ] += arg->qlt_test->distortion_gss_faulty[ p ][ e ][ m ];
            }
        }

        arg->stat->avg[ p ] /= tt;
        arg->stat->avg_gss[ p ] /= tt;
        arg->stat->avg_faulty[ p ] /= tt;
        arg->stat->avg_gss_faulty[ p ] /= tt;
        arg->stat->avgdist[ p ] /= tt;
        arg->stat->avgdist_gss[ p ] /= tt;
        arg->stat->avgdist_faulty[ p ] /= tt;
        arg->stat->avgdist_gss_faulty[ p ] /= tt;
    }

    for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
        for ( e = 0; e < nb_experiments; e++ ) {
            for ( m = 0; m < NBMAPS; m++ ) {
                arg->stat->stddev[ p ] += (arg->qlt_test->quantization[ p ][ e ][ m ] - arg->stat->avg[ p ]) *
                                          (arg->qlt_test->quantization[ p ][ e ][ m ] - arg->stat->avg[ p ]);
                arg->stat->stddev_gss[ p ] +=
                        (arg->qlt_test->quantization_gss[ p ][ e ][ m ] - arg->stat->avg_gss[ p ]) *
                        (arg->qlt_test->quantization_gss[ p ][ e ][ m ] - arg->stat->avg_gss[ p ]);
                arg->stat->stddev_faulty[ p ] +=
                        (arg->qlt_test->quantization_faulty[ p ][ e ][ m ] - arg->stat->avg_faulty[ p ]) *
                        (arg->qlt_test->quantization_faulty[ p ][ e ][ m ] - arg->stat->avg_faulty[ p ]);
                arg->stat->stddev_gss_faulty[ p ] +=
                        (arg->qlt_test->quantization_gss_faulty[ p ][ e ][ m ] - arg->stat->avg_gss_faulty[ p ]) *
                        (arg->qlt_test->quantization_gss_faulty[ p ][ e ][ m ] - arg->stat->avg_gss_faulty[ p ]);

                arg->stat->stddevdist[ p ] += (arg->qlt_test->distortion[ p ][ e ][ m ] - arg->stat->avgdist[ p ]) *
                                              (arg->qlt_test->distortion[ p ][ e ][ m ] - arg->stat->avgdist[ p ]);
                arg->stat->stddevdist_gss[ p ] +=
                        (arg->qlt_test->distortion_gss[ p ][ e ][ m ] - arg->stat->avgdist_gss[ p ]) *
                        (arg->qlt_test->distortion_gss[ p ][ e ][ m ] - arg->stat->avgdist_gss[ p ]);
                arg->stat->stddevdist_faulty[ p ] +=
                        (arg->qlt_test->distortion_faulty[ p ][ e ][ m ] - arg->stat->avgdist_faulty[ p ]) *
                        (arg->qlt_test->distortion_faulty[ p ][ e ][ m ] - arg->stat->avgdist_faulty[ p ]);
                arg->stat->stddevdist_gss_faulty[ p ] +=
                        (arg->qlt_test->distortion_gss_faulty[ p ][ e ][ m ] - arg->stat->avgdist_gss_faulty[ p ]) *
                        (arg->qlt_test->distortion_gss_faulty[ p ][ e ][ m ] - arg->stat->avgdist_gss_faulty[ p ]);
            }
        }
        arg->stat->stddev[ p ] = mysqrt(arg->stat->stddev[ p ] / (tt - 1));
        arg->stat->stddev_gss[ p ] = mysqrt(arg->stat->stddev_gss[ p ] / (tt - 1));
        arg->stat->stddev_faulty[ p ] = mysqrt(arg->stat->stddev_faulty[ p ] / (tt - 1));
        arg->stat->stddev_gss_faulty[ p ] = mysqrt(arg->stat->stddev_gss_faulty[ p ] / (tt - 1));

        arg->stat->stddevdist[ p ] = mysqrt(arg->stat->stddevdist[ p ] / (tt - 1));
        arg->stat->stddevdist_gss[ p ] = mysqrt(arg->stat->stddevdist_gss[ p ] / (tt - 1));
        arg->stat->stddevdist_faulty[ p ] = mysqrt(arg->stat->stddevdist_faulty[ p ] / (tt - 1));
        arg->stat->stddevdist_gss_faulty[ p ] = mysqrt(arg->stat->stddevdist_gss_faulty[ p ] / (tt - 1));
    }
    free(arg->status);
    temp_str = "Testing finished: ";
    arg->status = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(arg->status, temp_str);
    strcat(arg->status, arg->name);

    FILE *fp;
    char *temp = "learn_evaluting_";
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

    temp = "valid_evaluting_";
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

    temp = "test_evaluting_";
    free(buffer);
    buffer = malloc(strlen(temp) + strlen(arg->name) + strlen(file_ext) + 1);
    strcpy(buffer, temp);
    strcat(buffer, arg->name);
    strcat(buffer, file_ext);
    fp = fopen(buffer, "w+");
    fprintf(fp,
            "PercentageFaults;ExpNumber;MapNumber;quantization;quantization_gss;quantization_faulty;quantization_gss_faulty;distortion;distortion_gss;distortion_faulty;distortion_gss_faulty\n");
    for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
        for ( e = 0; e < nb_experiments; e++ ) {
            for ( m = 0; m < NBMAPS; m++ ) {
                fprintf(fp, "%-d; %-d; %-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p, e, m,
                        arg->qlt_test->quantization[ p ][ e ][ m ],
                        arg->qlt_test->quantization_gss[ p ][ e ][ m ],
                        arg->qlt_test->quantization_faulty[ p ][ e ][ m ],
                        arg->qlt_test->quantization_gss_faulty[ p ][ e ][ m ],
                        arg->qlt_test->distortion[ p ][ e ][ m ],
                        arg->qlt_test->distortion_gss[ p ][ e ][ m ],
                        arg->qlt_test->distortion_faulty[ p ][ e ][ m ],
                        arg->qlt_test->distortion_gss_faulty[ p ][ e ][ m ]);
            }
        }
    }
    fclose(fp);

    temp = "statistics_";
    free(buffer);
    buffer = malloc(strlen(temp) + strlen(arg->name) + strlen(file_ext) + 1);
    strcpy(buffer, temp);
    strcat(buffer, arg->name);
    strcat(buffer, file_ext);
    fp = fopen(buffer, "w+");
    fprintf(fp,
            "PercentageFaults;avg;avg_gss;avg_faulty;avg_gss_faulty;avgdist;avgdist_gss;avgdist_faulty;avgdist_gss_faulty; stddev;stddev_gss;stddev_faulty;stddev_gss;stddev_gss_faulty;stddevdist;stddevdist_gss;stddevdist_faulty;stddevdist_gss_faulty\n");
    for ( p = 0; p < MAXFAULTPERCENT; p++ ) {
        fprintf(fp, "%-d; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f; %-f\n", p,
                arg->stat->avg[ p ],
                arg->stat->avg_gss[ p ],
                arg->stat->avg_faulty[ p ],
                arg->stat->avg_gss_faulty[ p ],
                arg->stat->avgdist[ p ],
                arg->stat->avgdist_gss[ p ],
                arg->stat->avgdist_faulty[ p ],
                arg->stat->avgdist_gss_faulty[ p ],
                arg->stat->stddev[ p ],
                arg->stat->stddev_gss[ p ],
                arg->stat->stddev_faulty[ p ],
                arg->stat->stddev_gss_faulty[ p ],
                arg->stat->stddevdist[ p ],
                arg->stat->stddevdist_gss[ p ],
                arg->stat->stddevdist_faulty[ p ],
                arg->stat->stddevdist_gss_faulty[ p ]);
    }
    fclose(fp);

    freeMap(map2);

    free(arg->status);
    temp_str = "Complete: ";
    arg->status = malloc(strlen(temp_str) + strlen(arg->name) + 1);
    strcpy(arg->status, temp_str);
    strcat(arg->status, arg->name);


    clock_t stop = clock();
    double elapsed = (double) (stop - start) / 1000.0;
    printf("Time elapsed in ms: %f\n", elapsed);

    pthread_exit(NULL);
    return NULL;
}