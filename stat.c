#include "stat.h"

void avgstddev_init(Avgstddev * stat){
	stat->avg 				= 0 ;
	stat->avgdist			= 0 ;
	stat->stddev 			= 0 ;
	stat->stddevdist	= 0 ;
}

void print_avgstddev(Avgstddev stat){
	printf("Avg: %-f\n", stat.avg);
	printf("Std dev: %-f\n", stat.stddev);
	printf("Avg dist: %-f\n", stat.avgdist);
	printf("Std dev dist: %-f\n", stat.stddevdist);
}

void acc_avgstddev(Avgstddev * stat, float faults, float addeddist){
	stat->avg 				+= faults;
	stat->avgdist			+= addeddist;
	stat->stddev 			+= faults * faults;
	stat->stddevdist	+= addeddist * addeddist;
}

void calc_avgstddev(Avgstddev * stat){
	int tt 	= nb_experiments*NBMAPS;	

	stat->avg 			 /= tt;
	stat->avgdist		 /= tt;
	stat->stddev 		= mysqrt(((stat->stddev)-tt*(stat->avg)*(stat->avg))/
											(tt-1));
	stat->stddevdist	= mysqrt(((stat->stddevdist)-tt*(stat->avgdist)*
												(stat->avgdist))/(tt-1));
}

void Mainavgstddev_init(Mainavgstddev * mains, int size){
	mains->mainavg 				= (float *) malloc(size * sizeof(float));
	mains->mainavgdist 		= (float *) malloc(size * sizeof(float));
	mains->mainstddev 		= (float *) malloc(size * sizeof(float));
	mains->mainstddevdist = (float *) malloc(size * sizeof(float));
}

void add_avgstddev_mainavgstddev(Mainavgstddev * mains, Avgstddev stat, int pos){
	mains->mainavg[pos] 				+= stat.avg;
	mains->mainavgdist[pos] 		+= stat.avgdist;
	mains->mainstddev[pos] 			+= stat.stddev;
	mains->mainstddevdist[pos] 	+= stat.stddevdist;
}

void print_mainavgstddev(Mainavgstddev mains, int size){
	int i;
	printf("\nMainavg: ");
	for(i = 0; i < size; i++) printf("%-f\t", mains.mainavg[i]);
	printf("\nMainavgdist: ");
	for(i = 0; i < size; i++) printf("%-f\t", mains.mainavgdist[i]);
	printf("\nMainstddev: ");
	for(i = 0; i < size; i++) printf("%-f\t", mains.mainstddev[i]);
	printf("\nMainstddevdist: ");
	for(i = 0; i < size; i++) printf("%-f\t", mains.mainstddevdist[i]);
	printf("\n");
} 
	
int ** malloc_2darray(int height, int width){
	int ** cl;
	int i, j;

	cl = (int **)malloc(sizeof(int *) * height);
	for (i = 0; i < height; i++){
		cl[i] = (int *)malloc(sizeof(int) * width); 
	}

	return cl;
}
