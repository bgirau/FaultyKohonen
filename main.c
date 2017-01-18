
#include "func_def.h"
#include "gss_filter.h"


int main() {
  int v;
  srand(time(NULL));
  int  p,q,i,j,e,k,cnt,s,inp,m;
  double avg,stddev,avgdist,stddevdist;
  double avg2,stddev2,avgdist2,stddevdist2;
  double avg_th,stddev_th,avgdist_th,stddevdist_th;
  double avg_FI,stddev_FI,avgdist_FI,stddevdist_FI;
  double avg_NI,stddev_NI,avgdist_NI,stddevdist_NI;
  double avg_NF,stddev_NF,avgdist_NF,stddevdist_NF;
  double avg2_th,stddev2_th,avgdist2_th,stddevdist2_th;
  double avg2_FI,stddev2_FI,avgdist2_FI,stddevdist2_FI;
  double avg2_NI,stddev2_NI,avgdist2_NI,stddevdist2_NI;
  double avg2_NF,stddev2_NF,avgdist2_NF,stddevdist2_NF;
  double avg_dnf,stddev_dnf,avgdist_dnf,stddevdist_dnf;
  double avg2_dnf,stddev2_dnf,avgdist2_dnf,stddevdist2_dnf;
  double avg_th_dnf,stddev_th_dnf,avgdist_th_dnf,stddevdist_th_dnf;
  double avg_FI_dnf,stddev_FI_dnf,avgdist_FI_dnf,stddevdist_FI_dnf;
  double avg_NI_dnf,stddev_NI_dnf,avgdist_NI_dnf,stddevdist_NI_dnf;
  double avg_NF_dnf,stddev_NF_dnf,avgdist_NF_dnf,stddevdist_NF_dnf;
  double avg2_th_dnf,stddev2_th_dnf,avgdist2_th_dnf,stddevdist2_th_dnf;
  double avg2_FI_dnf,stddev2_FI_dnf,avgdist2_FI_dnf,stddevdist2_FI_dnf;
  double avg2_NI_dnf,stddev2_NI_dnf,avgdist2_NI_dnf,stddevdist2_NI_dnf;
  double avg2_NF_dnf,stddev2_NF_dnf,avgdist2_NF_dnf,stddevdist2_NF_dnf;
  float x;
  int **in;//=(int*)malloc(INS*sizeof(int));
  int **classes;
  int** count;
  int** classe;
  int** classe_th;
  int** classe_FI;
  int** classe_NI;
  int** classe_NF;
  char* st=(char*)malloc(1000*sizeof(char));
  Kohonen *map;
  Kohonen *map_th;
  Kohonen *map_FI;
  Kohonen *map_NI;
  Kohonen *map_NF;
  Kohonen *mapinit;
  // load inputs
  inp=0;
  FILE *f=fopen(INPUTFILENAME,"r");
  while (fscanf(f,"%s",st)==1) inp++;
  fclose(f);
  inp/=(INS+1);
  printf("nombre d'entrées = %d\n",inp);
  in=(int**)malloc(inp*sizeof(int*));
  f=fopen(INPUTFILENAME,"r");
  for (i=0;i<inp;i++) {
    in[i]=(int*)malloc((INS+1)*sizeof(int));
    for (j=0;j<INS;j++) {
      fscanf(f,"%f",&x);
      in[i][j]=(int)((1.0*one)*x);
      if (abs(in[i][j])>precision_int) {
	printf("warning: overflow\n");
	exit(1);
      }
    }
    fscanf(f,"%d",&p);
    in[i][INS]=p;
  }
  fclose(f);

  // normalisation dans [0,1] par coordonnée
  for (j=0;j<INS;j++) {
    float min,max;
    min=in[0][j];
    max=min;
    for (i=0;i<inp;i++) {
      if (in[i][j]<min) min=in[i][j];
      if (in[i][j]>max) max=in[i][j];
    }
    for (i=0;i<inp;i++) {
      in[i][j]=(int)((1.0*one*(in[i][j]-min)/(max-min)));
    }
  }
  
  // affichage base 
  for (i=0;i<inp;i++) {
    for (j=0;j<INS;j++) {
      printf("%f ",in[i][j]/(1.0*one));
    }
    printf("%d\n",in[i][INS]);
  }
  classe=(int**)malloc(SIZE*sizeof(int*));
  for (i=0;i<SIZE;i++) {
    classe[i]=(int*)malloc(SIZE*sizeof(int));
    for (j=0;j<SIZE;j++) {
      classe[i][j]=0;
    }
  }
  classe_th=(int**)malloc(SIZE*sizeof(int*));
  classe_FI=(int**)malloc(SIZE*sizeof(int*));
  classe_NI=(int**)malloc(SIZE*sizeof(int*));
  classe_NF=(int**)malloc(SIZE*sizeof(int*));
  for (i=0;i<SIZE;i++) {
    classe_th[i]=(int*)malloc(SIZE*sizeof(int));
    classe_FI[i]=(int*)malloc(SIZE*sizeof(int));
    classe_NI[i]=(int*)malloc(SIZE*sizeof(int));
    classe_NF[i]=(int*)malloc(SIZE*sizeof(int));
    for (j=0;j<SIZE;j++) {
      classe_th[i][j]=0;
      classe_FI[i][j]=0;
      classe_NI[i][j]=0;
      classe_NF[i][j]=0;
    }
  }
  for (v=0;v<NBVALIDPARTITIONS;v++) {
    printf("\n****************************************\n************************************\nNEW CROSS VALIDATION PARTITIONING\n--------------------------------------\n\n");
    // répartition base en test/validation/apprentissge des 150 exemples IRIS
    int **crossvalid=(int**)malloc(TESTDIV*sizeof(int*));
    int *select;
    select=(int*)malloc(inp*sizeof(int));
    int n;
    for (i=0;i<TESTDIV;i++) crossvalid[i]=(int*)malloc((inp/TESTDIV)*sizeof(int));
    for (i=0;i<(inp/TESTDIV);i++) {
      for (j=0;j<TESTDIV;j++) {
	n=rand()%inp;
	while (select[n]==1) n=rand()%inp;
	crossvalid[j][i]=n;
	select[n]=1;
      }
    }
    /*
    // affichage blocs test 
    for (i=0;i<TESTDIV;i++) {
    for (j=0;j<(inp/TESTDIV);j++) {
    printf("crossvalid[%d][%d]=%d, classe=%d\n",i,j,crossvalid[i][j],in[crossvalid[i][j]][INS]);
    }
    }
    */
    int testbloc;
    map=(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_th=(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_FI=(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_NI=(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_NF=(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    mapinit=(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    for (i=0;i<NBMAPS;i++)
      mapinit[i]=init();
    int tmin,nmin,t;
    float d,dmin;

    // all main... variables accumulate corresponding results for all testbloc positions of the current cross validation partition
    // all ...2 variables correspond to test results ???
    float *mainavg;
    float *mainavgdist;
    float *mainstddev;
    float *mainstddevdist;
    float *mainavg2;
    float *mainavgdist2;
    float *mainstddev2;
    float *mainstddevdist2;
    mainavg=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    for (p=0;p<MAXFAULTPERCENT;p++) {
      mainavg[p]=0.0;
      mainavgdist[p]=0.0;
      mainstddev[p]=0.0;
      mainstddevdist[p]=0.0;
      mainavg2[p]=0.0;
      mainavgdist2[p]=0.0;
      mainstddev2[p]=0.0;
      mainstddevdist2[p]=0.0;
    }
    float *mainavg_dnf;
    float *mainavgdist_dnf;
    float *mainstddev_dnf;
    float *mainstddevdist_dnf;
    float *mainavg2_dnf;
    float *mainavgdist2_dnf;
    float *mainstddev2_dnf;
    float *mainstddevdist2_dnf;
    mainavg_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    for (p=0;p<MAXFAULTPERCENT;p++) {
      mainavg_dnf[p]=0.0;
      mainavgdist_dnf[p]=0.0;
      mainstddev_dnf[p]=0.0;
      mainstddevdist_dnf[p]=0.0;
      mainavg2_dnf[p]=0.0;
      mainavgdist2_dnf[p]=0.0;
      mainstddev2_dnf[p]=0.0;
      mainstddevdist2_dnf[p]=0.0;
    }
    float *mainavg_th;
    float *mainavg_FI;
    float *mainavg_NI;
    float *mainavg_NF;
    float *mainavgdist_th;
    float *mainavgdist_FI;
    float *mainavgdist_NI;
    float *mainavgdist_NF;
    float *mainstddev_th;
    float *mainstddev_FI;
    float *mainstddev_NI;
    float *mainstddev_NF;
    float *mainstddevdist_th;
    float *mainstddevdist_FI;
    float *mainstddevdist_NI;
    float *mainstddevdist_NF;
    float *mainavg2_th;
    float *mainavg2_FI;
    float *mainavg2_NI;
    float *mainavg2_NF;
    float *mainavgdist2_th;
    float *mainavgdist2_FI;
    float *mainavgdist2_NI;
    float *mainavgdist2_NF;
    float *mainstddev2_th;
    float *mainstddev2_FI;
    float *mainstddev2_NI;
    float *mainstddev2_NF;
    float *mainstddevdist2_th;
    float *mainstddevdist2_FI;
    float *mainstddevdist2_NI;
    float *mainstddevdist2_NF;
    mainavg_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_th=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_FI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_NI=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_NF=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    for (p=0;p<MAXFAULTPERCENT;p++) {
      mainavg_th[p]=0.0;
      mainavg_FI[p]=0.0;
      mainavg_NI[p]=0.0;
      mainavg_NF[p]=0.0;
      mainavgdist_th[p]=0.0;
      mainavgdist_FI[p]=0.0;
      mainavgdist_NI[p]=0.0;
      mainavgdist_NF[p]=0.0;
      mainstddev_th[p]=0.0;
      mainstddev_FI[p]=0.0;
      mainstddev_NI[p]=0.0;
      mainstddev_NF[p]=0.0;
      mainstddevdist_th[p]=0.0;
      mainstddevdist_FI[p]=0.0;
      mainstddevdist_NI[p]=0.0;
      mainstddevdist_NF[p]=0.0;
      mainavg2_th[p]=0.0;
      mainavg2_FI[p]=0.0;
      mainavg2_NI[p]=0.0;
      mainavg2_NF[p]=0.0;
      mainavgdist2_th[p]=0.0;
      mainavgdist2_FI[p]=0.0;
      mainavgdist2_NI[p]=0.0;
      mainavgdist2_NF[p]=0.0;
      mainstddev2_th[p]=0.0;
      mainstddev2_FI[p]=0.0;
      mainstddev2_NI[p]=0.0;
      mainstddev2_NF[p]=0.0;
      mainstddevdist2_th[p]=0.0;
      mainstddevdist2_FI[p]=0.0;
      mainstddevdist2_NI[p]=0.0;
      mainstddevdist2_NF[p]=0.0;
    }
    float *mainavg_th_dnf;
    float *mainavg_FI_dnf;
    float *mainavg_NI_dnf;
    float *mainavg_NF_dnf;
    float *mainavgdist_th_dnf;
    float *mainavgdist_FI_dnf;
    float *mainavgdist_NI_dnf;
    float *mainavgdist_NF_dnf;
    float *mainstddev_th_dnf;
    float *mainstddev_FI_dnf;
    float *mainstddev_NI_dnf;
    float *mainstddev_NF_dnf;
    float *mainstddevdist_th_dnf;
    float *mainstddevdist_FI_dnf;
    float *mainstddevdist_NI_dnf;
    float *mainstddevdist_NF_dnf;
    float *mainavg2_th_dnf;
    float *mainavg2_FI_dnf;
    float *mainavg2_NI_dnf;
    float *mainavg2_NF_dnf;
    float *mainavgdist2_th_dnf;
    float *mainavgdist2_FI_dnf;
    float *mainavgdist2_NI_dnf;
    float *mainavgdist2_NF_dnf;
    float *mainstddev2_th_dnf;
    float *mainstddev2_FI_dnf;
    float *mainstddev2_NI_dnf;
    float *mainstddev2_NF_dnf;
    float *mainstddevdist2_th_dnf;
    float *mainstddevdist2_FI_dnf;
    float *mainstddevdist2_NI_dnf;
    float *mainstddevdist2_NF_dnf;
    mainavg_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavg2_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainavgdist2_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddev2_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_th_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_FI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_NI_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    mainstddevdist2_NF_dnf=(float*)malloc(MAXFAULTPERCENT*sizeof(float));
    for (p=0;p<MAXFAULTPERCENT;p++) {
      mainavg_th_dnf[p]=0.0;
      mainavg_FI_dnf[p]=0.0;
      mainavg_NI_dnf[p]=0.0;
      mainavg_NF_dnf[p]=0.0;
      mainavgdist_th_dnf[p]=0.0;
      mainavgdist_FI_dnf[p]=0.0;
      mainavgdist_NI_dnf[p]=0.0;
      mainavgdist_NF_dnf[p]=0.0;
      mainstddev_th_dnf[p]=0.0;
      mainstddev_FI_dnf[p]=0.0;
      mainstddev_NI_dnf[p]=0.0;
      mainstddev_NF_dnf[p]=0.0;
      mainstddevdist_th_dnf[p]=0.0;
      mainstddevdist_FI_dnf[p]=0.0;
      mainstddevdist_NI_dnf[p]=0.0;
      mainstddevdist_NF_dnf[p]=0.0;
      mainavg2_th_dnf[p]=0.0;
      mainavg2_FI_dnf[p]=0.0;
      mainavg2_NI_dnf[p]=0.0;
      mainavg2_NF_dnf[p]=0.0;
      mainavgdist2_th_dnf[p]=0.0;
      mainavgdist2_FI_dnf[p]=0.0;
      mainavgdist2_NI_dnf[p]=0.0;
      mainavgdist2_NF_dnf[p]=0.0;
      mainstddev2_th_dnf[p]=0.0;
      mainstddev2_FI_dnf[p]=0.0;
      mainstddev2_NI_dnf[p]=0.0;
      mainstddev2_NF_dnf[p]=0.0;
      mainstddevdist2_th_dnf[p]=0.0;
      mainstddevdist2_FI_dnf[p]=0.0;
      mainstddevdist2_NI_dnf[p]=0.0;
      mainstddevdist2_NF_dnf[p]=0.0;
    }
    for (testbloc=0;testbloc<TESTDIV;testbloc++) {
      // first step : learn all maps (NBMAPS different initializations) by means of all available algorithms without the current testbloc
      for (m=0;m<NBMAPS;m++) {
	printf("\n********************************\n learning map number %d : \n\n",m);
	map[m]=copy(mapinit[m]);
	map_th[m]=copy(mapinit[m]);
	map_FI[m]=copy(mapinit[m]);
	map_NI[m]=copy(mapinit[m]);
	map_NF[m]=copy(mapinit[m]);
	printneuronclasses(map[m],in,classe,crossvalid,testbloc,inp);
	printf("****************\nBefore learning\n");
	errorrate(map[m],in,inp,classe,-1,crossvalid,testbloc);
	errorrateDNF(map[m],in,inp,classe,-1,crossvalid,testbloc);
	learn(map[m],in,inp,classe,crossvalid,testbloc);
	printf("****************\nAfter standard learning\n");
	errorrate(map[m],in,inp,classe,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	errorrateDNF(map[m],in,inp,classe,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	learn_threshold(map_th[m],in,inp,classe_th,crossvalid,testbloc);
	printf("****************\nAfter thresholded learning\n");
	errorrate(map_th[m],in,inp,classe_th,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	errorrateDNF(map_th[m],in,inp,classe_th,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	learn_FI(map_FI[m],in,inp,classe_FI,crossvalid,testbloc);
	printf("****************\nAfter fault injection learning\n");
	errorrate(map_FI[m],in,inp,classe_FI,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	errorrateDNF(map_FI[m],in,inp,classe_FI,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	learn_NI(map_NI[m],in,inp,classe_NI,crossvalid,testbloc);
	printf("****************\nAfter noise injection learning\n");
	errorrate(map_NI[m],in,inp,classe_NI,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	errorrateDNF(map_NI[m],in,inp,classe_NI,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	learn_NF(map_NF[m],in,inp,classe_NF,crossvalid,testbloc);
	printf("****************\nAfter NF driven learning\n");
	errorrate(map_NF[m],in,inp,classe_NF,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	errorrateDNF(map_NF[m],in,inp,classe_NF,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
	/*
	//affichage classes neurones
	for (i=0;i<SIZE;i++) {
	for (j=0;j<SIZE;j++) {
	printf("%d ",classe[i][j]);
	}
	printf("\n");
	}
	*/
      }
      // for learning study alone
      break;
      float dist,dist_th;
      float dist_FI;
      float dist_NI;
      float dist_NF;
      int cnt_th;
      int cnt_FI;
      int cnt_NI;
      int cnt_NF;
      float dist_dnf,dist_th_dnf;
      float dist_FI_dnf;
      float dist_NI_dnf;
      float dist_NF_dnf;
      int cnt_th_dnf;
      int cnt_FI_dnf;
      int cnt_NI_dnf;
      int cnt_NF_dnf;
      int cnt_dnf;
      for (p=0;p<MAXFAULTPERCENT;p++) {
	// study with respect to a fault percentage
	// all non main variables are reset (results averaged w.r.t. all experiments and maps)
	avg=0.0;
	avg_dnf=0.0;
	avgdist=0.0;
	avgdist_dnf=0.0;
	stddev=0.0;
	stddev_dnf=0.0;
	stddevdist=0.0;
	stddevdist_dnf=0.0;
	avg2=0.0;
	avg2_dnf=0.0;
	avgdist2=0.0;
	avgdist2_dnf=0.0;
	stddev2=0.0;
	stddev2_dnf=0.0;
	stddevdist2=0.0;
	stddevdist2_dnf=0.0;
	avg_th=0.0;
	avg_th_dnf=0.0;
	avg_FI=0.0;
	avg_FI_dnf=0.0;
	avg_NI=0.0;
	avg_NF=0.0;
	avg_NI_dnf=0.0;
	avg_NF_dnf=0.0;
	avgdist_th=0.0;
	avgdist_th_dnf=0.0;
	avgdist_FI=0.0;
	avgdist_FI_dnf=0.0;
	avgdist_NI=0.0;
	avgdist_NF=0.0;
	avgdist_NI_dnf=0.0;
	avgdist_NF_dnf=0.0;
	stddev_th=0.0;
	stddev_th_dnf=0.0;
	stddev_FI=0.0;
	stddev_FI_dnf=0.0;
	stddev_NI=0.0;
	stddev_NF=0.0;
	stddev_NI_dnf=0.0;
	stddev_NF_dnf=0.0;
	stddevdist_th=0.0;
	stddevdist_th_dnf=0.0;
	stddevdist_FI=0.0;
	stddevdist_FI_dnf=0.0;
	stddevdist_NI=0.0;
	stddevdist_NF=0.0;
	stddevdist_NI_dnf=0.0;
	stddevdist_NF_dnf=0.0;
	avg2_th=0.0;
	avg2_th_dnf=0.0;
	avg2_FI=0.0;
	avg2_FI_dnf=0.0;
	avg2_NI=0.0;
	avg2_NF=0.0;
	avg2_NI_dnf=0.0;
	avg2_NF_dnf=0.0;
	avgdist2_th=0.0;
	avgdist2_th_dnf=0.0;
	avgdist2_FI=0.0;
	avgdist2_FI_dnf=0.0;
	avgdist2_NI=0.0;
	avgdist2_NF=0.0;
	avgdist2_NI_dnf=0.0;
	avgdist2_NF_dnf=0.0;
	stddev2_th=0.0;
	stddev2_th_dnf=0.0;
	stddev2_FI=0.0;
	stddev2_FI_dnf=0.0;
	stddev2_NI=0.0;
	stddev2_NF=0.0;
	stddev2_NI_dnf=0.0;
	stddev2_NF_dnf=0.0;
	stddevdist2_th=0.0;
	stddevdist2_th_dnf=0.0;
	stddevdist2_FI=0.0;
	stddevdist2_FI_dnf=0.0;
	stddevdist2_NI=0.0;
	stddevdist2_NF=0.0;
	stddevdist2_NI_dnf=0.0;
	stddevdist2_NF_dnf=0.0;
	for (e=0;e<nb_experiments;e++) {
	  for (m=0;m<NBMAPS;m++) {
	    Kohonen map2=copy(map[m]);
	    Kohonen map2_th=copy(map_th[m]);
	    Kohonen map2_FI=copy(map_FI[m]);
	    Kohonen map2_NI=copy(map_NI[m]);
	    Kohonen map2_NF=copy(map_NF[m]);
	    // class decision according to each learning method
	    neuronclasses(map2,in,classe,crossvalid,testbloc,inp);
	    neuronclasses(map2_th,in,classe_th,crossvalid,testbloc,inp);
	    neuronclasses(map2_FI,in,classe_FI,crossvalid,testbloc,inp);
	    neuronclasses(map2_NI,in,classe_NI,crossvalid,testbloc,inp);
	    neuronclasses(map2_NF,in,classe_NF,crossvalid,testbloc,inp);
	    // introduction of faults in the copies of the pre-learned maps
	    faulty_weights(map2,p);
	    faulty_weights(map2_th,p);
	    faulty_weights(map2_FI,p);
	    faulty_weights(map2_NI,p);
	    faulty_weights(map2_NF,p);
	    // computation of error rate on learn partition
	    cnt=0;
	    cnt_dnf=0.0;
	    dist=0;
	    dist_dnf=0.0;
	    cnt_th=0;
	    cnt_th_dnf=0.0;
	    cnt_FI=0;
	    cnt_FI_dnf=0.0;
	    cnt_NI=0;
	    cnt_NF=0;
	    cnt_NI_dnf=0.0;
	    cnt_NF_dnf=0.0;
	    dist_th=0;
	    dist_th_dnf=0.0;
	    dist_FI=0;
	    dist_FI_dnf=0.0;
	    dist_NI=0;
	    dist_NF=0;
	    dist_NI_dnf=0.0;
	    dist_NF_dnf=0.0;
	    for (i=0;i<TESTDIV-1;i++) {
	      // all blocks except testbloc are used
	      if (i==testbloc) i++;
	      for (j=0;j<(inp/TESTDIV);j++) {
		Winner win=recall(map[m],in[crossvalid[i][j]]);
		WinnerDNF win_dnf=recallDNF(map[m],in[crossvalid[i][j]]);
		Winner win_th=recall(map_th[m],in[crossvalid[i][j]]);
		WinnerDNF win_th_dnf=recallDNF(map_th[m],in[crossvalid[i][j]]);
		Winner win_FI=recall(map_FI[m],in[crossvalid[i][j]]);
		WinnerDNF win_FI_dnf=recallDNF(map_FI[m],in[crossvalid[i][j]]);
		Winner win_NI=recall(map_NI[m],in[crossvalid[i][j]]);
		Winner win_NF=recall(map_NF[m],in[crossvalid[i][j]]);
		WinnerDNF win_NI_dnf=recallDNF(map_NI[m],in[crossvalid[i][j]]);
		WinnerDNF win_NF_dnf=recallDNF(map_NF[m],in[crossvalid[i][j]]);
		Winner win2=recall(map2,in[crossvalid[i][j]]);
		WinnerDNF win2_dnf=recallDNF(map2,in[crossvalid[i][j]]);
		Winner win2_th=recall(map2_th,in[crossvalid[i][j]]);
		WinnerDNF win2_th_dnf=recallDNF(map2_th,in[crossvalid[i][j]]);
		Winner win2_FI=recall(map2_FI,in[crossvalid[i][j]]);
		WinnerDNF win2_FI_dnf=recallDNF(map2_FI,in[crossvalid[i][j]]);
		Winner win2_NI=recall(map2_NI,in[crossvalid[i][j]]);
		Winner win2_NF=recall(map2_NF,in[crossvalid[i][j]]);
		WinnerDNF win2_NI_dnf=recallDNF(map2_NI,in[crossvalid[i][j]]);
		WinnerDNF win2_NF_dnf=recallDNF(map2_NF,in[crossvalid[i][j]]);
		if (classe[win2.i][win2.j]!=in[crossvalid[i][j]][INS]) cnt++;
		dist+=abs(win2.i-win.i)+abs(win2.j-win.j);
		if (classe_th[win2_th.i][win2_th.j]!=in[crossvalid[i][j]][INS]) cnt_th++;
		if (classe_FI[win2_FI.i][win2_FI.j]!=in[crossvalid[i][j]][INS]) cnt_FI++;
		if (classe_NI[win2_NI.i][win2_NI.j]!=in[crossvalid[i][j]][INS]) cnt_NI++;
		if (classe_NF[win2_NF.i][win2_NF.j]!=in[crossvalid[i][j]][INS]) cnt_NF++;
		dist_th+=abs(win2_th.i-win_th.i)+abs(win2_th.j-win_th.j);
		dist_FI+=abs(win2_FI.i-win_FI.i)+abs(win2_FI.j-win_FI.j);
		dist_NI+=abs(win2_NI.i-win_NI.i)+abs(win2_NI.j-win_NI.j);
		dist_NF+=abs(win2_NF.i-win_NF.i)+abs(win2_NF.j-win_NF.j);
		if (DNFclass(map2,in,crossvalid,testbloc,inp)!=in[crossvalid[i][j]][INS]) cnt_dnf++;
		if (DNFclass(map2_th,in,crossvalid,testbloc,inp)!=in[crossvalid[i][j]][INS]) cnt_th_dnf++;
		if (DNFclass(map2_FI,in,crossvalid,testbloc,inp)!=in[crossvalid[i][j]][INS]) cnt_FI_dnf++;
		if (DNFclass(map2_NI,in,crossvalid,testbloc,inp)!=in[crossvalid[i][j]][INS]) cnt_NI_dnf++;
		if (DNFclass(map2_NF,in,crossvalid,testbloc,inp)!=in[crossvalid[i][j]][INS]) cnt_NF_dnf++;
		dist_dnf+=fabs(win2_dnf.i-win_dnf.i)+fabs(win2_dnf.j-win_dnf.j);
		dist_th_dnf+=fabs(win2_th_dnf.i-win_th_dnf.i)+fabs(win2_th_dnf.j-win_th_dnf.j);
		dist_FI_dnf+=fabs(win2_FI_dnf.i-win_FI_dnf.i)+fabs(win2_FI_dnf.j-win_FI_dnf.j);
		dist_NI_dnf+=fabs(win2_NI_dnf.i-win_NI_dnf.i)+fabs(win2_NI_dnf.j-win_NI_dnf.j);
		dist_NF_dnf+=fabs(win2_NF_dnf.i-win_NF_dnf.i)+fabs(win2_NF_dnf.j-win_NF_dnf.j);
	      }
	    }
	    float faults=cnt/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float addeddist=dist/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    avg+=faults;
	    avgdist+=addeddist;
	    stddev+=faults*faults;
	    stddevdist+=addeddist*addeddist;
	    float faults_th=cnt_th/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float faults_FI=cnt_FI/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float faults_NI=cnt_NI/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float faults_NF=cnt_NF/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float addeddist_th=dist_th/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    float addeddist_FI=dist_FI/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    float addeddist_NI=dist_NI/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    float addeddist_NF=dist_NF/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    avg_th+=faults_th;
	    avg_FI+=faults_FI;
	    avg_NI+=faults_NI;
	    avg_NF+=faults_NF;
	    avgdist_th+=addeddist_th;
	    avgdist_FI+=addeddist_FI;
	    avgdist_NI+=addeddist_NI;
	    avgdist_NF+=addeddist_NF;
	    stddev_th+=faults_th*faults_th;
	    stddev_FI+=faults_FI*faults_FI;
	    stddev_NI+=faults_NI*faults_NI;
	    stddev_NF+=faults_NF*faults_NF;
	    stddevdist_th+=addeddist_th*addeddist_th;
	    stddevdist_FI+=addeddist_FI*addeddist_FI;
	    stddevdist_NI+=addeddist_NI*addeddist_NI;
	    stddevdist_NF+=addeddist_NF*addeddist_NF;
	    float faults_dnf=cnt_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float addeddist_dnf=dist_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    avg_dnf+=faults_dnf;
	    avgdist_dnf+=addeddist_dnf;
	    stddev_dnf+=faults_dnf*faults_dnf;
	    stddevdist_dnf+=addeddist_dnf*addeddist_dnf;
	    float faults_th_dnf=cnt_th_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float faults_FI_dnf=cnt_FI_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float faults_NI_dnf=cnt_NI_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float faults_NF_dnf=cnt_NF_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    float addeddist_th_dnf=dist_th_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    float addeddist_FI_dnf=dist_FI_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    float addeddist_NI_dnf=dist_NI_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    float addeddist_NF_dnf=dist_NF_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    avg_th_dnf+=faults_th_dnf;
	    avg_FI_dnf+=faults_FI_dnf;
	    avg_NI_dnf+=faults_NI_dnf;
	    avg_NF_dnf+=faults_NF_dnf;
	    avgdist_th_dnf+=addeddist_th_dnf;
	    avgdist_FI_dnf+=addeddist_FI_dnf;
	    avgdist_NI_dnf+=addeddist_NI_dnf;
	    avgdist_NF_dnf+=addeddist_NF_dnf;
	    stddev_th_dnf+=faults_th_dnf*faults_th_dnf;
	    stddev_FI_dnf+=faults_FI_dnf*faults_FI_dnf;
	    stddev_NI_dnf+=faults_NI_dnf*faults_NI_dnf;
	    stddev_NF_dnf+=faults_NF_dnf*faults_NF_dnf;
	    stddevdist_th_dnf+=addeddist_th_dnf*addeddist_th_dnf;
	    stddevdist_FI_dnf+=addeddist_FI_dnf*addeddist_FI_dnf;
	    stddevdist_NI_dnf+=addeddist_NI_dnf*addeddist_NI_dnf;
	    stddevdist_NF_dnf+=addeddist_NF_dnf*addeddist_NF_dnf;
	    if (VERBOSE) {
	      printf("validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults,addeddist);
	      printf("(DNF) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_dnf,addeddist_dnf);
	      printf("(threshold) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_th,addeddist_th);
	      printf("(DNF threshold) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_th_dnf,addeddist_th_dnf);
	      printf("(fault injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_FI,addeddist_FI);
	      printf("(DNF fault injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_FI_dnf,addeddist_FI_dnf);
	      printf("(noise injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_NI,addeddist_NI);
	      printf("(driven by NF) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_NF,addeddist_NF);
	      printf("(DNF noise injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_NI_dnf,addeddist_NI_dnf);
	      printf("(DNF driven by NF) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, LEARN error avg=%f, LEARN dist avgdist=%f\n",v,testbloc,p,e,m,faults_NF_dnf,addeddist_NF_dnf);
	    }
	    // computation of the error rate with test database
	    cnt=0;
	    cnt_dnf=0.0;
	    dist=0;
	    dist_dnf=0.0;
	    cnt_th=0;
	    cnt_th_dnf=0.0;
	    cnt_FI=0;
	    cnt_FI_dnf=0.0;
	    cnt_NI=0;
	    cnt_NF=0;
	    cnt_NI_dnf=0.0;
	    cnt_NF_dnf=0.0;
	    dist_th=0;
	    dist_th_dnf=0.0;
	    dist_FI=0;
	    dist_FI_dnf=0.0;
	    dist_NI=0;
	    dist_NF=0;
	    dist_NI_dnf=0.0;
	    dist_NF_dnf=0.0;
	    for (j=0;j<(inp/TESTDIV);j++) {
	      Winner win=recall(map[m],in[crossvalid[testbloc][j]]);
	      WinnerDNF win_dnf=recallDNF(map[m],in[crossvalid[testbloc][j]]);
	      Winner win_th=recall(map_th[m],in[crossvalid[testbloc][j]]);
	      WinnerDNF win_th_dnf=recallDNF(map_th[m],in[crossvalid[testbloc][j]]);
	      Winner win_FI=recall(map_FI[m],in[crossvalid[testbloc][j]]);
	      WinnerDNF win_FI_dnf=recallDNF(map_FI[m],in[crossvalid[testbloc][j]]);
	      Winner win_NI=recall(map_NI[m],in[crossvalid[testbloc][j]]);
	      Winner win_NF=recall(map_NF[m],in[crossvalid[testbloc][j]]);
	      WinnerDNF win_NI_dnf=recallDNF(map_NI[m],in[crossvalid[testbloc][j]]);
	      WinnerDNF win_NF_dnf=recallDNF(map_NF[m],in[crossvalid[testbloc][j]]);
	      Winner win2=recall(map2,in[crossvalid[testbloc][j]]);
	      WinnerDNF win2_dnf=recallDNF(map2,in[crossvalid[testbloc][j]]);
	      Winner win2_th=recall(map2_th,in[crossvalid[testbloc][j]]);
	      WinnerDNF win2_th_dnf=recallDNF(map2_th,in[crossvalid[testbloc][j]]);
	      Winner win2_FI=recall(map2_FI,in[crossvalid[testbloc][j]]);
	      WinnerDNF win2_FI_dnf=recallDNF(map2_FI,in[crossvalid[testbloc][j]]);
	      Winner win2_NI=recall(map2_NI,in[crossvalid[testbloc][j]]);
	      Winner win2_NF=recall(map2_NF,in[crossvalid[testbloc][j]]);
	      WinnerDNF win2_NI_dnf=recallDNF(map2_NI,in[crossvalid[testbloc][j]]);
	      WinnerDNF win2_NF_dnf=recallDNF(map2_NF,in[crossvalid[testbloc][j]]);
	      if (classe[win2.i][win2.j]!=in[crossvalid[testbloc][j]][INS]) cnt++;
	      dist+=abs(win2.i-win.i)+abs(win2.j-win.j);
	      if (classe_th[win2_th.i][win2_th.j]!=in[crossvalid[testbloc][j]][INS]) cnt_th++;
	      if (classe_FI[win2_FI.i][win2_FI.j]!=in[crossvalid[testbloc][j]][INS]) cnt_FI++;
	      if (classe_NI[win2_NI.i][win2_NI.j]!=in[crossvalid[testbloc][j]][INS]) cnt_NI++;
	      if (classe_NF[win2_NF.i][win2_NF.j]!=in[crossvalid[testbloc][j]][INS]) cnt_NF++;
	      dist_th+=abs(win2_th.i-win_th.i)+abs(win2_th.j-win_th.j);
	      dist_FI+=abs(win2_FI.i-win_FI.i)+abs(win2_FI.j-win_FI.j);
	      dist_NI+=abs(win2_NI.i-win_NI.i)+abs(win2_NI.j-win_NI.j);
	      dist_NF+=abs(win2_NF.i-win_NF.i)+abs(win2_NF.j-win_NF.j);
	      if (DNFclass(map2,in,crossvalid,testbloc,inp)!=in[crossvalid[testbloc][j]][INS]) cnt_dnf++;
	      if (DNFclass(map2_th,in,crossvalid,testbloc,inp)!=in[crossvalid[testbloc][j]][INS]) cnt_th_dnf++;
	      if (DNFclass(map2_FI,in,crossvalid,testbloc,inp)!=in[crossvalid[testbloc][j]][INS]) cnt_FI_dnf++;
	      if (DNFclass(map2_NI,in,crossvalid,testbloc,inp)!=in[crossvalid[testbloc][j]][INS]) cnt_NI_dnf++;
	      if (DNFclass(map2_NF,in,crossvalid,testbloc,inp)!=in[crossvalid[testbloc][j]][INS]) cnt_NF_dnf++;
	      dist_dnf+=fabs(win2_dnf.i-win_dnf.i)+fabs(win2_dnf.j-win_dnf.j);
	      dist_th_dnf+=fabs(win2_th_dnf.i-win_th_dnf.i)+fabs(win2_th_dnf.j-win_th_dnf.j);
	      dist_FI_dnf+=fabs(win2_FI_dnf.i-win_FI_dnf.i)+fabs(win2_FI_dnf.j-win_FI_dnf.j);
	      dist_NI_dnf+=fabs(win2_NI_dnf.i-win_NI_dnf.i)+fabs(win2_NI_dnf.j-win_NI_dnf.j);
	      dist_NF_dnf+=fabs(win2_NF_dnf.i-win_NF_dnf.i)+fabs(win2_NF_dnf.j-win_NF_dnf.j);
	    }
	    faults=cnt/(1.0*inp/TESTDIV);
	    addeddist=dist/(1.0*inp*2*SIZE/TESTDIV);
	    avg2+=faults;
	    avgdist2+=addeddist;
	    stddev2+=faults*faults;
	    stddevdist2+=addeddist*addeddist;
	    faults_th=cnt_th/(1.0*inp/TESTDIV);
	    faults_FI=cnt_FI/(1.0*inp/TESTDIV);
	    faults_NI=cnt_NI/(1.0*inp/TESTDIV);
	    faults_NF=cnt_NF/(1.0*inp/TESTDIV);
	    addeddist_th=dist_th/(1.0*inp*2*SIZE/TESTDIV);
	    addeddist_FI=dist_FI/(1.0*inp*2*SIZE/TESTDIV);
	    addeddist_NI=dist_NI/(1.0*inp*2*SIZE/TESTDIV);
	    addeddist_NF=dist_NF/(1.0*inp*2*SIZE/TESTDIV);
	    avg2_th+=faults_th;
	    avg2_FI+=faults_FI;
	    avg2_NI+=faults_NI;
	    avg2_NF+=faults_NF;
	    avgdist2_th+=addeddist_th;
	    avgdist2_FI+=addeddist_FI;
	    avgdist2_NI+=addeddist_NI;
	    avgdist2_NF+=addeddist_NF;
	    stddev2_th+=faults_th*faults_th;
	    stddev2_FI+=faults_FI*faults_FI;
	    stddev2_NI+=faults_NI*faults_NI;
	    stddev2_NF+=faults_NF*faults_NF;
	    stddevdist2_th+=addeddist_th*addeddist_th;
	    stddevdist2_FI+=addeddist_FI*addeddist_FI;
	    stddevdist2_NI+=addeddist_NI*addeddist_NI;
	    stddevdist2_NF+=addeddist_NF*addeddist_NF;
	    faults_dnf=cnt_dnf/(1.0*inp/TESTDIV);
	    addeddist_dnf=dist_dnf/(1.0*inp*2*SIZE/TESTDIV);
	    avg2_dnf+=faults_dnf;
	    avgdist2_dnf+=addeddist_dnf;
	    stddev2_dnf+=faults_dnf*faults_dnf;
	    stddevdist2_dnf+=addeddist_dnf*addeddist_dnf;
	    faults_th_dnf=cnt_th_dnf/(1.0*inp/TESTDIV);
	    faults_FI_dnf=cnt_FI_dnf/(1.0*inp/TESTDIV);
	    faults_NI_dnf=cnt_NI_dnf/(1.0*inp/TESTDIV);
	    faults_NF_dnf=cnt_NF_dnf/(1.0*inp/TESTDIV);
	    addeddist_th_dnf=dist_th_dnf/(1.0*inp*2*SIZE/TESTDIV);
	    addeddist_FI_dnf=dist_FI_dnf/(1.0*inp*2*SIZE/TESTDIV);
	    addeddist_NI_dnf=dist_NI_dnf/(1.0*inp*2*SIZE/TESTDIV);
	    addeddist_NF_dnf=dist_NF_dnf/(1.0*inp*2*SIZE/TESTDIV);
	    avg2_th_dnf+=faults_th_dnf;
	    avg2_FI_dnf+=faults_FI_dnf;
	    avg2_NI_dnf+=faults_NI_dnf;
	    avg2_NF_dnf+=faults_NF_dnf;
	    avgdist2_th_dnf+=addeddist_th_dnf;
	    avgdist2_FI_dnf+=addeddist_FI_dnf;
	    avgdist2_NI_dnf+=addeddist_NI_dnf;
	    avgdist2_NF_dnf+=addeddist_NF_dnf;
	    stddev2_th_dnf+=faults_th_dnf*faults_th_dnf;
	    stddev2_FI_dnf+=faults_FI_dnf*faults_FI_dnf;
	    stddev2_NI_dnf+=faults_NI_dnf*faults_NI_dnf;
	    stddev2_NF_dnf+=faults_NF_dnf*faults_NF_dnf;
	    stddevdist2_th_dnf+=addeddist_th_dnf*addeddist_th_dnf;
	    stddevdist2_FI_dnf+=addeddist_FI_dnf*addeddist_FI_dnf;
	    stddevdist2_NI_dnf+=addeddist_NI_dnf*addeddist_NI_dnf;
	    stddevdist2_NF_dnf+=addeddist_NF_dnf*addeddist_NF_dnf;
	    if (VERBOSE) {
	      printf("validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults,addeddist);
	      printf("(DNF) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_dnf,addeddist_dnf);
	      printf("(threshold) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_th,addeddist_th);
	      printf("(DNF threshold) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_th_dnf,addeddist_th_dnf);
	      printf("(fault injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_FI,addeddist_FI);
	      printf("(DNF fault injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_FI_dnf,addeddist_FI_dnf);
	      printf("(noise injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_NI,addeddist_NI);
	      printf("(driven by NF) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_NF,addeddist_NF);
	      printf("(DNF noise injection) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_NI_dnf,addeddist_NI_dnf);
	      printf("(DNF driven by NF) validation partition number %d, testbloc %d, fault percent %d, expe %d, map number %d, TEST error avg=%f, TEST dist avgdist=%f\n",v,testbloc,p,e,m,faults_NF_dnf,addeddist_NF_dnf);
	    }
	  } // end loop on map initializations
	} // end loop on experiments (faulty versions)
	int tt=nb_experiments*NBMAPS;
	avg=avg/tt;
	avgdist=avgdist/tt;
	stddev=mysqrt((stddev-tt*avg*avg)/(tt-1)); //corrected sample standard deviation
	stddevdist=mysqrt((stddevdist-tt*avgdist*avgdist)/(tt-1)); //corrected sample standard deviation
	avg_th=avg_th/tt;
	avg_FI=avg_FI/tt;
	avg_NI=avg_NI/tt;
	avg_NF=avg_NF/tt;
	avgdist_th=avgdist_th/tt;
	avgdist_FI=avgdist_FI/tt;
	avgdist_NI=avgdist_NI/tt;
	avgdist_NF=avgdist_NF/tt;
	stddev_th=mysqrt((stddev_th-tt*avg_th*avg_th)/(tt-1)); //corrected sample standard deviation
	stddev_FI=mysqrt((stddev_FI-tt*avg_FI*avg_FI)/(tt-1)); //corrected sample standard deviation
	stddev_NI=mysqrt((stddev_NI-tt*avg_NI*avg_NI)/(tt-1)); //corrected sample standard deviation
	stddev_NF=mysqrt((stddev_NF-tt*avg_NF*avg_NF)/(tt-1)); //corrected sample standard deviation
	stddevdist_th=mysqrt((stddevdist_th-tt*avgdist_th*avgdist_th)/(tt-1)); //corrected sample standard deviation
	stddevdist_FI=mysqrt((stddevdist_FI-tt*avgdist_FI*avgdist_FI)/(tt-1)); //corrected sample standard deviation
	stddevdist_NI=mysqrt((stddevdist_NI-tt*avgdist_NI*avgdist_NI)/(tt-1)); //corrected sample standard deviation
	stddevdist_NF=mysqrt((stddevdist_NF-tt*avgdist_NF*avgdist_NF)/(tt-1)); //corrected sample standard deviation
	mainavg[p]+=avg;
	mainavgdist[p]+=avgdist;
	mainstddev[p]+=stddev;
	mainstddevdist[p]+=stddevdist;
	printf("for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg,stddev,avgdist,stddevdist);
	avg2=avg2/tt;
	avgdist2=avgdist2/tt;
	stddev2=mysqrt((stddev2-tt*avg2*avg2)/(tt-1)); //corrected sample standard deviation
	stddevdist2=mysqrt((stddevdist2-tt*avgdist2*avgdist2)/(tt-1)); //corrected sample standard deviation
	mainavg2[p]+=avg2;
	mainavgdist2[p]+=avgdist2;
	mainstddev2[p]+=stddev2;
	mainstddevdist2[p]+=stddevdist2;
	printf("for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg2,stddev2,avgdist2,stddevdist2);
	mainavg_th[p]+=avg_th;
	mainavg_FI[p]+=avg_FI;
	mainavg_NI[p]+=avg_NI;
	mainavg_NF[p]+=avg_NF;
	mainavgdist_th[p]+=avgdist_th;
	mainavgdist_FI[p]+=avgdist_FI;
	mainavgdist_NI[p]+=avgdist_NI;
	mainavgdist_NF[p]+=avgdist_NF;
	mainstddev_th[p]+=stddev_th;
	mainstddev_FI[p]+=stddev_FI;
	mainstddev_NI[p]+=stddev_NI;
	mainstddev_NF[p]+=stddev_NF;
	mainstddevdist_th[p]+=stddevdist_th;
	mainstddevdist_FI[p]+=stddevdist_FI;
	mainstddevdist_NI[p]+=stddevdist_NI;
	mainstddevdist_NF[p]+=stddevdist_NF;
	printf("(threshold) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_th,stddev_th,avgdist_th,stddevdist_th);
	printf("(fault injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_FI,stddev_FI,avgdist_FI,stddevdist_FI);
	printf("(noise injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_NI,stddev_NI,avgdist_NI,stddevdist_NI);
	printf("(driven by NF) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_NF,stddev_NF,avgdist_NF,stddevdist_NF);
	avg2_th=avg2_th/tt;
	avg2_FI=avg2_FI/tt;
	avg2_NI=avg2_NI/tt;
	avg2_NF=avg2_NF/tt;
	avgdist2_th=avgdist2_th/tt;
	avgdist2_FI=avgdist2_FI/tt;
	avgdist2_NI=avgdist2_NI/tt;
	avgdist2_NF=avgdist2_NF/tt;
	stddev2_th=mysqrt((stddev2_th-tt*avg2_th*avg2_th)/(tt-1)); //corrected sample standard deviation
	stddev2_FI=mysqrt((stddev2_FI-tt*avg2_FI*avg2_FI)/(tt-1)); //corrected sample standard deviation
	stddev2_NI=mysqrt((stddev2_NI-tt*avg2_NI*avg2_NI)/(tt-1)); //corrected sample standard deviation
	stddev2_NF=mysqrt((stddev2_NF-tt*avg2_NF*avg2_NF)/(tt-1)); //corrected sample standard deviation
	stddevdist2_th=mysqrt((stddevdist2_th-tt*avgdist2_th*avgdist2_th)/(tt-1)); //corrected sample standard deviation
	stddevdist2_FI=mysqrt((stddevdist2_FI-tt*avgdist2_FI*avgdist2_FI)/(tt-1)); //corrected sample standard deviation
	stddevdist2_NI=mysqrt((stddevdist2_NI-tt*avgdist2_NI*avgdist2_NI)/(tt-1)); //corrected sample standard deviation
	stddevdist2_NF=mysqrt((stddevdist2_NF-tt*avgdist2_NF*avgdist2_NF)/(tt-1)); //corrected sample standard deviation
	mainavg2_th[p]+=avg2_th;
	mainavg2_FI[p]+=avg2_FI;
	mainavg2_NI[p]+=avg2_NI;
	mainavg2_NF[p]+=avg2_NF;
	mainavgdist2_th[p]+=avgdist2_th;
	mainavgdist2_FI[p]+=avgdist2_FI;
	mainavgdist2_NI[p]+=avgdist2_NI;
	mainavgdist2_NF[p]+=avgdist2_NF;
	mainstddev2_th[p]+=stddev2_th;
	mainstddev2_FI[p]+=stddev2_FI;
	mainstddev2_NI[p]+=stddev2_NI;
	mainstddev2_NF[p]+=stddev2_NF;
	mainstddevdist2_th[p]+=stddevdist2_th;
	mainstddevdist2_FI[p]+=stddevdist2_FI;
	mainstddevdist2_NI[p]+=stddevdist2_NI;
	mainstddevdist2_NF[p]+=stddevdist2_NF;
	printf("(threshold) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg2_th,stddev2_th,avgdist2_th,stddevdist2_th);
	printf("(fault injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg2_FI,stddev2_FI,avgdist2_FI,stddevdist2_FI);
	printf("(noise injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,avg2_NI,stddev2_NI,avgdist2_NI,stddevdist2_NI);
	printf("(driven by NF) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,avg2_NF,stddev2_NF,avgdist2_NF,stddevdist2_NF);
	/* DNF */
	avg_dnf=avg_dnf/tt;
	avgdist_dnf=avgdist_dnf/tt;
	stddev_dnf=mysqrt((stddev_dnf-tt*avg_dnf*avg_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist_dnf=mysqrt((stddevdist_dnf-tt*avgdist_dnf*avgdist_dnf)/(tt-1)); //corrected sample standard deviation
	avg_th_dnf=avg_th_dnf/tt;
	avg_FI_dnf=avg_FI_dnf/tt;
	avg_NI_dnf=avg_NI_dnf/tt;
	avg_NF_dnf=avg_NF_dnf/tt;
	avgdist_th_dnf=avgdist_th_dnf/tt;
	avgdist_FI_dnf=avgdist_FI_dnf/tt;
	avgdist_NI_dnf=avgdist_NI_dnf/tt;
	avgdist_NF_dnf=avgdist_NF_dnf/tt;
	stddev_th_dnf=mysqrt((stddev_th_dnf-tt*avg_th_dnf*avg_th_dnf)/(tt-1)); //corrected sample standard deviation
	stddev_FI_dnf=mysqrt((stddev_FI_dnf-tt*avg_FI_dnf*avg_FI_dnf)/(tt-1)); //corrected sample standard deviation
	stddev_NI_dnf=mysqrt((stddev_NI_dnf-tt*avg_NI_dnf*avg_NI_dnf)/(tt-1)); //corrected sample standard deviation
	stddev_NF_dnf=mysqrt((stddev_NF_dnf-tt*avg_NF_dnf*avg_NF_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist_th_dnf=mysqrt((stddevdist_th_dnf-tt*avgdist_th_dnf*avgdist_th_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist_FI_dnf=mysqrt((stddevdist_FI_dnf-tt*avgdist_FI_dnf*avgdist_FI_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist_NI_dnf=mysqrt((stddevdist_NI_dnf-tt*avgdist_NI_dnf*avgdist_NI_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist_NF_dnf=mysqrt((stddevdist_NF_dnf-tt*avgdist_NF_dnf*avgdist_NF_dnf)/(tt-1)); //corrected sample standard deviation
	mainavg_dnf[p]+=avg_dnf;
	mainavgdist_dnf[p]+=avgdist_dnf;
	mainstddev_dnf[p]+=stddev_dnf;
	mainstddevdist_dnf[p]+=stddevdist_dnf;
	printf("(DNF) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_dnf,stddev_dnf,avgdist_dnf,stddevdist_dnf);
	avg2_dnf=avg2_dnf/tt;
	avgdist2_dnf=avgdist2_dnf/tt;
	stddev2_dnf=mysqrt((stddev2_dnf-tt*avg2_dnf*avg2_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist2_dnf=mysqrt((stddevdist2_dnf-tt*avgdist2_dnf*avgdist2_dnf)/(tt-1)); //corrected sample standard deviation
	mainavg2_dnf[p]+=avg2_dnf;
	mainavgdist2_dnf[p]+=avgdist2_dnf;
	mainstddev2_dnf[p]+=stddev2_dnf;
	mainstddevdist2_dnf[p]+=stddevdist2_dnf;
	printf("(DNF) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg2_dnf,stddev2_dnf,avgdist2_dnf,stddevdist2_dnf);
	mainavg_th_dnf[p]+=avg_th_dnf;
	mainavg_FI_dnf[p]+=avg_FI_dnf;
	mainavg_NI_dnf[p]+=avg_NI_dnf;
	mainavg_NF_dnf[p]+=avg_NF_dnf;
	mainavgdist_th_dnf[p]+=avgdist_th_dnf;
	mainavgdist_FI_dnf[p]+=avgdist_FI_dnf;
	mainavgdist_NI_dnf[p]+=avgdist_NI_dnf;
	mainavgdist_NF_dnf[p]+=avgdist_NF_dnf;
	mainstddev_th_dnf[p]+=stddev_th_dnf;
	mainstddev_FI_dnf[p]+=stddev_FI_dnf;
	mainstddev_NI_dnf[p]+=stddev_NI_dnf;
	mainstddev_NF_dnf[p]+=stddev_NF_dnf;
	mainstddevdist_th_dnf[p]+=stddevdist_th_dnf;
	mainstddevdist_FI_dnf[p]+=stddevdist_FI_dnf;
	mainstddevdist_NI_dnf[p]+=stddevdist_NI_dnf;
	mainstddevdist_NF_dnf[p]+=stddevdist_NF_dnf;
	printf("(DNF threshold) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_th_dnf,stddev_th_dnf,avgdist_th_dnf,stddevdist_th_dnf);
	printf("(DNF fault injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg_FI_dnf,stddev_FI_dnf,avgdist_FI_dnf,stddevdist_FI_dnf);
	printf("(DNF noise injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,avg_NI_dnf,stddev_NI_dnf,avgdist_NI_dnf,stddevdist_NI_dnf);
	printf("(DNF driven by NF) for %d percents of flipped bits in a %dx%d map with %d inputs, average of LEARN error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,avg_NF_dnf,stddev_NF_dnf,avgdist_NF_dnf,stddevdist_NF_dnf);
	avg2_th_dnf=avg2_th_dnf/tt;
	avg2_FI_dnf=avg2_FI_dnf/tt;
	avg2_NI_dnf=avg2_NI_dnf/tt;
	avg2_NF_dnf=avg2_NF_dnf/tt;
	avgdist2_th_dnf=avgdist2_th_dnf/tt;
	avgdist2_FI_dnf=avgdist2_FI_dnf/tt;
	avgdist2_NI_dnf=avgdist2_NI_dnf/tt;
	avgdist2_NF_dnf=avgdist2_NF_dnf/tt;
	stddev2_th_dnf=mysqrt((stddev2_th_dnf-tt*avg2_th_dnf*avg2_th_dnf)/(tt-1)); //corrected sample standard deviation
	stddev2_FI_dnf=mysqrt((stddev2_FI_dnf-tt*avg2_FI_dnf*avg2_FI_dnf)/(tt-1)); //corrected sample standard deviation
	stddev2_NI_dnf=mysqrt((stddev2_NI_dnf-tt*avg2_NI_dnf*avg2_NI_dnf)/(tt-1)); //corrected sample standard deviation
	stddev2_NF_dnf=mysqrt((stddev2_NF_dnf-tt*avg2_NF_dnf*avg2_NF_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist2_th_dnf=mysqrt((stddevdist2_th_dnf-tt*avgdist2_th_dnf*avgdist2_th_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist2_FI_dnf=mysqrt((stddevdist2_FI_dnf-tt*avgdist2_FI_dnf*avgdist2_FI_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist2_NI_dnf=mysqrt((stddevdist2_NI_dnf-tt*avgdist2_NI_dnf*avgdist2_NI_dnf)/(tt-1)); //corrected sample standard deviation
	stddevdist2_NF_dnf=mysqrt((stddevdist2_NF_dnf-tt*avgdist2_NF_dnf*avgdist2_NF_dnf)/(tt-1)); //corrected sample standard deviation
	mainavg2_th_dnf[p]+=avg2_th_dnf;
	mainavg2_FI_dnf[p]+=avg2_FI_dnf;
	mainavg2_NI_dnf[p]+=avg2_NI_dnf;
	mainavg2_NF_dnf[p]+=avg2_NF_dnf;
	mainavgdist2_th_dnf[p]+=avgdist2_th_dnf;
	mainavgdist2_FI_dnf[p]+=avgdist2_FI_dnf;
	mainavgdist2_NI_dnf[p]+=avgdist2_NI_dnf;
	mainavgdist2_NF_dnf[p]+=avgdist2_NF_dnf;
	mainstddev2_th_dnf[p]+=stddev2_th_dnf;
	mainstddev2_FI_dnf[p]+=stddev2_FI_dnf;
	mainstddev2_NI_dnf[p]+=stddev2_NI_dnf;
	mainstddev2_NF_dnf[p]+=stddev2_NF_dnf;
	mainstddevdist2_th_dnf[p]+=stddevdist2_th_dnf;
	mainstddevdist2_FI_dnf[p]+=stddevdist2_FI_dnf;
	mainstddevdist2_NI_dnf[p]+=stddevdist2_NI_dnf;
	mainstddevdist2_NF_dnf[p]+=stddevdist2_NF_dnf;
	printf("(DNF threshold) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg2_th_dnf,stddev2_th_dnf,avgdist2_th_dnf,stddevdist2_th_dnf);
	printf("(DNF fault injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,avg2_FI_dnf,stddev2_FI_dnf,avgdist2_FI_dnf,stddevdist2_FI_dnf);
	printf("(DNF noise injection) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,avg2_NI_dnf,stddev2_NI_dnf,avgdist2_NI_dnf,stddevdist2_NI_dnf);
	printf("(DNF driven by NF) for %d percents of flipped bits in a %dx%d map with %d inputs, average of TEST error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,avg2_NF_dnf,stddev2_NF_dnf,avgdist2_NF_dnf,stddevdist2_NF_dnf);
      } // end loop on fault percentages
      for (m=0;m<NBMAPS;m++) {
	freeMap(map[m]);
	freeMap(map_th[m]);
	freeMap(map_FI[m]);
	freeMap(map_NI[m]);
	freeMap(map_NF[m]);
      }
    } // end loop on testblocs
    // in case of learning study alone
    exit(1);
    for (p=0;p<MAXFAULTPERCENT;p++) {
      printf("GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg[p]/TESTDIV,mainstddev[p]/TESTDIV,mainavgdist[p]/TESTDIV,mainstddevdist[p]/TESTDIV);
      printf("(DNF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_dnf[p]/TESTDIV,mainstddev_dnf[p]/TESTDIV,mainavgdist_dnf[p]/TESTDIV,mainstddevdist_dnf[p]/TESTDIV);
      printf("(threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_th[p]/TESTDIV,mainstddev_th[p]/TESTDIV,mainavgdist_th[p]/TESTDIV,mainstddevdist_th[p]/TESTDIV);
      printf("(DNF threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_th_dnf[p]/TESTDIV,mainstddev_th_dnf[p]/TESTDIV,mainavgdist_th_dnf[p]/TESTDIV,mainstddevdist_th_dnf[p]/TESTDIV);
      printf("(fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_FI[p]/TESTDIV,mainstddev_FI[p]/TESTDIV,mainavgdist_FI[p]/TESTDIV,mainstddevdist_FI[p]/TESTDIV);
      printf("(DNF fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_FI_dnf[p]/TESTDIV,mainstddev_FI_dnf[p]/TESTDIV,mainavgdist_FI_dnf[p]/TESTDIV,mainstddevdist_FI_dnf[p]/TESTDIV);
      printf("(noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_NI[p]/TESTDIV,mainstddev_NI[p]/TESTDIV,mainavgdist_NI[p]/TESTDIV,mainstddevdist_NI[p]/TESTDIV);
      printf("(driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_NF[p]/TESTDIV,mainstddev_NF[p]/TESTDIV,mainavgdist_NF[p]/TESTDIV,mainstddevdist_NF[p]/TESTDIV);
      printf("(DNF noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg_NI_dnf[p]/TESTDIV,mainstddev_NI_dnf[p]/TESTDIV,mainavgdist_NI_dnf[p]/TESTDIV,mainstddevdist_NI_dnf[p]/TESTDIV);
      printf("(DNF driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg_NF_dnf[p]/TESTDIV,mainstddev_NF_dnf[p]/TESTDIV,mainavgdist_NF_dnf[p]/TESTDIV,mainstddevdist_NF_dnf[p]/TESTDIV);
      printf("GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2[p]/TESTDIV,mainstddev2[p]/TESTDIV,mainavgdist2[p]/TESTDIV,mainstddevdist2[p]/TESTDIV);
      printf("(DNF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_dnf[p]/TESTDIV,mainstddev2_dnf[p]/TESTDIV,mainavgdist2_dnf[p]/TESTDIV,mainstddevdist2_dnf[p]/TESTDIV);
      printf("(threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_th[p]/TESTDIV,mainstddev2_th[p]/TESTDIV,mainavgdist2_th[p]/TESTDIV,mainstddevdist2_th[p]/TESTDIV);
      printf("(DNF threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_th_dnf[p]/TESTDIV,mainstddev2_th_dnf[p]/TESTDIV,mainavgdist2_th_dnf[p]/TESTDIV,mainstddevdist2_th_dnf[p]/TESTDIV);
      printf("(fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_FI[p]/TESTDIV,mainstddev2_FI[p]/TESTDIV,mainavgdist2_FI[p]/TESTDIV,mainstddevdist2_FI[p]/TESTDIV);
      printf("(DNF fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_FI_dnf[p]/TESTDIV,mainstddev2_FI_dnf[p]/TESTDIV,mainavgdist2_FI_dnf[p]/TESTDIV,mainstddevdist2_FI_dnf[p]/TESTDIV);
      printf("(noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_NI[p]/TESTDIV,mainstddev2_NI[p]/TESTDIV,mainavgdist2_NI[p]/TESTDIV,mainstddevdist2_NI[p]/TESTDIV);
      printf("(driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_NF[p]/TESTDIV,mainstddev2_NF[p]/TESTDIV,mainavgdist2_NF[p]/TESTDIV,mainstddevdist2_NF[p]/TESTDIV);
      printf("(DNF noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg2_NI_dnf[p]/TESTDIV,mainstddev2_NI_dnf[p]/TESTDIV,mainavgdist2_NI_dnf[p]/TESTDIV,mainstddevdist2_NI_dnf[p]/TESTDIV);
      printf("(DNF driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg2_NF_dnf[p]/TESTDIV,mainstddev2_NF_dnf[p]/TESTDIV,mainavgdist2_NF_dnf[p]/TESTDIV,mainstddevdist2_NF_dnf[p]/TESTDIV);
    } // end loop on GLOBAL printing for all fault percentages
  } // end loop on cross validation partitions
}
