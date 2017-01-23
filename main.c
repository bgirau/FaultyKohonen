
#include "func_def.h"
#include "gss_filter.h"
#include "gss_som.h"
#include "stat.h"

int main(){
	int v;
	srand(time(NULL));
	int  p,q,i,j,e,k,s,inp,m;

	Avgstddev av, av2, av_dnf, av2_dnf;
 	Avgstddev av_th, av2_th, av_th_dnf, av2_th_dnf;
 	Avgstddev av_FI, av2_FI, av_FI_dnf, av2_FI_dnf;
 	Avgstddev av_NI, av2_NI, av_NI_dnf, av2_NI_dnf;
 	Avgstddev av_NF, av2_NF, av_NF_dnf, av2_NF_dnf;

 	float x;
 	int ** in;
 	int ** count;
 	int ** classe  = (int **)malloc_2darray(SIZE,SIZE);
 	int ** classe_th = (int **)malloc_2darray(SIZE,SIZE);
 	int ** classe_FI = (int **)malloc_2darray(SIZE,SIZE);
	int ** classe_NI = (int **)malloc_2darray(SIZE,SIZE);
  int ** classe_NF = (int **)malloc_2darray(SIZE,SIZE);
  int ** classe2 = (int **)malloc_2darray(SIZE,SIZE);
  int ** classe_th2 = (int **)malloc_2darray(SIZE,SIZE);
  int ** classe_FI2 = (int **)malloc_2darray(SIZE,SIZE);
	int ** classe_NI2 = (int **)malloc_2darray(SIZE,SIZE);
  int ** classe_NF2 = (int **)malloc_2darray(SIZE,SIZE);
  char* st=(char*)malloc(1000*sizeof(char));
  Kohonen *map;
  Kohonen *map_th;
  Kohonen *map_FI;
  Kohonen *map_NI;
  Kohonen *map_NF;
  Kohonen *mapinit;

  inp = 0;
  FILE *f = fopen(INPUTFILENAME,"r");
  while (fscanf(f,"%s",st) == 1) inp++;
  fclose(f);
  inp /= (INS + 1);
  printf("nombre d'entrées = %d\n",inp);

  in = (int**)malloc(inp*sizeof(int*));
  f = fopen(INPUTFILENAME,"r");
  for (i = 0; i < inp; i++) {
    in[i]=(int*)malloc((INS+1)*sizeof(int));
    for (j = 0;j < INS;j++) {
      fscanf(f,"%f",&x);
      in[i][j] = (int)((1.0 * one) * x);
      if (abs(in[i][j]) > precision_int) {
				printf("warning: overflow\n");
				exit(1);
      }
    }
    fscanf(f,"%d",&p);
    in[i][INS] = p;
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

  // classe 			= malloc_2darray(SIZE,SIZE);
  // classe_th 	= malloc_2darray(SIZE,SIZE);
  // classe_FI 	= malloc_2darray(SIZE,SIZE);
  // classe_NI 	= malloc_2darray(SIZE,SIZE);
  // classe_NF 	= malloc_2darray(SIZE,SIZE);
  // classe2 		= malloc_2darray(SIZE,SIZE);
  // classe_th2 	= malloc_2darray(SIZE,SIZE);
  // classe_FI2 	= malloc_2darray(SIZE,SIZE);
  // classe_NI2 	= malloc_2darray(SIZE,SIZE);
  // classe_NF2 	= malloc_2darray(SIZE,SIZE);

  for (v=0;v<NBVALIDPARTITIONS;v++) {
    printf("\n*******************     \
    				*********************\n   \
    				***********************.  \
    				*************\nNEW CROSS  \
    				VALIDATION PARTITIONING\n \
    				------------------------- \
    				-------------\n\n");
    /* répartition base en test/validation/apprentissge 
    des 150 exemples IRIS */
    int **crossvalid = (int**) malloc_2darray(TESTDIV,inp/TESTDIV);
    int *select = malloc(inp * sizeof(int));
    int n;

    for (i=0;i<(inp/TESTDIV);i++) {
    	for (j=0;j<TESTDIV;j++) {
				n=rand()%inp;
				while (select[n]==1) n=rand()%inp;
				crossvalid[j][i]=n;
				select[n]=1;
      }
    }

    int testbloc;

    map 		= (Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_th 	= (Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_FI 	= (Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_NI	=	(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    map_NF	=	(Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    mapinit = (Kohonen*)malloc(NBMAPS*sizeof(Kohonen));
    
    for (i=0;i<NBMAPS;i++) mapinit[i] = init();
    
    int tmin,nmin,t;
    float d,dmin;

    Mainavgstddev mainav, mainav2, mainav_dnf, mainav2_dnf;
    Mainavgstddev mainav_th, mainav2_th, mainav_th_dnf, mainav2_th_dnf;
    Mainavgstddev mainav_FI, mainav2_FI, mainav_FI_dnf, mainav2_FI_dnf;
    Mainavgstddev mainav_NI, mainav2_NI, mainav_NI_dnf, mainav2_NI_dnf;
    Mainavgstddev mainav_NF, mainav2_NF, mainav_NF_dnf, mainav2_NF_dnf;
    /* mainav */
    Mainavgstddev_init(&mainav,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav_dnf,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_dnf,MAXFAULTPERCENT);
    /* mainav th*/
    Mainavgstddev_init(&mainav_th,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_th,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav_th_dnf,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_th_dnf,MAXFAULTPERCENT);
    /* mainav FI*/
    Mainavgstddev_init(&mainav_FI,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_FI,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav_FI_dnf,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_FI_dnf,MAXFAULTPERCENT);
		/* mainav NI*/
    Mainavgstddev_init(&mainav_NI,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_NI,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav_NI_dnf,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_NI_dnf,MAXFAULTPERCENT);
    /* mainav NF*/
    Mainavgstddev_init(&mainav_NF,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_NF,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav_NF_dnf,MAXFAULTPERCENT);
    Mainavgstddev_init(&mainav2_NF_dnf,MAXFAULTPERCENT);

    for (testbloc=0;testbloc<TESTDIV;testbloc++) {
      /* first step : learn all maps 
      	 (NBMAPS different initializations) by means 
      	 of all available algorithms without the current testbloc
      */
      for (m=0;m<NBMAPS;m++) {
      	printf("\n************** \
      		******************\n learning map number %d : \n\n",m);
      	map[m]		=	copy(mapinit[m]);
      	map_th[m]	=	copy(mapinit[m]);
      	map_FI[m]	=	copy(mapinit[m]);
      	map_NI[m]	=	copy(mapinit[m]);
      	map_NF[m]	=	copy(mapinit[m]);

      	printneuronclasses(map[m],in,classe,crossvalid,testbloc,inp);

      	printf("****************\nBefore learning\n");
      	errorrate(map[m],in,inp,classe,-1,crossvalid,testbloc);
      	errorrateGSS(map[m],in,inp,classe,-1,crossvalid,testbloc);
      	// errorrateDNF(map[m],in,inp,classe,-1,crossvalid,testbloc);
      	learn(map[m],in,inp,classe,crossvalid,testbloc);

      	printf("****************\nAfter standard learning\n");
      	errorrate(map[m],in,inp,classe,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	errorrateGSS(map[m],in,inp,classe,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	// errorrateDNF(map[m],in,inp,classe,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	learn_threshold(map_th[m],in,inp,classe_th,crossvalid,testbloc);

      	printf("****************\nAfter thresholded learning\n");
      	errorrate(map_th[m],in,inp,classe_th,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	errorrateGSS(map_th[m],in,inp,classe_th,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	// errorrateDNF(map_th[m],in,inp,classe_th,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	learn_FI(map_FI[m],in,inp,classe_FI,crossvalid,testbloc);

      	printf("****************\nAfter fault injection learning\n");
      	errorrate(map_FI[m],in,inp,classe_FI,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	errorrateGSS(map_FI[m],in,inp,classe_FI,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	// errorrateDNF(map_FI[m],in,inp,classe_FI,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	learn_NI(map_NI[m],in,inp,classe_NI,crossvalid,testbloc);

      	printf("****************\nAfter noise injection learning\n");
      	errorrate(map_NI[m],in,inp,classe_NI,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	errorrateGSS(map_NI[m],in,inp,classe_NI,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	// errorrateDNF(map_NI[m],in,inp,classe_NI,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	learn_NF(map_NF[m],in,inp,classe_NF,crossvalid,testbloc);
      	printf("****************\nAfter NF driven learning\n");
      	errorrate(map_NF[m],in,inp,classe_NF,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	errorrateGSS(map_NF[m],in,inp,classe_NF,
      						NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      	// errorrateDNF(map_NF[m],in,inp,classe_NF,NBITEREPOCH*NBEPOCHLEARN,crossvalid,testbloc);
      }
      break;

      float dist = 0.0, dist_th = 0.0;
      float dist_FI = 0.0;
      float dist_NI = 0.0;
      float dist_NF = 0.0;
      int cnt = 0;
      int cnt_th = 0;
      int cnt_FI = 0;
      int cnt_NI = 0;
      int cnt_NF = 0;
      float dist_dnf = 0.0, dist_th_dnf = 0.0;
      float dist_FI_dnf = 0.0;
      float dist_NI_dnf = 0.0;
      float dist_NF_dnf = 0.0;
      int cnt_th_dnf	= 0;
      int cnt_FI_dnf	= 0;
      int cnt_NI_dnf	= 0;
      int cnt_NF_dnf	= 0;
      int cnt_dnf			= 0;

      for (p=0;p<MAXFAULTPERCENT;p++) {
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

			      neuronclassesGSS(map2,in,classe2,crossvalid,testbloc,inp);
			      neuronclassesGSS(map2_th,in,classe_th2,crossvalid,testbloc,inp);
			      neuronclassesGSS(map2_FI,in,classe_FI2,crossvalid,testbloc,inp);
			      neuronclassesGSS(map2_NI,in,classe_NI2,crossvalid,testbloc,inp);
			      neuronclassesGSS(map2_NF,in,classe_NF2,crossvalid,testbloc,inp);
				    // introduction of faults in the copies of the pre-learned maps
				    faulty_weights(map2,p);
				    faulty_weights(map2_th,p);
				    faulty_weights(map2_FI,p);
				    faulty_weights(map2_NI,p);
				    faulty_weights(map2_NF,p);

				    for (i=0;i<TESTDIV-1;i++) {
	      			// all blocks except testbloc are used
	      			if (i==testbloc) i++;
	      			for (j=0;j<(inp/TESTDIV);j++) {
	      				Winner win=recall(map[m],in[crossvalid[i][j]]);
	      				Winner win_th=recall(map_th[m],in[crossvalid[i][j]]);
	      				Winner win_FI=recall(map_FI[m],in[crossvalid[i][j]]);
	      				Winner win_NI=recall(map_NI[m],in[crossvalid[i][j]]);
								Winner win_NF=recall(map_NF[m],in[crossvalid[i][j]]);

								Winner win_dnf=recallGSS(map[m],in[crossvalid[i][j]],1.0);
								Winner win_th_dnf=recallGSS(map_th[m],in[crossvalid[i][j]],1.0);
								Winner win_FI_dnf=recallGSS(map_FI[m],in[crossvalid[i][j]],1.0);
								Winner win_NI_dnf=recallGSS(map_NI[m],in[crossvalid[i][j]],1.0);
								Winner win_NF_dnf=recallGSS(map_NF[m],in[crossvalid[i][j]],1.0);

								Winner win2=recall(map2,in[crossvalid[i][j]]);
								Winner win2_th=recall(map2_th,in[crossvalid[i][j]]);
								Winner win2_FI=recall(map2_FI,in[crossvalid[i][j]]);
								Winner win2_NI=recall(map2_NI,in[crossvalid[i][j]]);
								Winner win2_NF=recall(map2_NF,in[crossvalid[i][j]]);

								Winner win2_dnf=recallGSS(map2,in[crossvalid[i][j]],1.0);
								Winner win2_th_dnf=recallGSS(map2_th,in[crossvalid[i][j]],1.0);
								Winner win2_FI_dnf=recallGSS(map2_FI,in[crossvalid[i][j]],1.0);
								Winner win2_NI_dnf=recallGSS(map2_NI,in[crossvalid[i][j]],1.0);
								Winner win2_NF_dnf=recallGSS(map2_NF,in[crossvalid[i][j]],1.0);

								if (classe[win2.i][win2.j]!=in[crossvalid[i][j]][INS]) cnt++;
								if (classe_th[win2_th.i][win2_th.j]!=in[crossvalid[i][j]][INS]) cnt_th++;
								if (classe_FI[win2_FI.i][win2_FI.j]!=in[crossvalid[i][j]][INS]) cnt_FI++;
								if (classe_NI[win2_NI.i][win2_NI.j]!=in[crossvalid[i][j]][INS]) cnt_NI++;
								if (classe_NF[win2_NF.i][win2_NF.j]!=in[crossvalid[i][j]][INS]) cnt_NF++;
								if (classe2[win2_dnf.i][win2_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_dnf++;
								if (classe_th2[win2_th_dnf.i][win2_th_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_th_dnf++;
								if (classe_FI2[win2_FI_dnf.i][win2_FI_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_FI_dnf++;
								if (classe_NI2[win2_NI_dnf.i][win2_NI_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_NI_dnf++;
								if (classe_NF2[win2_NF_dnf.i][win2_NF_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_NF_dnf++;
								
								dist+=abs(win2.i-win.i)+abs(win2.j-win.j);
								dist_th+=abs(win2_th.i-win_th.i)+abs(win2_th.j-win_th.j);
								dist_FI+=abs(win2_FI.i-win_FI.i)+abs(win2_FI.j-win_FI.j);
								dist_NI+=abs(win2_NI.i-win_NI.i)+abs(win2_NI.j-win_NI.j);
								dist_NF+=abs(win2_NF.i-win_NF.i)+abs(win2_NF.j-win_NF.j);
								
								dist_dnf+=fabs(win2_dnf.i-win_dnf.i)+fabs(win2_dnf.j-win_dnf.j);
								dist_th_dnf+=fabs(win2_th_dnf.i-win_th_dnf.i)+fabs(win2_th_dnf.j-win_th_dnf.j);
								dist_FI_dnf+=fabs(win2_FI_dnf.i-win_FI_dnf.i)+fabs(win2_FI_dnf.j-win_FI_dnf.j);
								dist_NI_dnf+=fabs(win2_NI_dnf.i-win_NI_dnf.i)+fabs(win2_NI_dnf.j-win_NI_dnf.j);
								dist_NF_dnf+=fabs(win2_NF_dnf.i-win_NF_dnf.i)+fabs(win2_NF_dnf.j-win_NF_dnf.j);
							}
						}
						float faults = cnt/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			float addeddist = dist/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av, faults, addeddist);

						float faults_th=cnt_th/(1.0*inp*(TESTDIV-1)/TESTDIV);
						float addeddist_th=dist_th/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av_th, faults_th, addeddist_th);

	    			float faults_FI=cnt_FI/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			float addeddist_FI=dist_FI/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    			acc_avgstddev(&av_FI, faults_FI, addeddist_FI);

	    			float faults_NI=cnt_NI/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			float addeddist_NI=dist_NI/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    			acc_avgstddev(&av_NI, faults_NI, addeddist_NI);

	    			float faults_NF=cnt_NF/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			float addeddist_NF=dist_NF/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av_NF, faults_NF, addeddist_NF);

	    			float faults_dnf=cnt_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    float addeddist_dnf=dist_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av_dnf, faults_dnf, addeddist_dnf);

				    float faults_th_dnf=cnt_th_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    float addeddist_th_dnf=dist_th_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av_th_dnf, faults_th_dnf, addeddist_th_dnf);

				    float faults_FI_dnf=cnt_FI_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    float addeddist_FI_dnf=dist_FI_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av_FI_dnf, faults_FI_dnf, addeddist_FI_dnf);

				    float faults_NI_dnf=cnt_NI_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    float addeddist_NI_dnf=dist_NI_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av_NI_dnf, faults_NI_dnf, addeddist_NI_dnf);

				    float faults_NF_dnf=cnt_NF_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    float addeddist_NF_dnf=dist_NF_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av_NF_dnf, faults_NF_dnf, addeddist_NF_dnf);
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
      				Winner win=recall(map[m],in[crossvalid[i][j]]);
      				Winner win_th=recall(map_th[m],in[crossvalid[i][j]]);
      				Winner win_FI=recall(map_FI[m],in[crossvalid[i][j]]);
      				Winner win_NI=recall(map_NI[m],in[crossvalid[i][j]]);
							Winner win_NF=recall(map_NF[m],in[crossvalid[i][j]]);

							Winner win_dnf=recallGSS(map[m],in[crossvalid[i][j]],1.0);
							Winner win_th_dnf=recallGSS(map_th[m],in[crossvalid[i][j]],1.0);
							Winner win_FI_dnf=recallGSS(map_FI[m],in[crossvalid[i][j]],1.0);
							Winner win_NI_dnf=recallGSS(map_NI[m],in[crossvalid[i][j]],1.0);
							Winner win_NF_dnf=recallGSS(map_NF[m],in[crossvalid[i][j]],1.0);

							Winner win2=recall(map2,in[crossvalid[i][j]]);
							Winner win2_th=recall(map2_th,in[crossvalid[i][j]]);
							Winner win2_FI=recall(map2_FI,in[crossvalid[i][j]]);
							Winner win2_NI=recall(map2_NI,in[crossvalid[i][j]]);
							Winner win2_NF=recall(map2_NF,in[crossvalid[i][j]]);

							Winner win2_dnf=recallGSS(map2,in[crossvalid[i][j]],1.0);
							Winner win2_th_dnf=recallGSS(map2_th,in[crossvalid[i][j]],1.0);
							Winner win2_FI_dnf=recallGSS(map2_FI,in[crossvalid[i][j]],1.0);
							Winner win2_NI_dnf=recallGSS(map2_NI,in[crossvalid[i][j]],1.0);
							Winner win2_NF_dnf=recallGSS(map2_NF,in[crossvalid[i][j]],1.0);

							if (classe[win2.i][win2.j]!=in[crossvalid[i][j]][INS]) cnt++;
							if (classe_th[win2_th.i][win2_th.j]!=in[crossvalid[i][j]][INS]) cnt_th++;
							if (classe_FI[win2_FI.i][win2_FI.j]!=in[crossvalid[i][j]][INS]) cnt_FI++;
							if (classe_NI[win2_NI.i][win2_NI.j]!=in[crossvalid[i][j]][INS]) cnt_NI++;
							if (classe_NF[win2_NF.i][win2_NF.j]!=in[crossvalid[i][j]][INS]) cnt_NF++;
							if (classe2[win2_dnf.i][win2_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_dnf++;
							if (classe_th2[win2_th_dnf.i][win2_th_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_th_dnf++;
							if (classe_FI2[win2_FI_dnf.i][win2_FI_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_FI_dnf++;
							if (classe_NI2[win2_NI_dnf.i][win2_NI_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_NI_dnf++;
							if (classe_NF2[win2_NF_dnf.i][win2_NF_dnf.j]!=in[crossvalid[i][j]][INS]) cnt_NF_dnf++;
							
							dist+=abs(win2.i-win.i)+abs(win2.j-win.j);
							dist_th+=abs(win2_th.i-win_th.i)+abs(win2_th.j-win_th.j);
							dist_FI+=abs(win2_FI.i-win_FI.i)+abs(win2_FI.j-win_FI.j);
							dist_NI+=abs(win2_NI.i-win_NI.i)+abs(win2_NI.j-win_NI.j);
							dist_NF+=abs(win2_NF.i-win_NF.i)+abs(win2_NF.j-win_NF.j);
							
							dist_dnf+=fabs(win2_dnf.i-win_dnf.i)+fabs(win2_dnf.j-win_dnf.j);
							dist_th_dnf+=fabs(win2_th_dnf.i-win_th_dnf.i)+fabs(win2_th_dnf.j-win_th_dnf.j);
							dist_FI_dnf+=fabs(win2_FI_dnf.i-win_FI_dnf.i)+fabs(win2_FI_dnf.j-win_FI_dnf.j);
							dist_NI_dnf+=fabs(win2_NI_dnf.i-win_NI_dnf.i)+fabs(win2_NI_dnf.j-win_NI_dnf.j);
							dist_NF_dnf+=fabs(win2_NF_dnf.i-win_NF_dnf.i)+fabs(win2_NF_dnf.j-win_NF_dnf.j);
						}
						faults = cnt/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			addeddist = dist/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av2, faults, addeddist);

						faults_th=cnt_th/(1.0*inp*(TESTDIV-1)/TESTDIV);
						addeddist_th=dist_th/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av2_th, faults_th, addeddist_th);

	    			faults_FI=cnt_FI/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			addeddist_FI=dist_FI/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    			acc_avgstddev(&av2_FI, faults_FI, addeddist_FI);

	    			faults_NI=cnt_NI/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			addeddist_NI=dist_NI/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
	    			acc_avgstddev(&av2_NI, faults_NI, addeddist_NI);

	    			faults_NF=cnt_NF/(1.0*inp*(TESTDIV-1)/TESTDIV);
	    			addeddist_NF=dist_NF/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av2_NF, faults_NF, addeddist_NF);

	    			faults_dnf=cnt_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    addeddist_dnf=dist_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av2_dnf, faults_dnf, addeddist_dnf);

				    faults_th_dnf=cnt_th_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    addeddist_th_dnf=dist_th_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
						acc_avgstddev(&av2_th_dnf, faults_th_dnf, addeddist_th_dnf);

				    faults_FI_dnf=cnt_FI_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    addeddist_FI_dnf=dist_FI_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av2_FI_dnf, faults_FI_dnf, addeddist_FI_dnf);

				    faults_NI_dnf=cnt_NI_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    addeddist_NI_dnf=dist_NI_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av2_NI_dnf, faults_NI_dnf, addeddist_NI_dnf);

				    faults_NF_dnf=cnt_NF_dnf/(1.0*inp*(TESTDIV-1)/TESTDIV);
				    addeddist_NF_dnf=dist_NF_dnf/(1.0*inp*2*SIZE*(TESTDIV-1)/TESTDIV);
				    acc_avgstddev(&av2_NF_dnf, faults_NF_dnf, addeddist_NF_dnf);

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
				calc_avgstddev(&av);
				calc_avgstddev(&av_th);
				calc_avgstddev(&av_FI);
				calc_avgstddev(&av_NI);
				calc_avgstddev(&av_NF);

				add_avgstddev_mainavgstddev(&mainav, av, p);
				add_avgstddev_mainavgstddev(&mainav_th, av_th, p);
				add_avgstddev_mainavgstddev(&mainav_FI, av_FI, p);
				add_avgstddev_mainavgstddev(&mainav_NI, av_NI, p);
				add_avgstddev_mainavgstddev(&mainav_NF, av_NF, p);

				printf("for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of LEARN error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f, \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av.avg,av.stddev,av.avgdist,av.stddevdist);

				printf("(threshold) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of LEARN error rate = %f, \
					standard deviation = %f, average of normalized Manhattan distance %f, \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_th.avg,av_th.stddev,av_th.avgdist,av_th.stddevdist);

				printf("(fault injection) for %d percents of flipped bits in a %dx%d \
					map with %d inputs, average of LEARN error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f, \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_FI.avg,av_FI.stddev,av_FI.avgdist,av_FI.stddevdist);

				printf("(noise injection) for %d percents of flipped bits in a %dx%d \
				 	map with %d inputs, average of LEARN error rate = %f, \
				 	standard deviation = %f, average of normalized Manhattan distance %f, \
				 	standard deviation %f\n", 
				 	p,SIZE,SIZE,INS,av_NI.avg,av_NI.stddev,av_NI.avgdist,av_NI.stddevdist);

				printf("(driven by NF) for %d percents of flipped bits in a %dx%d   \
					map with %d inputs, average of LEARN error rate = %f, \
					standard deviation = %f, average of normalized Manhatta n distance %f, \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_NF.avg,av_NF.stddev,av_NF.avgdist,av_NF.stddevdist);

				calc_avgstddev(&av2);
				calc_avgstddev(&av2_th);
				calc_avgstddev(&av2_FI);
				calc_avgstddev(&av2_NI);
				calc_avgstddev(&av2_NF);

				add_avgstddev_mainavgstddev(&mainav2, av2, p);
				add_avgstddev_mainavgstddev(&mainav2_th, av2_th, p);
				add_avgstddev_mainavgstddev(&mainav2_FI, av2_FI, p);
				add_avgstddev_mainavgstddev(&mainav2_NI, av2_NI, p);
				add_avgstddev_mainavgstddev(&mainav2_NF, av2_NF, p);

				printf("for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of TEST error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2.avg,av2.stddev,av2.avgdist,av2.stddevdist);

				printf("(threshold) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of TEST error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_th.avg,av2_th.stddev,av2_th.avgdist,av2_th.stddevdist);

				printf("(fault injection) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of TEST error rate = %f, \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_FI.avg,av2_FI.stddev,av2_FI.avgdist,av2_FI.stddevdist);

				printf("(noise injection) for %d percents of flipped bits in a %dx%d \
				 	map with %d inputs, average of TEST error rate = %f,  \
				 	standard deviation = %f, average of normalized Manhattan distance %f,  \
				 	standard deviation %f\n",
				 	p,SIZE,SIZE,INS,av2_NI.avg,av2_NI.stddev,av2_NI.avgdist,av2_NI.stddevdist);

				printf("(driven by NF) for %d percents of flipped bits in a %dx%d \
					map with %d inputs, average of TEST error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_NF.avg,av2_NF.stddev,av2_NF.avgdist,av2_NF.stddevdist);
				/* DNF */ 
				calc_avgstddev(&av_dnf);
				calc_avgstddev(&av_th_dnf);
				calc_avgstddev(&av_FI_dnf);
				calc_avgstddev(&av_NI_dnf);
				calc_avgstddev(&av_NF_dnf);

				add_avgstddev_mainavgstddev(&mainav_dnf, av_dnf, p);
				add_avgstddev_mainavgstddev(&mainav_th_dnf, av_th_dnf, p);
				add_avgstddev_mainavgstddev(&mainav_FI_dnf, av_FI_dnf, p);
				add_avgstddev_mainavgstddev(&mainav_NI_dnf, av_NI_dnf, p);
				add_avgstddev_mainavgstddev(&mainav_NF_dnf, av_NF_dnf, p);

				printf("(DNF) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of LEARN error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_dnf.avg,av_dnf.stddev,av_dnf.avgdist,av_dnf.stddevdist);

				printf("(DNF threshold) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of LEARN error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_th_dnf.avg,av_th_dnf.stddev,
					av_th_dnf.avgdist,av_th_dnf.stddevdist);

				printf("(DNF fault injection) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of LEARN error rate = %f, \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_FI_dnf.avg,av_FI_dnf.stddev,
					av_FI_dnf.avgdist,av_FI_dnf.stddevdist);

				printf("(DNF noise injection) for %d percents of flipped bits in a %dx%d \
				 	map with %d inputs, average of LEARN error rate = %f,  \
				 	standard deviation = %f, average of normalized Manhattan distance %f,  \
				 	standard deviation %f\n",
				 	p,SIZE,SIZE,INS,av_NI_dnf.avg,av_NI_dnf.stddev,
				 	av_NI_dnf.avgdist,av_NI_dnf.stddevdist);

				printf("(DNF driven by NF) for %d percents of flipped bits in a %dx%d \
					map with %d inputs, average of LEARN error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av_NF_dnf.avg,av_NF_dnf.stddev,
					av_NF_dnf.avgdist,av_NF_dnf.stddevdist);

				calc_avgstddev(&av2_dnf);
				calc_avgstddev(&av2_th_dnf);
				calc_avgstddev(&av2_FI_dnf);
				calc_avgstddev(&av2_NI_dnf);
				calc_avgstddev(&av2_NF_dnf);

				add_avgstddev_mainavgstddev(&mainav2_dnf, av2_dnf, p);
				add_avgstddev_mainavgstddev(&mainav2_th_dnf, av2_th_dnf, p);
				add_avgstddev_mainavgstddev(&mainav2_FI_dnf, av2_FI_dnf, p);
				add_avgstddev_mainavgstddev(&mainav2_NI_dnf, av2_NI_dnf, p);
				add_avgstddev_mainavgstddev(&mainav2_NF_dnf, av2_NF_dnf, p);

				printf("(DNF) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of TEST error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_dnf.avg,av2_dnf.stddev,
					av2_dnf.avgdist,av2_dnf.stddevdist);

				printf("(DNF threshold) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of TEST error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_th_dnf.avg,av2_th_dnf.stddev,
					av2_th_dnf.avgdist,av2_th_dnf.stddevdist);

				printf("(DNF fault injection) for %d percents of flipped bits in a %dx%d  \
					map with %d inputs, average of TEST error rate = %f, \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_FI_dnf.avg,av2_FI_dnf.stddev,
					av2_FI_dnf.avgdist,av2_FI_dnf.stddevdist);

				printf("(DNF noise injection) for %d percents of flipped bits in a %dx%d \
				 	map with %d inputs, average of TEST error rate = %f,  \
				 	standard deviation = %f, average of normalized Manhattan distance %f,  \
				 	standard deviation %f\n", 
				 	p,SIZE,SIZE,INS,av2_NI_dnf.avg,av2_NI_dnf.stddev,
				 	av2_NI_dnf.avgdist,av2_NI_dnf.stddevdist);

				printf("(DNF driven by NF) for %d percents of flipped bits in a %dx%d \
					map with %d inputs, average of TEST error rate = %f,  \
					standard deviation = %f, average of normalized Manhattan distance %f,  \
					standard deviation %f\n",
					p,SIZE,SIZE,INS,av2_NF_dnf.avg,av2_NF_dnf.stddev,
					av2_NF_dnf.avgdist,av2_NF_dnf.stddevdist);
			}
			for (m=0;m<NBMAPS;m++) {
				freeMap(map[m]);
				freeMap(map_th[m]);
				freeMap(map_FI[m]);
				freeMap(map_NI[m]);
				freeMap(map_NF[m]);
      }
    } // end loop on testblocs
    exit(1);
    // for (p=0;p<MAXFAULTPERCENT;p++) {
    //   printf("GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg[p]/TESTDIV,mainstddev[p]/TESTDIV,mainavgdist[p]/TESTDIV,mainstddevdist[p]/TESTDIV);
    //   printf("(DNF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_dnf[p]/TESTDIV,mainstddev_dnf[p]/TESTDIV,mainavgdist_dnf[p]/TESTDIV,mainstddevdist_dnf[p]/TESTDIV);
    //   printf("(threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_th[p]/TESTDIV,mainstddev_th[p]/TESTDIV,mainavgdist_th[p]/TESTDIV,mainstddevdist_th[p]/TESTDIV);
    //   printf("(DNF threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_th_dnf[p]/TESTDIV,mainstddev_th_dnf[p]/TESTDIV,mainavgdist_th_dnf[p]/TESTDIV,mainstddevdist_th_dnf[p]/TESTDIV);
    //   printf("(fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_FI[p]/TESTDIV,mainstddev_FI[p]/TESTDIV,mainavgdist_FI[p]/TESTDIV,mainstddevdist_FI[p]/TESTDIV);
    //   printf("(DNF fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_FI_dnf[p]/TESTDIV,mainstddev_FI_dnf[p]/TESTDIV,mainavgdist_FI_dnf[p]/TESTDIV,mainstddevdist_FI_dnf[p]/TESTDIV);
    //   printf("(noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_NI[p]/TESTDIV,mainstddev_NI[p]/TESTDIV,mainavgdist_NI[p]/TESTDIV,mainstddevdist_NI[p]/TESTDIV);
    //   printf("(driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg_NF[p]/TESTDIV,mainstddev_NF[p]/TESTDIV,mainavgdist_NF[p]/TESTDIV,mainstddevdist_NF[p]/TESTDIV);
    //   printf("(DNF noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg_NI_dnf[p]/TESTDIV,mainstddev_NI_dnf[p]/TESTDIV,mainavgdist_NI_dnf[p]/TESTDIV,mainstddevdist_NI_dnf[p]/TESTDIV);
    //   printf("(DNF driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of learn error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg_NF_dnf[p]/TESTDIV,mainstddev_NF_dnf[p]/TESTDIV,mainavgdist_NF_dnf[p]/TESTDIV,mainstddevdist_NF_dnf[p]/TESTDIV);
    //   printf("GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2[p]/TESTDIV,mainstddev2[p]/TESTDIV,mainavgdist2[p]/TESTDIV,mainstddevdist2[p]/TESTDIV);
    //   printf("(DNF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_dnf[p]/TESTDIV,mainstddev2_dnf[p]/TESTDIV,mainavgdist2_dnf[p]/TESTDIV,mainstddevdist2_dnf[p]/TESTDIV);
    //   printf("(threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_th[p]/TESTDIV,mainstddev2_th[p]/TESTDIV,mainavgdist2_th[p]/TESTDIV,mainstddevdist2_th[p]/TESTDIV);
    //   printf("(DNF threshold) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_th_dnf[p]/TESTDIV,mainstddev2_th_dnf[p]/TESTDIV,mainavgdist2_th_dnf[p]/TESTDIV,mainstddevdist2_th_dnf[p]/TESTDIV);
    //   printf("(fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_FI[p]/TESTDIV,mainstddev2_FI[p]/TESTDIV,mainavgdist2_FI[p]/TESTDIV,mainstddevdist2_FI[p]/TESTDIV);
    //   printf("(DNF fault injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_FI_dnf[p]/TESTDIV,mainstddev2_FI_dnf[p]/TESTDIV,mainavgdist2_FI_dnf[p]/TESTDIV,mainstddevdist2_FI_dnf[p]/TESTDIV);
    //   printf("(noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_NI[p]/TESTDIV,mainstddev2_NI[p]/TESTDIV,mainavgdist2_NI[p]/TESTDIV,mainstddevdist2_NI[p]/TESTDIV);
    //   printf("(driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n",p,SIZE,SIZE,INS,mainavg2_NF[p]/TESTDIV,mainstddev2_NF[p]/TESTDIV,mainavgdist2_NF[p]/TESTDIV,mainstddevdist2_NF[p]/TESTDIV);
    //   printf("(DNF noise injection) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg2_NI_dnf[p]/TESTDIV,mainstddev2_NI_dnf[p]/TESTDIV,mainavgdist2_NI_dnf[p]/TESTDIV,mainstddevdist2_NI_dnf[p]/TESTDIV);
    //   printf("(DNF driven by NF) GLOBAL (cross validated): for %d percents of flipped bits in a %dx%d map with %d inputs, average of test error rate = %f, standard deviation = %f, average of normalized Manhattan distance %f, standard deviation %f\n\n",p,SIZE,SIZE,INS,mainavg2_NF_dnf[p]/TESTDIV,mainstddev2_NF_dnf[p]/TESTDIV,mainavgdist2_NF_dnf[p]/TESTDIV,mainstddevdist2_NF_dnf[p]/TESTDIV);
    // } // end loop on GLOBAL printing for all fault percentages
  } // end loop on cross validation partitions
}