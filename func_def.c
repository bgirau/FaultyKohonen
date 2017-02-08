
#include "func_def.h"


float mysqrt(double x) {
  if (x<0) {
    if (x<EPS_SQRT)
      printf("WARNING : sqrt(%f<0)\n",x);
    return 0;
  }
  return sqrt(x);
}

Kohonen copy(Kohonen map) {
  /* copies a given Kohonen map, weight per weight */
  Kohonen map2;
  map2.size=map.size;
  map2.nb_inputs=map.nb_inputs;
  int i,j,k;
  map2.weights=(int***)malloc(map2.size*sizeof(int**));
  map2.dnf=(int**)malloc(map2.size*sizeof(int*));
  map2.vals=(int**)malloc(map2.size*sizeof(int*));
  for (i=0;i<map2.size;i++) {
    map2.weights[i]=(int**)malloc(map2.size*sizeof(int*));
    map2.dnf[i]=(int*)malloc(map2.size*sizeof(int));
    map2.vals[i]=(int*)malloc(map2.size*sizeof(int));
    for (j=0;j<map2.size;j++) {
      map2.weights[i][j]=(int*)malloc(map2.nb_inputs*sizeof(int));
      for (k=0;k<map2.nb_inputs;k++) {
	map2.weights[i][j][k]=map.weights[i][j][k];
      }
    }
  }
  map2.FI=(int*)malloc(4*sizeof(int));
  return map2;
}

void freeMap(Kohonen map) {
  int i,j;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      free(map.weights[i][j]);
    }
    free(map.weights[i]);
    free(map.dnf[i]);
    free(map.vals[i]);
  }
  free(map.weights);
  free(map.dnf);
  free(map.vals);
}

void faulty_weights(Kohonen map, int p) {
  /* choose randomly p percent of the bits among all weights and flip them */
  /* total number of bits : precision*SIZE*SIZE*INS */
  int i, j, k, b;
  int mask;
  int taille = precision * map.size * map.size * map.nb_inputs;
  for (i = 0; i < map.size; i++) {
    for (j = 0; j < map.size; j++) {
      for (k = 0; k < map.nb_inputs; k++) {
        for (b = 0; b < precision; b++) {
          if (rand() % taille < (p / 100.0) * taille) {
            mask = (1 << b);
            map.weights[i][j][k] = map.weights[i][j][k] ^ mask;
          }
        }
      }
    }
  }
}

void faulty_bits(Kohonen map,int N) {
  /* choose randomly N bits among all weights and flip them */
  int i,j,k,b,n,mask;
  for (n=0;n<N;n++) {
    i=rand()%map.size;
    j=rand()%map.size;
    k=rand()%map.nb_inputs;
    b=rand()%precision;
    mask=(1<<b);
    map.weights[i][j][k]=map.weights[i][j][k]^mask;
  }
}

void faulty_bit(Kohonen map) {
  /* choose randomly one bit among all weights and flip it */
  int i,j,k,b,mask;
  map.FI[0]=rand()%map.size;
  map.FI[1]=rand()%map.size;
  map.FI[2]=rand()%map.nb_inputs;
  map.FI[3]=rand()%precision;
  mask=(1<<map.FI[3]);
  map.weights[map.FI[0]][map.FI[1]][map.FI[2]]=map.weights[map.FI[0]][map.FI[1]][map.FI[2]]^mask;
}

void reverse_faulty_bit(Kohonen map) {
  int mask;
  mask=(1<<map.FI[3]);
  map.weights[map.FI[0]][map.FI[1]][map.FI[2]]=map.weights[map.FI[0]][map.FI[1]][map.FI[2]]^mask;
}

Kohonen init() {
  /* creates a new Kohonen map with uniformly random weights between -1 and 1 in fixed point representation */
  Kohonen map;
  int i,j,k;
  map.size=SIZE;
  map.nb_inputs=INS;
  map.weights=(int***)malloc(map.size*sizeof(int**));
  map.dnf=(int**)malloc(map.size*sizeof(int**));
  map.vals=(int**)malloc(map.size*sizeof(int**));
  for (i=0;i<map.size;i++) {
    map.weights[i]=(int**)malloc(map.size*sizeof(int*));
    map.dnf[i]=(int*)malloc(map.size*sizeof(int));
    map.vals[i]=(int*)malloc(map.size*sizeof(int));
    for (j=0;j<map.size;j++) {
      map.weights[i][j]=(int*)malloc(map.nb_inputs*sizeof(int));
      for (k=0;k<map.nb_inputs;k++) {
	map.weights[i][j][k]=(rand()%(2*one+1))-one;
      }
    }
  }
  map.FI=(int*)malloc(4*sizeof(int));
  return map;
}

Kohonen init_pos() {
  /* creates a new Kohonen map with uniformly random weights between 0 and max in fixed point representation */
  Kohonen map;
  int i,j,k;
  map.size=SIZE;
  map.nb_inputs=INS;
  map.weights=(int***)malloc(map.size*sizeof(int**));
  for (i=0;i<map.size;i++) {
    map.weights[i]=(int**)malloc(map.size*sizeof(int*));
    for (j=0;j<map.size;j++) {
      map.weights[i][j]=(int*)malloc(map.nb_inputs*sizeof(int));
      for (k=0;k<map.nb_inputs;k++) {
	map.weights[i][j][k]=(int)(rand()%one);
      }
    }
  }
  map.FI=(int*)malloc(4*sizeof(int));
  return map;
}

double distance(int *A,int *B,int n) {
   // computes the Euclidian distance between two integer vectors of size n 
  double norm=0.0;
  int i;
  for (i=0;i<n;i++) {
    double val=(A[i]-B[i])/(1.0*one);
    norm += val*val;
  }
  norm = sqrt(norm);
  return norm;
}

int distance_L1(int *A,int *B,int n) {
   // computes the Manhattan distance between two integer vectors of size n 
  int norm=0;
  int i;
  for (i=0;i<n;i++) {
    norm += abs(A[i]-B[i]);
  }
  return norm;
}
  

Winner recall(Kohonen map,int *input) {
  /* computes the winner, i.e. the neuron that is at minimum distance from the given input (integer or fixed point) */
  int min = distance_L1(input,map.weights[0][0],map.nb_inputs);
  int min_i=0,min_j=0;
  int i,j,k;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      int dist = distance_L1(input,map.weights[i][j],map.nb_inputs);
      map.dnf[i][j]=0;
      if (dist<min) {
      	min=dist;
      	min_i=i;
      	min_j=j;
      }
    }
  }
  Winner win;
  win.i=min_i;
  win.j=min_j;
  win.value=min;
  return win;
}

void printVALS(Kohonen map) {
  int i,j;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      printf("%f ",map.vals[i][j]/(one*1.0));
    }
    printf("\n");
  }
  printf("\n");
}

void printDNF(Kohonen map) {
  int i,j;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      printf("%f ",map.dnf[i][j]/(one*1.0));
    }
    printf("\n");
  }
  printf("\n");
}

/*
void updateDNF(Kohonen map) {
  int i,j,k,n,i0,j0;
  for (k=0;k<NBITERDNF;k++) {
    for (n=0;n<map.size*map.size;n++) {
      i0=rand()%map.size;
      j0=rand()%map.size;
      map.dnf[i0][j0]+=(int)(TAU_DNF*(-map.dnf[i0][j0]+ALPHA*map.vals[i0][j0]));
      for (i=-SIGMA_E;i<=SIGMA_E;i++) {
	for (j=-SIGMA_E;j<=SIGMA_E;j++) {
	  int i1,j1;
	  i1=(i0+i+map.size)%(map.size);
	  j1=(j0+j+map.size)%(map.size);
	  map.dnf[i0][j0]+=(int)(TAU_DNF*W_E*map.dnf[i1][j1]);
	}
      }
      for (i=-SIGMA_I;i<=SIGMA_I;i++) {
	for (j=-SIGMA_I;j<=SIGMA_I;j++) {
	  int i1,j1;
	  i1=(i0+i+map.size)%(map.size);
	  j1=(j0+j+map.size)%(map.size);
	  map.dnf[i0][j0]+=(int)(TAU_DNF*W_I*map.dnf[i1][j1]);
	}
      }
      if (map.dnf[i0][j0]<0) map.dnf[i0][j0]=0;
      if (map.dnf[i0][j0]>one) map.dnf[i0][j0]=one;
    }
    //printDNF(map);
  }
}
*/

double weightDNFbase(int i,int j,int i0,int j0) {
  double dist=((i-i0)*(i-i0)+(j-j0)*(j-j0))/(1.0*SIZE*SIZE);
    double gss1 = k_w*W_I*exp(-dist/(2*k_s*k_s*SIGMA_I*SIGMA_I));
    double gss2 = W_I*exp(-dist/(2*SIGMA_I*SIGMA_I));
  return gss1-gss2;
}


double weightDNF(int i,int j,int i0,int j0,double s_e,double s_i) {
  double dist=(i-i0)*(i-i0)+(j-j0)*(j-j0);
  return W_E*exp(-dist/(s_e*s_e))+W_I*exp(-dist/(s_i*s_i));
}

void printDNFkernel(double s_e,double s_i) {
  int i0=SIZE/2;
  int j0=SIZE/2;
  int i,j;
  for (i=0;i<SIZE;i++) {
    for (j=0;j<SIZE;j++) {
      printf("%f ",weightDNF(i,j,i0,j0,s_e,s_i));
    }
    printf("\n");
  }
  printf("\n");
}

void printDNFkernelbase() {
  int i0=SIZE/2;
  int j0=SIZE/2;
  int i,j;
  for (i=0;i<SIZE;i++) {
    for (j=0;j<SIZE;j++) {
      printf("%f ",weightDNFbase(i,j,i0,j0));
    }
    printf("\n");
  }
  printf("\n");
}
double * weightDNFbase_kernel(int i0, int size, double A0, double B0) {
  double *  kernel = calloc(size, sizeof(double));
  int i;
  double  dist;
  for (i = 0; i < size; i++){
    dist = ((i-i0)*(i-i0))/(1.0*SIZE*SIZE);
    kernel[i] = A0 * exp(-dist/(2* B0 *SIGMA_I*SIGMA_I));
  }
  return kernel;
}

//void updateDNF(Kohonen map,double sig_e,double sig_i) {
//    int i,j,k,n,i0,j0;
//  printf("DNF init\n");
//   printDNF(map);
//  for (k=0;k<NBITERDNF;k++) {
//    for (n=0;n<map.size*map.size;n++) {
//      i0=rand()%map.size;
//      j0=rand()%map.size;
//      map.dnf[i0][j0]+=(int)(TAU_DNF*(REST-map.dnf[i0][j0]+ALPHA*map.vals[i0][j0]));
//      for (i=0;i<map.size;i++) {
//  for (j=0;j<map.size;j++) {
//    map.dnf[i0][j0]+=(int)(TAU_DNF*weightDNFbase(i,j,i0,j0)*map.dnf[i][j]);
//  }
//      }
//      if (map.dnf[i0][j0]<0) map.dnf[i0][j0]=0;
//      if (map.dnf[i0][j0]>one) map.dnf[i0][j0]=one;
//    }
//
//    printf("DNF after iteration %d\n",k);
//    printVALS(map);
//    printDNF(map);
//
//  }
//  printf("DNF after convergence\n");
//  printDNF(map);
//}

void updateDNF(Kohonen map,double sig_e,double sig_i) {
  int i,j,k,n,i0,j0;

  printf("DNF init\n");
   printDNF(map);
  double * kernel_x;
  double * kernel_y;
  double * kernel_x2;
  double * kernel_y2;

  for (k=0;k<NBITERDNF;k++) {
    for (n=0;n<map.size*map.size;n++) {
       i0 = rand() % map.size;
       j0 = rand() % map.size;

      map.dnf[i0][j0]+=(int)(TAU_DNF * (REST - map.dnf[i0][j0] + ALPHA * map.vals[i0][j0]));
      kernel_x = (double *) weightDNFbase_kernel(j0, map.size, k_w*W_I, k_s*k_s);
      kernel_y = (double *) weightDNFbase_kernel(i0, map.size, 1.0, k_s*k_s);
      kernel_x2 = (double *) weightDNFbase_kernel(j0, map.size, W_I, 1.0);
      kernel_y2 = (double *) weightDNFbase_kernel(i0, map.size, 1.0, 1.0);
      for (i = 0; i < map.size; i++) {
          for (j = 0; j < map.size; j++) {
              map.dnf[i0][j0] += (int) (TAU_DNF * kernel_x[j] * kernel_y[i] * map.dnf[i][j] -
                                        TAU_DNF * kernel_x2[j] * kernel_y2[i] * map.dnf[i][j]);
          }
      }
      if (map.dnf[i0][j0] < 0) map.dnf[i0][j0] = 0;
      if (map.dnf[i0][j0] > one) map.dnf[i0][j0] = one;
    }
    /*
    printf("DNF after iteration %d\n",k);
    printVALS(map);
    printDNF(map);
    */
  }
  printf("DNF after convergence\n");
  printDNF(map);
}

void initVALS(Kohonen map,int *input) {
  int i,j,maxdist=0;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      int dist=distance_L1(input,map.weights[i][j],map.nb_inputs);
      map.vals[i][j]=dist;
      if (dist>maxdist) maxdist=dist;
      map.dnf[i][j]=0;
    }
  }
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      map.vals[i][j]=maxdist-map.vals[i][j];
    }
  }
}
  

WinnerDNF recallDNF(Kohonen map,int *input) {
  /* computes the winner filtered by the DNF */
  int i,j,k;
  int min;
  initVALS(map,input);
  updateDNF(map,1.6,SIGMA_I);
  min=map.dnf[0][0]; // indeed, search for maximum dnf value
  float min_i=0,min_j=0,sum=0;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      if (min<map.dnf[i][j]) min=map.dnf[i][j];
      float val=(map.dnf[i][j])/(one*1.0);
      sum+=val;
      min_i+=val*i;
      min_j+=val*j;
    }
  }
  min_i/=sum;
  min_j/=sum;
  WinnerDNF win;
  win.i=min_i;
  win.j=min_j;
  //  printf("winnerDNF : i= %f j=%f\n",min_i,min_j);
  return win;
}

int *prototypeDNF(Kohonen map) {
  int *res;
  res=(int*)malloc(map.nb_inputs*sizeof(int));
  int i,j,k,l;
  int total=0;
  for (k=0;k<map.nb_inputs;k++) res[k]=0;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      total+=map.dnf[i][j];
      for (k=0;k<map.nb_inputs;k++) {
	res[k]+=map.dnf[i][j]*map.weights[i][j][k];
      }
    }
  }
  for (k=0;k<map.nb_inputs;k++) {
    res[k]/=total;
  }
  return res;
}

void gaussianlearnstep(Kohonen map, int *input, double sig, double eps) {
  /* learning step with a gaussian decrease of learning from the winner neuron */
  Winner win = recall(map,input);
  int i,j,k;
  double dx;
  double dy;
  double coeff;
  // gaussian width should be proportional to map size
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      dx=1.0*(i-win.i);
      dy=1.0*(j-win.j);
      coeff=exp(-1*(dx*dx+dy*dy)/(2*sig*sig));
      for (k=0;k<map.nb_inputs;k++) {
        map.weights[i][j][k]+=(int)(eps*coeff*(input[k]-map.weights[i][j][k]));
        if (map.weights[i][j][k]>precision_int) map.weights[i][j][k]=precision_int;
        if (map.weights[i][j][k]<-precision_int) map.weights[i][j][k]=-precision_int;
      }
    }
  }
}

void NFlearnstep(Kohonen map, int *input, double sig, double eps) {
  /* learning step driven by a DNF */
  int i,j,k;
  double dx;
  double dy;
  double coeff;

  initVALS(map,input);
  updateDNF(map,sig,SIGMA_I);

  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      for (k=0;k<map.nb_inputs;k++) {
        map.weights[i][j][k]+=(int)(eps*map.dnf[i][j]*(input[k]-map.weights[i][j][k]));
        if (map.weights[i][j][k]>precision_int) map.weights[i][j][k]=precision_int;
        if (map.weights[i][j][k]<-precision_int) map.weights[i][j][k]=-precision_int;
      }
    }
  }
}

void gaussianlearnstep_threshold(Kohonen map,int *input,double sig,double eps) {
  /* learning step with a gaussian decrease of learning from the winner neuron */
  Winner win = recall(map,input);
  int i,j,k;
  double dx;
  double dy;
  double coeff;
  int avg;
  // compute weight average
  avg=0;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      for (k=0;k<map.nb_inputs;k++) {
        avg+=abs(map.weights[i][j][k]);
      }
    }
  }
  avg/=(map.size*map.size*map.nb_inputs);
  // gaussian width should be proportional to map size
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      dx=1.0*(i-win.i);
      dy=1.0*(j-win.j);
      coeff=exp(-1*(dx*dx+dy*dy)/(2*sig*sig));
      for (k=0;k<map.nb_inputs;k++) {
      	if (abs(map.weights[i][j][k])<=avg) {
      	  map.weights[i][j][k]+=(int)(eps*coeff*(input[k]-map.weights[i][j][k]));
      	  if (map.weights[i][j][k]>precision_int) map.weights[i][j][k]=precision_int;
      	  if (map.weights[i][j][k]<-precision_int) map.weights[i][j][k]=-precision_int;
      	}
      }
    }
  }
}

void heavisidelearnstep(Kohonen map,int *input,int radius,double eps) {
  /* learning step with a constant rate for all neurons within a given radius distance from the winner neuron */
  Winner win=recall(map,input);
  int i,j,k;
  // radius should be proportional to map size
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      int d=abs(i-win.i);
      if (d<abs(j-win.j)) d=abs(j-win.j);
      if (d<=radius)
	    for (k=0;k<map.nb_inputs;k++) {
	      map.weights[i][j][k]+=(int)(eps*(input[k]-map.weights[i][j][k]));
	      if (map.weights[i][j][k]>precision_int) map.weights[i][j][k]=precision_int;
	      if (map.weights[i][j][k]<-precision_int) map.weights[i][j][k]=-precision_int;
	    }
    }
  }
}

/*
void NN1neuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {
  int i,j,t,n,tmin,nmin;
  float d,dmin=-1;
  for (i=0;i<SIZE;i++) {
    for (j=0;j<SIZE;j++) {
      for (t=0;t<TESTDIV-1;t++) {
	if (t==testbloc) t++;
	for (n=0;n<(inp/TESTDIV);n++) {
	  float d=distance(in[crossvalid[t][n]],map.weights[i][j],map.nb_inputs);
	  if ((dmin==-1)||(d<dmin)) {
	    dmin=d;
	    tmin=t;
	    nmin=n;
	  }
	}
      }
      classe[i][j]=in[crossvalid[tmin][nmin]][INS];
    }
  }
}
*/

void NN5neuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {
  int x,y;
  int *neighbs;
  float *neighbdists;
  int i,j,m;
  int k=5;
  neighbs=(int*)malloc(k*sizeof(int));
  neighbdists=(float*)malloc(k*sizeof(float));
  for (x=0;x<SIZE;x++) {
    for (y=0;y<SIZE;y++) {
      for (i=0;i<k;i++) neighbs[i]=-1;
      for (j=0;j<4;j++) {
	if (j==testbloc) j++;
	for (i=0;i<(inp/TESTDIV);i++) {
	  // OPEN QUESTION : distance or distance_L1 ???
	  float d=distance(map.weights[x][y],in[crossvalid[j][i]],INS);
	  // cas particulier : distance nulle
	  if (d==0) d=MINDIST;
	  int from=0;
	  while ((from<k)&&(neighbs[from]!=-1)) from++;
	  if (from==0) {
	    neighbs[0]=crossvalid[j][i];
	    neighbdists[0]=d;
	  } else {
	    int to=from-1;
	    while ((to>=0)&&(d<neighbdists[to])) to--;
	    for (m=from-1;m>to;m--) {
	      neighbs[m+1]=neighbs[m];
	      neighbdists[m+1]=neighbdists[m];
	    }
	    neighbs[to+1]=crossvalid[j][i];
	    neighbdists[to+1]=d;
	  }
	}
      }
      /* déterminatin de la classe pondérée la plus représentée */
      float *rep;
      rep=(float*)malloc((NBCLASSES+1)*sizeof(float));
      for (i=1;i<=NBCLASSES;i++) rep[i]=0;
      for (i=0;i<k;i++) {
	rep[in[neighbs[i]][INS]]+=1/neighbdists[i];
      }
      int max=1;
      for (i=2;i<=NBCLASSES;i++)
	if (rep[i]>rep[max]) max=i;
      classe[x][y]=max;
      free(rep);
    }
  }
  free(neighbs);
  free(neighbdists);
}

int NN5DNFclass(Kohonen map,int **in,int **crossvalid,int testbloc,int inp) {
  /* 5-NN determination of the class attributed to the prototype obtained by weighting all neuron prototypes by their DNF activity */
  int *neighbs;
  float *neighbdists;
  int i,j,m;
  int k=5;
  int *prototype=prototypeDNF(map);
  neighbs=(int*)malloc(k*sizeof(int));
  neighbdists=(float*)malloc(k*sizeof(float));
  for (i=0;i<k;i++) neighbs[i]=-1;
  for (j=0;j<4;j++) {
    if (j==testbloc) j++;
    for (i=0;i<(inp/TESTDIV);i++) {
      // OPEN QUESTION : distance or distance_L1 ???
      float d=distance(prototype,in[crossvalid[j][i]],INS);
      // cas particulier : distance nulle
      if (d==0) d=MINDIST;
      int from=0;
      while ((from<k)&&(neighbs[from]!=-1)) from++;
      if (from==0) {
	neighbs[0]=crossvalid[j][i];
	neighbdists[0]=d;
      } else {
	int to=from-1;
	while ((to>=0)&&(d<neighbdists[to])) to--;
	for (m=from-1;m>to;m--) {
	  neighbs[m+1]=neighbs[m];
	  neighbdists[m+1]=neighbdists[m];
	}
	neighbs[to+1]=crossvalid[j][i];
	neighbdists[to+1]=d;
      }
    }
  }
  free(prototype);
  /* déterminatin de la classe pondérée la plus représentée */
  float *rep;
  rep=(float*)malloc((NBCLASSES+1)*sizeof(float));
  for (i=1;i<=NBCLASSES;i++) rep[i]=0;
  for (i=0;i<k;i++) {
    rep[in[neighbs[i]][INS]]+=1/neighbdists[i];
  }
  int max=1;
  for (i=2;i<=NBCLASSES;i++)
    if (rep[i]>rep[max]) max=i;
  free(rep);
  free(neighbs);
  free(neighbdists);
  return max;
}

int DNFclass(Kohonen map,int **in,int **crossvalid,int testbloc,int inp) {
  return NN5DNFclass(map,in,crossvalid,testbloc,inp);
}

void MLneuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {
  int i,j,k,t,n,tmin,nmin;
  float d,dmin=-1;
  int*** cntclasses;
  cntclasses=(int***)malloc(SIZE*sizeof(int**));
  for (i=0;i<SIZE;i++) {
    cntclasses[i]=(int**)malloc(SIZE*sizeof(int*));
    for (j=0;j<SIZE;j++) {
      cntclasses[i][j]=(int*)malloc((NBCLASSES+1)*sizeof(int));
      for (k=1;k<=NBCLASSES;k++)
	cntclasses[i][j][k]=0;
    }
  }
  for (t=0;t<TESTDIV-1;t++) {
    if (t==testbloc) t++;
    for (n=0;n<(inp/TESTDIV);n++) {
      Winner win=recall(map,in[crossvalid[t][n]]);
      cntclasses[win.i][win.j][in[crossvalid[t][n]][INS]]++;
    }
  }

  // then for each neuron computes the most represented class (0 if none or if equality between at least two classes)
  for (i=0;i<SIZE;i++) {
    for (j=0;j<SIZE;j++) {
      int max=0;
      int imax=0;
      for (k=1;k<=NBCLASSES;k++) {
	if (cntclasses[i][j][k]==max) {
	  max=0;
	  imax=0;
	}
	if (cntclasses[i][j][k]>max) {
	  max=cntclasses[i][j][k];
	  imax=k;
	}
      }
      classe[i][j]=imax;
    }
  }
  for (i=0;i<SIZE;i++) {
    for (j=0;j<SIZE;j++) {
      free(cntclasses[i][j]);
    }
    free(cntclasses[i]);
  }
  free(cntclasses);
}

void neuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {
  NN5neuronclasses(map,in,classe,crossvalid,testbloc,inp);
}

void printneuronclasses(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {
  int i,j;
  neuronclasses(map,in,classe,crossvalid,testbloc,inp);
    //affichage classes neurones
    for (i=0;i<SIZE;i++) {
      for (j=0;j<SIZE;j++) {
	printf("%d ",classe[i][j]);
      }
      printf("\n");
    }
    printf("\n");
}

void errorrateDNF(Kohonen map,int** inputs,int inp,int** classe,int it,int **crossvalid,int testbloc) {
  int i,j,cnt;
  // computes the number of test patterns that select a neuron of a different class
  cnt=0;
  for (i=0;i<(inp/TESTDIV);i++) {
    WinnerDNF win=recallDNF(map,inputs[crossvalid[testbloc][i]]);
    int cls=DNFclass(map,inputs,crossvalid,testbloc,inp);
    if (cls!=inputs[crossvalid[testbloc][i]][INS]) cnt++;
  }
  double errortest=cnt/(1.0*(inp/TESTDIV));
  printf("(DNF) test error after %d learning iterations : %f (cnt=%d)\n", it,errortest,cnt);
  // computes the number of learn patterns that select a neuron of a different class
  cnt=0;
  for (j=0;j<TESTDIV-1;j++) {
    if (j==testbloc) j++;
    for (i=0;i<(inp/TESTDIV);i++) {
      WinnerDNF win=recallDNF(map,inputs[crossvalid[j][i]]);
      int cls=DNFclass(map,inputs,crossvalid,testbloc,inp);
      if (cls!=inputs[crossvalid[j][i]][INS]) cnt++;
    }
  }
  double errorlearn=cnt/(1.0*((TESTDIV-1)*inp/TESTDIV));
  printf("(DNF) learn error after %d learning iterations : %f (cnt=%d)\n", it,errorlearn,cnt);
}

double errorrate(Kohonen map, int ** inputs,int inp, int epoch) {

  double aqe = avg_quant_error(map, inputs,inp);
    printf("learn aqe after %d learning iterations : %f\n",
           epoch * NBITEREPOCH, aqe);
    return aqe;
}

double evaldistortion(Kohonen map, int ** inputs,int inp,int epoch) {

  double aqe = distortion_measure(map, inputs,inp,SIGMA_GAUSS);
    printf("learn distortion after %d learning iterations : %f\n",
           epoch * NBITEREPOCH, aqe);
    return aqe;
}

double distortion_measure(Kohonen map, int** inputs, int inp,double sig) {
// sig = (0.2 + 0.01)/2*MAPSIZE
// sig = 0.1*SIZE
  Winner win;
  int    i, j, k;
  double dx;
  double dy;
  double coeff;
  double dist;
  double distortion;
  double global_distortion=0.0;
  double normalise;

  for (k = 0;k < inp; k++) {
    normalise=0.0;
    distortion=0.0;
    win = recall(map, inputs[k]);
    for (i = 0; i < map.size; i++) {
      for (j = 0;j < map.size; j++) {
        dx    = 1.0 * (i - win.i) / map.size;
        dy    = 1.0 * (j - win.j) / map.size;
        coeff = exp(-1 * (dx * dx + dy * dy) / (2 * sig * sig));
	normalise+=coeff;
        dist  = distance(inputs[k], map.weights[i][j], map.nb_inputs);
        distortion += coeff * dist * dist;
      }
    }
    global_distortion+=distortion/normalise;
  }
  return global_distortion/inp;
}

double distortion_measure_L1(Kohonen map, int** inputs, int inp, double sig) {
// sig = (0.2 + 0.01)/2*MAPSIZE
// sig = 0.1*SIZE
  Winner win;
  int    i, j, k;
  double dx;
  double dy;
  double coeff;
  double dist;
  double distortion;
  double global_distortion=0.0;
  double normalise;

  for (k = 0;k < inp; k++) {
    normalise=0.0;
    distortion=0.0;
    win = recall(map, inputs[k]);
    for (i = 0; i < map.size; i++) {
      for (j = 0;j < map.size; j++) {
        dx    = 1.0 * (i - win.i) / map.size;
        dy    = 1.0 * (j - win.j) / map.size;
        coeff = exp(-1 * (dx * dx + dy * dy) / (2 * sig * sig));
	normalise+=coeff;
        dist  = distance_L1(inputs[k], map.weights[i][j], map.nb_inputs)/(1.0*one);
        distortion += coeff * dist * dist;
      }
    }
    global_distortion+=distortion/normalise;
  }
  return global_distortion/inp;
}

double avg_quant_error(Kohonen map, int ** inputs,int inp){
  Winner win;
  int i, j;
  double error=0.0;

  for (i = 0; i < inp; i++){
    win = recall(map,inputs[i]);
    error  += distance(inputs[i], map.weights[win.i][win.j], map.nb_inputs); 
  }
  error /= inp;
  return error;
}

double avg_quant_error_L1(Kohonen map, int ** inputs,int inp){
  Winner win;
  int i, j;
  double error=0.0;

  for (i = 0; i < inp; i++){
    win = recall(map,inputs[i]);
    error  += distance_L1(inputs[i], map.weights[win.i][win.j], map.nb_inputs)/(1.0*one); 
  }
  error /= inp;
  return error;
}

void learn(Kohonen map, int ** inputs, int epoch) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning */
  int it;
  // radius decrease from 3 until 1
  // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
  double sig = SIZE * (0.2 - 0.19 * epoch / NBEPOCHLEARN);
  double eps = TAUMIN + (TAU - TAUMIN)*((NBEPOCHLEARN+1-1.0*epoch)/NBEPOCHLEARN);

  for (it = 0; it < NBITEREPOCH; it++) {
    // int radius=map.size/2-1-3*it/NBITERLEARN;
    // learning rate decreases from TAU to TAUMIN
    gaussianlearnstep(map, inputs[it], sig, eps);
  } 
}

void learn_NF(Kohonen map, int ** inputs, int epoch) {
  /* complete learning, with decreasing learning rate and DNF-driven winner selection */
  int it;  
  double sig = SIZE * SIGMA_E * (0.5 - 0.2 * epoch / NBEPOCHLEARN);
  double rate = TAUMIN + (TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*epoch)/NBEPOCHLEARN);

  printf("DNF kernel at epoch %d : \n", epoch+1);
  printDNFkernel(k_s*SIGMA_I*SIZE,SIZE*SIGMA_I);
  printDNFkernelbase();

  for (it=0;it<NBITEREPOCH;it++) {
    // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
    // learning rate decreases from TAU to TAUMIN
    NFlearnstep(map,inputs[it], sig, rate);
  }  
}

void learn_FI(Kohonen map, int ** inputs, int epoch) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning
     FAULT INJECTION VERSION : faults are injected during learning */
  int it;
  // radius decrease from 3 until 1
  //int radius=map.size/2-1-3*it/NBITERLEARN;
  // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
  double sig = SIZE * (0.2 - 0.19 * epoch/NBEPOCHLEARN);
  double eps = TAUMIN + (TAU-TAUMIN) * ((NBEPOCHLEARN+1-1.0*epoch)/NBEPOCHLEARN);

  for (it=0;it<NBITEREPOCH;it++) {    
    // learning rate decreases from TAU to TAUMIN
    faulty_bit(map);
    gaussianlearnstep(map,inputs[it], sig, eps);
    reverse_faulty_bit(map);
  } 
}

int noise() {
  // bruit gaussien de moyenne nulle, de variance NI_LEVEL
  double r, theta,u,v,x,y ;
  u=rand()*1.0/RAND_MAX;
  v=rand()*1.0/RAND_MAX;
  theta=2*PI*u;
  r=mysqrt(-2*log(v));
  x=NI_LEVEL*r*cos(theta);
  //  printf("noise %g\n",x);
  return (int)(one*x);
}

void learn_NI(Kohonen map, int ** inputs, int epoch) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning
     FAULT INJECTION VERSION : faults are injected during learning */
  int it,i;
  // radius decrease from 3 until 1
  //int radius=map.size/2-1-3*it/NBITERLEARN;
  // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
  double sig = SIZE*(0.2-0.19*epoch/NBEPOCHLEARN);
  // learning rate decreases from TAU to TAUMIN
  double eps = TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*epoch)/NBEPOCHLEARN);
  int *noisy = malloc(INS * sizeof(int));
  for (it=0;it<NBITEREPOCH;it++) {
    for (i=0;i<INS;i++) noisy[i]=inputs[it][i] + noise();
    gaussianlearnstep(map, noisy, sig, eps);
  }  
  free(noisy);
}

void learn_threshold(Kohonen map, int ** inputs, int epoch) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning 
      THRESHOLDING VERSION : only weights which abs is below the average abs are updated */
  int it;
  // radius decrease from 3 until 1
  //int radius=map.size/2-1-3*it/NBITERLEARN;
  // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
  double sig = SIZE*(0.2-0.19*epoch/NBEPOCHLEARN);
  // learning rate decreases from TAU to TAUMIN
  double eps = TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*epoch)/NBEPOCHLEARN);

  for (it = 0; it < NBITEREPOCH; it++) {
    gaussianlearnstep_threshold(map,inputs[it], sig, eps);
  }
}
