/* TODO
   batch version : proto_i=(sum_x neighb_func(proto_i,proto_win(x)).x)/(sum_x neighb_func(proto_i,proto_win(x)))
   FI : remettre le bit normal après chaque apprentissage batch
*/ 


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define VERBOSE 0
#define PI 3.14159
#define SIZE 8
#define INS 4
#define NBCLASSES 3
#define TESTDIV 5
#define TAU 0.3
#define TAUMIN 0.07
#define MINDIST 0.00001
#define NBEPOCHLEARN 100
#define NBITEREPOCH 100
#define FI_LEVEL 1 /* level of fault injection (number of bits) during learning */
#define NI_LEVEL 0.01 /* level of noise injection during learning */
#define NBMAPS 10
#define MAXFAULTPERCENT 2
#define NBVALIDPARTITIONS 1
#define FILENAME "weights8x8-4.txt"
#define INPUTFILENAME "inputs-4.txt"
#define precision 16 /* precision of weights/inputs/values, 1 bit for sign coding */
#define fractional 10 /* size of the fractional part */
#define one 1024 /* fixed point value for 1.0 */
#define precision_int 16384 /* precision of weights/inputs/fractional part of intermediate computations */
#define nb_experiments 4 /* number of faulty versions of the same map */
#define REST -0.15
#define k_w 1.1
#define W_E 1
#define W_I 1.5
#define k_s 0.5
#define SIGMA_E 0.25
#define SIGMA_I 0.4
#define TAU_DNF 0.2
#define ALPHA 0.5
#define NBITERDNF 100
/* computation time for 1000 exp. and 1000 test. = 1.8s for 10 different percentages
   linear time / INS, expe., test.
   less than quadratic time / SIZE */

typedef struct kohonen {
  int size;
  int nb_inputs;
  int ***weights;
  int **vals;
  int **dnf;
  int *FI; //FI_i,FI_j,FI_k,FI_b
} Kohonen;

typedef struct winner {
  int i;
  int j;
  int value;
} Winner;

typedef struct {
  float i;
  float j;
} WinnerDNF;

#define EPS_SQRT -0.000001

double mysqrt(double x) {
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

void faulty_weights(Kohonen map,int p) {
  /* choose randomly p percent of the bits among all weights and flip them */
  /* total number of bits : precision*SIZE*SIZE*INS */
  int i,j,k,b;
  int taille=precision*map.size*map.size*map.nb_inputs;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      for (k=0;k<map.nb_inputs;k++) {
	for (b=0;b<precision;b++) {
	  if (rand()%taille<(p/100.0)*taille) {
	    int mask=(1<<b);
	    map.weights[i][j][k]=map.weights[i][j][k]^mask;
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

int distance(int *A,int *B,int n) {
  /* computes the Manhattan distance between two integer vectors of size n */
  int norm=0;
  int i;
  for (i=0;i<n;i++) {
    norm+=abs(A[i]-B[i]);
  }
  return norm;
}
  

Winner recall(Kohonen map,int *input) {
  /* computes the winner, i.e. the neuron that is at minimum distance from the given input (integer or fixed point) */
  int min=distance(input,map.weights[0][0],map.nb_inputs);
  int min_i=0,min_j=0;
  int i,j,k;
  for (i=0;i<map.size;i++) {
    for (j=0;j<map.size;j++) {
      int dist=distance(input,map.weights[i][j],map.nb_inputs);
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
  return k_w*W_I*exp(-dist/(2*k_s*k_s*SIGMA_I*SIGMA_I))-W_I*exp(-dist/(2*SIGMA_I*SIGMA_I));
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

void updateDNF(Kohonen map,double sig_e,double sig_i) {
  int i,j,k,n,i0,j0;
  printf("DNF init\n");
  printDNF(map);
  for (k=0;k<NBITERDNF;k++) {
    for (n=0;n<map.size*map.size;n++) {
      i0=rand()%map.size;
      j0=rand()%map.size;
      map.dnf[i0][j0]+=(int)(TAU_DNF*(REST-map.dnf[i0][j0]+ALPHA*map.vals[i0][j0]));
      for (i=0;i<map.size;i++) {
	for (j=0;j<map.size;j++) {
	  map.dnf[i0][j0]+=(int)(TAU_DNF*weightDNFbase(i,j,i0,j0)*map.dnf[i][j]);
	}
      }
      if (map.dnf[i0][j0]<0) map.dnf[i0][j0]=0;
      if (map.dnf[i0][j0]>one) map.dnf[i0][j0]=one;
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
      int dist=distance(input,map.weights[i][j],map.nb_inputs);
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

void gaussianlearnstep(Kohonen map,int *input,double sig,double eps) {
  /* learning step with a gaussian decrease of learning from the winner neuron */
  Winner win=recall(map,input);
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

void NFlearnstep(Kohonen map,int *input,double sig,double eps) {
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
  Winner win=recall(map,input);
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

void errorrate(Kohonen map,int** inputs,int inp,int** classe,int it,int **crossvalid,int testbloc) {
  int i,j,cnt;
  // the class of a neuron is the class of the closest input sample
    
  printneuronclasses(map,inputs,classe,crossvalid,testbloc,inp);
  // computes the number of test patterns that select a neuron of a different class
  cnt=0;
  for (i=0;i<(inp/TESTDIV);i++) {
    Winner win=recall(map,inputs[crossvalid[testbloc][i]]);
    if (classe[win.i][win.j]!=inputs[crossvalid[testbloc][i]][INS]) cnt++;
  }
  double errortest=cnt/(1.0*(inp/TESTDIV));
  printf("test error after %d learning iterations : %f (cnt=%d)\n", it,errortest,cnt);
  // computes the number of learn patterns that select a neuron of a different class
  cnt=0;
  for (j=0;j<TESTDIV-1;j++) {
    if (j==testbloc) j++;
    for (i=0;i<(inp/TESTDIV);i++) {
      Winner win=recall(map,inputs[crossvalid[j][i]]);
      if (classe[win.i][win.j]!=inputs[crossvalid[j][i]][INS]) cnt++;
    }
  }
  double errorlearn=cnt/(1.0*((TESTDIV-1)*inp/TESTDIV));
  printf("learn error after %d learning iterations : %f (cnt=%d)\n", it,errorlearn,cnt);
}

void learn(Kohonen map,int** inputs,int inp,int** classe,int **crossvalid,int testbloc) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning */
  int it,ep;
  for (ep=0;ep<NBEPOCHLEARN;ep++) {
    for (it=0;it<NBITEREPOCH;it++) {
      int numbloc=rand()%TESTDIV;
      while (numbloc==testbloc) numbloc=rand()%TESTDIV;
      int num=rand()%(inp/TESTDIV); /* randomly choose a input pattern */
      // radius decrease from 3 until 1
      //int radius=map.size/2-1-3*it/NBITERLEARN;
      // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
      double sig=SIZE*(0.2-0.19*ep/NBEPOCHLEARN);
      // learning rate decreases from TAU to TAUMIN
      gaussianlearnstep(map,inputs[crossvalid[numbloc][num]],sig,TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*ep)/NBEPOCHLEARN));
    } 
   /* computation of the new error */
    errorrate(map,inputs,inp,classe,ep*NBITEREPOCH+it,crossvalid,testbloc);
  }
}

void learn_NF(Kohonen map,int** inputs,int inp,int** classe,int **crossvalid,int testbloc) {
  /* complete learning, with decreasing learning rate and DNF-driven winner selection */
  int it,ep;
  for (ep=0;ep<NBEPOCHLEARN;ep++) {
    double sig=SIZE*SIGMA_E*(0.5-0.2*ep/NBEPOCHLEARN);
    double rate=TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*ep)/NBEPOCHLEARN);
    printf("DNF kernel at epoch %d : \n",ep);
    printDNFkernel(k_s*SIGMA_I*SIZE,SIZE*SIGMA_I);
    printDNFkernelbase();
    for (it=0;it<NBITEREPOCH;it++) {
      int numbloc=rand()%TESTDIV;
      while (numbloc==testbloc) numbloc=rand()%TESTDIV;
      int num=rand()%(inp/TESTDIV); /* randomly choose a input pattern */
      // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
      // learning rate decreases from TAU to TAUMIN
      NFlearnstep(map,inputs[crossvalid[numbloc][num]],sig,rate);
    }
    /* computation of the new error */
    errorrate(map,inputs,inp,classe,ep*NBITEREPOCH+it,crossvalid,testbloc);
  }
}

void learn_FI(Kohonen map,int** inputs,int inp,int** classe,int **crossvalid,int testbloc) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning
     FAULT INJECTION VERSION : faults are injected during learning */
  int it,ep;
  for (ep=0;ep<NBEPOCHLEARN;ep++) {
    for (it=0;it<NBITEREPOCH;it++) {
      int numbloc=rand()%TESTDIV;
      while (numbloc==testbloc) numbloc=rand()%TESTDIV;
      int num=rand()%(inp/TESTDIV); /* randomly choose a input pattern */
      // radius decrease from 3 until 1
      //int radius=map.size/2-1-3*it/NBITERLEARN;
      // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
      double sig=SIZE*(0.2-0.19*ep/NBEPOCHLEARN);
      // learning rate decreases from TAU to TAUMIN
      faulty_bit(map);
      gaussianlearnstep(map,inputs[crossvalid[numbloc][num]],sig,TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*ep)/NBEPOCHLEARN));
      reverse_faulty_bit(map);
    }
    /* computation of the new error */
    errorrate(map,inputs,inp,classe,ep*NBITEREPOCH+it,crossvalid,testbloc);
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

void learn_NI(Kohonen map,int** inputs,int inp,int** classe,int **crossvalid,int testbloc) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning
     FAULT INJECTION VERSION : faults are injected during learning */
  int it,ep,i;
  int *noisy;
  noisy=(int*)malloc(INS*sizeof(int));
  for (ep=0;ep<NBEPOCHLEARN;ep++) {
    for (it=0;it<NBITEREPOCH;it++) {
      int numbloc=rand()%TESTDIV;
      while (numbloc==testbloc) numbloc=rand()%TESTDIV;
      int num=rand()%(inp/TESTDIV); /* randomly choose a input pattern */
      // radius decrease from 3 until 1
      //int radius=map.size/2-1-3*it/NBITERLEARN;
      // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
      double sig=SIZE*(0.2-0.19*ep/NBEPOCHLEARN);
      // learning rate decreases from TAU to TAUMIN
      for (i=0;i<INS;i++) noisy[i]=inputs[crossvalid[numbloc][num]][i]+noise();
      gaussianlearnstep(map,noisy,sig,TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*ep)/NBEPOCHLEARN));
    }
    /* computation of the new error */
    errorrate(map,inputs,inp,classe,ep*NBITEREPOCH+it,crossvalid,testbloc);
  }
  free(noisy);
}

void learn_threshold(Kohonen map,int** inputs,int inp,int** classe,int **crossvalid,int testbloc) {
  /* complete learning, with decreasing radius of influence for the winner neurons, NBITERLEARN iterations of learning 
      THRESHOLDING VERSION : only weights which abs is below the average abs are updated */
  int it,ep;
  for (ep=0;ep<NBEPOCHLEARN;ep++) {
    for (it=0;it<NBITEREPOCH;it++) {
      int numbloc=rand()%TESTDIV;
      while (numbloc==testbloc) numbloc=rand()%TESTDIV;
      int num=rand()%(inp/TESTDIV); /* randomly choose a input pattern */
      // radius decrease from 3 until 1
      //int radius=map.size/2-1-3*it/NBITERLEARN;
      // gaussian width decreases from 0.2*SIZE until 0.01*SIZE
      double sig=SIZE*(0.2-0.19*ep/NBEPOCHLEARN);
      // learning rate decreases from TAU to TAUMIN
      gaussianlearnstep_threshold(map,inputs[crossvalid[numbloc][num]],sig,TAUMIN+(TAU-TAUMIN)*((NBEPOCHLEARN+1-1.0*ep)/NBEPOCHLEARN));
    }
    /* computation of the new error */
    errorrate(map,inputs,inp,classe,ep*NBITEREPOCH+it,crossvalid,testbloc);
  }
}

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
