#include "gss_som.h"


float gauss_distance(Kohonen map, int *B, int n, float std_dev, int x_node, int y_node){

	float		filtered_distance;
	float	**	kernel = (float **)gauss_kernel(map.size, std_dev, x_node, y_node);
	int 		i, j;

	filtered_distance = 0.0;

	for(i = 0; i < map.size; i++){
		for (j = 0; j < map.size; j++)
		{
			filtered_distance += distance(map.weights[i][j], B, n) * kernel[i][j];
		}
	}
	return filtered_distance;
}

Winner recallGSS(Kohonen map, int *input, float std_dev) {
  /* computes the winner, i.e. the neuron that is at minimum distance from the given input (integer or fixed point) */
	int min = gauss_distance(map, input, map.nb_inputs, std_dev, 0, 0);
 	int min_i = 0, min_j = 0;
  	int i,j,k;

  	for (i = 0; i < map.size; i++) {
    	for (j = 0; j < map.size; j++) {
      		float dist = gauss_distance(map, input, map.nb_inputs, std_dev, j, i);
      		// map.dnf[i][j] = 0;
      		if (dist < min) {
				min 	= dist;
				min_i 	= i;
				min_j	= j;
      		}
    	}
  	}
  	Winner win;
  	
  	win.i 		= min_i;
  	win.j 		= min_j;
  	win.value 	= min;

  	return win;
}


void NN5neuronclassesGSS(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {

	int			x, y;
  	int 	*	neighbs;
  	float 	*	neighbdists;
  	int 		i, j, m;
	int 		k = 5;
  
  	neighbs 	= 	(int*)malloc(k*sizeof(int));
  	neighbdists = (float*)malloc(k*sizeof(float));

	for (x = 0; x < SIZE; x++) {
	  	for (y = 0; y < SIZE; y++) {
	    	for (i = 0; i < k; i++) neighbs[i]=-1;
	    	for (j = 0; j < 4; j++) {
				if (j == testbloc) j++;
				for (i = 0; i < (inp/TESTDIV); i++) {
					float d = gauss_distance(map, in[crossvalid[j][i]], INS, 1.0, y, x);
		  			// 
		  			if (d==0) d = MINDIST;

		  			int from = 0;

		  			while ((from<k)&&(neighbs[from]!=-1)) from++;

		  			if (from==0) {
		    			neighbs[0]=crossvalid[j][i];
		    			neighbdists[0]=d;
		  			} 
		  			else {
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
	      	/*  */
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

void neuronclassesGSS(Kohonen map, int **in,int **classe,int **crossvalid,int testbloc,int inp) {
	NN5neuronclassesGSS(map,in,classe,crossvalid,testbloc,inp);
}

void printneuronclassesGSS(Kohonen map,int **in,int **classe,int **crossvalid,int testbloc,int inp) {
	int i,j;
	NN5neuronclassesGSS(map, in, classe, crossvalid, testbloc, inp);
    //affichage classes neurones
    for (i=0;i<SIZE;i++) {
    	for (j=0;j<SIZE;j++) {
			printf("%d ",classe[i][j]);
      	}
      	printf("\n");
    }
    printf("\n");
}

void errorrateGSS(Kohonen map,int** inputs,int inp,int** classe,int it,int **crossvalid,int testbloc) {
  int i,j,cnt;
  // the class of a neuron is the class of the closest input sample
    
  printneuronclassesGSS(map,inputs,classe,crossvalid,testbloc,inp);
  // computes the number of test patterns that select a neuron of a different class
  cnt=0;
  for (i=0;i<(inp/TESTDIV);i++) {
    Winner win = recallGSS(map,inputs[crossvalid[testbloc][i]], 1.0);
    if (classe[win.i][win.j]!=inputs[crossvalid[testbloc][i]][INS]) cnt++;
  }
  double errortest=cnt/(1.0*(inp/TESTDIV));
  printf("test error after %d learning iterations : %f (cnt=%d)\n", it,errortest,cnt);
  // computes the number of learn patterns that select a neuron of a different class
  cnt=0;
  for (j=0;j<TESTDIV-1;j++) {
    if (j==testbloc) j++;
    for (i=0;i<(inp/TESTDIV);i++) {
      Winner win = recallGSS(map,inputs[crossvalid[j][i]], 1.0);
      if (classe[win.i][win.j]!=inputs[crossvalid[j][i]][INS]) cnt++;
    }
  }
  double errorlearn=cnt/(1.0*((TESTDIV-1)*inp/TESTDIV));
  printf("learn error after %d learning iterations : %f (cnt=%d)\n", it,errorlearn,cnt);
}