from SOM import *

OPTIMIZED_WEIGHTS=0
INDIVIDUAL_WEIGHTS=1

precision=16
fractional=10
one=1024

def prec_weights(w):
  w=abs(w)
  un=one
  i=0
  while(un<w):
    un*=2
    i=i+1
  return i+fractional
  
def prec_indiv_weights(w):
  w=abs(w)
  un=1
  i=0
  while(un<w):
    un*=2
    i=i+1
  return i
  
def faulty_weights(FP_som, p):
  # choose randomly p percent of the bits among all weights and flip them */
  # total number of bits : precision*SIZE*SIZE*INS */
  # warning : error positions generated randomly, two errors at the same bit result in no error */
  for i in range(neuron_nbr):
    for j in range(neuron_nbr):
      for k in range(pictures_dim[0]*pictures_dim[1]):
        poids=int(FP_som.nodes[i,j].weight[k]*one)
        if (OPTIMIZED_WEIGHTS==1):
          prec=prec_weights(poids)
        else:
          if (INDIVIDUAL_WEIGHTS==1):
            prec=prec_indiv_weights(poids)
          else:
            prec=precision
        taille = precision * neuron_nbr * neuron_nbr * pictures_dim[0] * pictures_dim[1]
        for b in range(prec):
          if (np.random.random_sample() * taille < (p / 100.0) * taille):
            mask = (1 << b)
            poids = poids ^ mask
        FP_som.nodes[i,j].weight[k]=poids/float(one)

def faulty(carte,p):
    Fsom=carte.copy()
    faulty_weights(Fsom,p)
    return Fsom
    
