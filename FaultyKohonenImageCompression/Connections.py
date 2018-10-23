from Connections_Models import *
from Parameters import *


def kohonen():
    connexion_matrix = np.empty((neuron_nbr, neuron_nbr, 5, 5))
    for y in range(neuron_nbr):
        for x in range(neuron_nbr):
            connexion_matrix[x, y] = kohonen_matrix
    return connexion_matrix


