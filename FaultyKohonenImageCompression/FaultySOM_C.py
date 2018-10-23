from Images import *
from SOM import *
from Connections import *
from Faulty import *
import os

def noLink():
    pix = np.full(pictures_dim,255)
    return pix


def hLink():
    pix = np.full(pictures_dim,255)
    mid = pictures_dim[0]//2
    for j in range(pictures_dim[1]):
        pix[mid][j]=0
    return pix


def vLink():
    pix = np.full(pictures_dim,255)
    mid = pictures_dim[1]//2
    for i in range(pictures_dim[0]):
        pix[i][mid]=0
    return pix


def display_som(som_list):
    som_list = som_list*255
    px2 = []
    lst2 = ()
    for y in range(neuron_nbr):
        lst = ()
        for x in range(neuron_nbr):
            som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(pictures_dim)
            lst = lst + (som_list[y * neuron_nbr + x],)
        px2.append(np.concatenate(lst, axis=1))
        lst2 += (px2[y],)
    px = np.concatenate(lst2, axis=0)
    px = np.array(px, 'uint8')

    som_image = Image.fromarray(px)
    #  som_image.show()
    return som_image


def load_som_as_image(path, som):
    img = Dataset(path)
    som.set_som_as_list(img.data)


def display_som_links(som_list, adj):
    px2 = []
    lst2 = ()
    som_list = som_list*255
    for y in range(neuron_nbr):
        lst = ()
        for x in range(neuron_nbr):
            som_list[y * neuron_nbr + x] = som_list[y * neuron_nbr + x].reshape(pictures_dim)
            lst = lst + (som_list[y * neuron_nbr + x],)
            if x < neuron_nbr-1:
                if (adj[y*neuron_nbr+x][y*neuron_nbr+x+1] == 0) or (adj[y*neuron_nbr+x][y*neuron_nbr+x+1] == np.Infinity):
                    lst = lst + (noLink(),)
                else:
                    lst = lst + (hLink(),)
        px2.append(np.concatenate(lst, axis=1))
        lst2 += (px2[2*y],)
        lst = ()
        if y < neuron_nbr-1:
            for x in range(neuron_nbr):
                if (adj[y*neuron_nbr+x][(y+1)*neuron_nbr+x] == 0) or (adj[y*neuron_nbr+x][(y+1)*neuron_nbr+x] == np.Infinity):
                    lst = lst + (noLink(),)
                else:
                    lst = lst + (vLink(),)
                if x < neuron_nbr-1:
                    lst = lst + (noLink(),)
            px2.append(np.concatenate(lst, axis=1))
            lst2 += (px2[2*y+1],)
    px = np.concatenate(lst2, axis=0)
    px = np.array(px, 'uint8')

    som_image = Image.fromarray(px)
    return som_image


def compute_mean_error(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean(np.abs(datamat[i] - SOMList[datacomp[i]]))*255
    return np.mean(error)


def peak_signal_to_noise_ratio(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean((datamat[i] - SOMList[datacomp[i]])**2)
    return 10*np.log10(1/np.mean(error))


def run_from_som():
    img = Dataset("./image/Audrey.png")
    data = img.data
    carte = SOM(data, kohonen())
    load_som_as_image("./results/deep/star_12n_3x3_500epoch_comp.png", carte)
    img.compression(carte, "reconstruction_500epoch.png")
    im2 = display_som(carte.get_som_as_list())
    im2.save(output_path + "som_500epoch.png")

def run(num):
    img = Dataset("./image/Audrey.png")
    data = img.data
    #data = load_image_folder("./image/")

    datacomp = np.zeros(len(data), int)  # datacomp est la liste du numero du neurone vainqueur pour l'imagette correspondante

#    nb_epoch = 20
    epoch_time = len(data)
    nb_iter = epoch_time * epoch_nbr

    carte = SOM(data, kohonen())

    datacomp = carte.winners()
    print("Initial mean error SOM: ", compute_mean_error(datacomp, data, carte.get_som_as_list()))
    print("Initial PSNR SOM : ", peak_signal_to_noise_ratio(datacomp, data, carte.get_som_as_list()))
    for i in range(nb_iter):
         # The training vector is chosen randomly
        if i % epoch_time == 0:
             carte.generate_random_list()

        vect = carte.unique_random_vector()
        carte.train(i, epoch_time, vect)

    datacomp = carte.winners()
    print("Mean pixel error SOM: ", compute_mean_error(datacomp, data, carte.get_som_as_list()))
    print("PSNR SOM: ", peak_signal_to_noise_ratio(datacomp, data, carte.get_som_as_list()))

    img.compression(carte,output_path+"comp"+str(num)+".png")
    im1 = display_som_links(carte.get_som_as_list(),carte.neural_graph.get_binary_adjacency_matrix())
    im1.save(output_path+"carte"+str(num)+".png")

    for p in range(1,10):
        print("FAULT RATE : "+str(p))
        print()
        for e in range(40):
            Fcarte = faulty(carte,p)
            Fdatacomp = Fcarte.winners()
            print("experiment "+str(e))
            print("    Mean pixel error FaultySOM: ", compute_mean_error(Fdatacomp, data, Fcarte.get_som_as_list()))
            print("    PSNR FaultySOM: ", peak_signal_to_noise_ratio(Fdatacomp, data, Fcarte.get_som_as_list()))
    
            img.compression(Fcarte,output_path+"Fcomp"+str(num)+"_FR"+str(p)+"_expe"+str(e)+".png")
            im2 = display_som_links(Fcarte.get_som_as_list(),Fcarte.neural_graph.get_binary_adjacency_matrix())
            im2.save(output_path+"Fcarte"+str(num)+"_FR"+str(p)+"_expe"+str(e)+".png")

# FaultySOM :
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(20,30):
    run(i)
    print()
