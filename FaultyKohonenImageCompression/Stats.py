from Model import *
import itertools
import multiprocessing as mp

connexions_matrices = {"koh": kohonen(), "sw": small_worlds(), "star": star()}

class Run:
    def __init__(self, eS, eE, sS, sE, connexion):
        self.conn = connexion
        self.epoch_nbr = epoch_nbr
        self.epsilon_start = eS
        self.epsilon_end = eE
        self.sigma_start = sS
        self.sigma_end = sE
        self.mean = -1
        self.psnr = -1
        self.diff = -1
        self.comp = -1

    def run_fitness(self, data, width):
        epoch_time = len(data)
        nb_iter = epoch_time * self.epoch_nbr

        som = SOM(data, connexions_matrices[self.conn], self.epsilon_start, self.epsilon_end, self.sigma_start, self.sigma_end, self.epoch_nbr)
        for i in range(nb_iter):
            if i % epoch_time == 0:
                som.generate_random_list()
            vector = som.unique_random_vector()
            som.train(i, epoch_time, vector)
        data_comp = som.winners()
        self.mean = compute_mean_error(data_comp, data, som.get_som_as_list())
        self.psnr = peak_signal_to_noise_ratio(data_comp, data, som.get_som_as_list())
        self.diff, self.comp = Dataset.compute_compression_ratio(data, som, width, data_comp)
        print(self.to_string())

    def to_string(self):
        res = self.conn+";"
        res += str(self.epsilon_start)+";"+str(self.epsilon_end)+";"
        res += str(self.sigma_start)+";"+str(self.sigma_end)+";"
        res += str(self.mean)+";"+str(self.psnr)+";"
        res += str(self.diff)+";"+str(self.comp)+"\n"
        return res


class FullTest:
    def __init__(self):
        self.img = Dataset("./image/Audrey.png")
        self.data = self.img.data
        self.current = []
        connex = ("koh", "sw", "star")
        eS = (1, 0.3)
        eE = (0.1, 0.01)
        sS = (0.6, 0.1)
        sE = (0.01, 0.001)
        for i in eS:
            for j in eE:
                for k in sS:
                    for l in sE:
                        for m in connex:
                            self.current.append(Run(i, j, k, l, m))

    def run(self):
        pool = mp.Pool(len(self.current))
        self.current = pool.starmap(Run.run_fitness, zip(self.current, itertools.repeat(self.data), itertools.repeat(self.img.nb_pictures[1])))
        pool.close()
        pool.join()
