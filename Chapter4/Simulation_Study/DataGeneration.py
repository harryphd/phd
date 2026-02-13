import numpy as np
import random
from scipy.stats import genpareto, norm, gamma


class DataGenerator:
    def __init__(self, seed=47):
        self.seed = seed

    def generate_data(self, Taus, K, T, Shapes, Scales, NumTaus):
        NumTau = len(Taus)

        if NumTau != len(Shapes):
            raise ValueError("Number of time windows and dimension of input shape and scale parameters are mismatched.")

        TimeWindowVec = [0] + Taus
        DataMat = np.zeros(shape=(K, T))

        GeneratedShapes = np.zeros(shape=(K, NumTau))
        GeneratedScales = np.zeros(shape=(K, NumTau))

        Z_prop = np.zeros(shape=(K, T), dtype='uint8')

        tau_prop = [int((T/NumTaus)*i) for i in range(NumTaus)] + [T]

        # tau_prop = [0, 650, 700, 1000]

        for i in range(K):
            y = []
            for k in range(NumTau):

                GeneratedShapes[i, k] = norm.rvs(loc=Shapes[k][i], scale=0.015, random_state=self.seed + i + 7 * k)
                GeneratedScales[i, k] = norm.rvs(loc=Scales[k][i], scale=0.1, random_state=self.seed + i + 8 * k)
                #GeneratedScales[i, k] = gamma.rvs(100, scale=1 / Scales[k][i], random_state=self.seed + i + 8 * k)

                y1 = genpareto.rvs(GeneratedShapes[i, k], loc=0, scale=GeneratedScales[i, k],
                                   size=TimeWindowVec[k + 1] - TimeWindowVec[k], random_state=self.seed + i).tolist()
                y = y + y1

                Z_prop[i, tau_prop[k]:tau_prop[k + 1]] = random.choice([k*K, k*K+1])

            DataMat[i, :] = y

        print(GeneratedShapes)
        print(GeneratedScales)

        return DataMat, Z_prop, tau_prop


