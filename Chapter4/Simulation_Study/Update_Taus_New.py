import numpy as np
import random
import math
from scipy.stats import geom


class Update_Tau:

    def __init__(self, seed=42):
        self.seed = seed

    def find_taus(self, Zs):
        z = Zs[0, :]
        ztest1 = Zs[1, :]
        ztest2 = Zs[2, :]
        ztest3 = Zs[3, :]
        ztest4 = Zs[4, :]
        tau_locs = [1] + np.where(z[:-1] != z[1:])[0]
        if len(tau_locs) == len([1] + np.where(ztest1[:-1] != ztest1[1:])[0]) == len(
                [1] + np.where(ztest2[:-1] != ztest2[1:])[0]) == len(
                [1] + np.where(ztest3[:-1] != ztest3[1:])[0]) == len([1] + np.where(ztest4[:-1] != ztest4[1:])[0]):
            return tau_locs.tolist()
        else:
            raise ValueError("Find Tau")

    def is_greater_than_threshold(self, val, threshold):
        return val > threshold

    def likelihood_tau(self, tau, Z, GPDparams, y):
        if tau[0] < tau[1]:
            uniqvals = np.unique(Z[:, tau[0]])
            ys = []
            flat_ys = []
            for i in range(len(uniqvals)):
                ys.append(
                    y[np.where(Z[:, tau[0]] == uniqvals[i])[0], tau[0]: tau[1]].tolist()
                )

            for i in range(len(uniqvals)):
                flat_ys.append(sum(ys[i], []))

            total = 0
            for i in range(len(uniqvals)):
                totali = 0
                (xi, sig) = GPDparams[
                    uniqvals[i]
                ]
                for j in range(len(ys[i])):
                    totali = totali + np.log((1 + xi * (flat_ys[i][j] / sig)))
                total = total + totali * (-1 / xi - 1) - np.log(sig) * len(flat_ys[i])
        elif tau[0] == tau[1]:
            total = 0
        else:
            uniqvals = np.unique(Z[:, tau[1]])
            ys = []
            flat_ys = []
            for i in range(len(uniqvals)):
                ys.append(
                    y[np.where(Z[:, tau[1]] == uniqvals[i])[0], tau[1]: tau[0]].tolist()
                )

            for i in range(len(uniqvals)):
                flat_ys.append(sum(ys[i], []))

            total = 0
            for i in range(len(uniqvals)):
                totali = 0
                (xi, sig) = GPDparams[
                    uniqvals[i]
                ]
                for j in range(len(ys[i])):
                    totali = totali + np.log((1 + xi * (flat_ys[i][j] / sig)))
                total = total + totali * (-1 / xi - 1) - np.log(sig) * len(flat_ys[i])
        return total


    def acceptance_ratio_tau(self, lookuptau, GPDparams, Z, Zprop, y):
        if Update_Tau.likelihood_tau(self,lookuptau, Zprop, GPDparams, y) - Update_Tau.likelihood_tau(self,lookuptau, Z, GPDparams, y) > 709:
            return 10000000
        else:
            return np.exp(
            (Update_Tau.likelihood_tau(self,lookuptau, Zprop, GPDparams, y)) - (Update_Tau.likelihood_tau(self,lookuptau, Z, GPDparams, y)))

    def check_tau(self, prop_tau, T):
        original = [0] + [item[0] for item in prop_tau] + [T]
        newtaus = [0] + [item[1] for item in prop_tau] + [T]
        if sorted(newtaus) == newtaus:
            if not np.where(np.diff(newtaus) < 10)[0].tolist():
                return prop_tau
            else:
                Diffs = np.where(np.diff(newtaus) < 10)[0]
                for i in range(len(Diffs)):
                    if newtaus[Diffs[i]] < 10:
                        newtaus[Diffs[i] + 1] = 10
                    else:
                        newtaus[Diffs[i]] = original[Diffs[i]]
                        newtaus[Diffs[i] + 1] = original[Diffs[i] + 1]
                newlist = []
                for bb in range(len(newtaus) - 2):
                    newlist.append([original[bb + 1], newtaus[bb + 1]])
                return newlist
        else:
            for bb in range(len(newtaus) - 2):
                if newtaus[bb + 1] <= newtaus[bb] or newtaus[bb + 1] >= newtaus[bb + 2]:
                    newtaus[bb + 1] = original[bb + 1]
            newlist = []
            for bb in range(len(newtaus) - 2):
                newlist.append([original[bb + 1], newtaus[bb + 1]])
            return newlist


    def Update_Tau_Iterator(self, taus, GPDParams, Z, T, K, DataMat):
        if len(taus) == 1:
            taus = [0, T]
            return taus, Z
        else:
            prop_tau = []
            for j in range(len(Update_Tau.find_taus(self,Z))):
                direc = random.choices([1, -1], [0.5, 0.5])[0]
                listind = direc
                if listind == 1:
                    listind = 0

                NumScip = geom.rvs(0.95)
                prop_tau.append([Update_Tau.find_taus(self,Z)[j], Update_Tau.find_taus(self,Z)[j] + direc * NumScip])

            prop_tau = Update_Tau.check_tau(self, prop_tau, T)

            ZNew = Z.copy()
            GPDParamsNew = GPDParams.copy()

            for j in range(len(Update_Tau.find_taus(self,Z))):
                if j == 0:
                    boo = 0
                if prop_tau[j][1] - Update_Tau.find_taus(self,Z)[j] < 0:
                    ZNew[:, prop_tau[j][1]: Update_Tau.find_taus(self,Z)[j]] = np.reshape(
                        np.repeat(
                            ZNew[:, Update_Tau.find_taus(self,Z)[j]],
                            (Update_Tau.find_taus(self,Z)[j] - prop_tau[j][1]),
                            axis=0,
                        ),
                        (K, Update_Tau.find_taus(self,Z)[j] - prop_tau[j][1]),
                    )

                    r = Update_Tau.acceptance_ratio_tau(self,
                        prop_tau[j], GPDParams, Z, ZNew, DataMat
                    )
                    if math.isnan(r):
                        r = 0
                    rstart = min(1, r)
                    U = np.random.uniform(0, 1, 1)
                    if U < rstart:
                        Z = ZNew.copy()

                else:
                    ZNew[
                    :,
                    Update_Tau.find_taus(self,Z)[j]: Update_Tau.find_taus(self,Z)[j]
                                     + (prop_tau[j][1] - Update_Tau.find_taus(self,Z)[j]),
                    ] = np.reshape(
                        np.repeat(
                            ZNew[:, Update_Tau.find_taus(self,Z)[j] - 1],
                            (prop_tau[j][1] - Update_Tau.find_taus(self,Z)[j]),
                            axis=0,
                        ),
                        (K, prop_tau[j][1] - Update_Tau.find_taus(self,Z)[j]),
                    )

                    r = Update_Tau.acceptance_ratio_tau(self,
                        prop_tau[j], GPDParams, Z, ZNew, DataMat
                    )
                    if math.isnan(r):
                        r = 0
                    rstart = min(1, r)
                    U = np.random.uniform(0, 1, 1)
                    if U < rstart:
                        Z = ZNew.copy()

            taus = [0] + [Update_Tau.find_taus(self,Z)[j] for j in range(len(Update_Tau.find_taus(self,Z)))] + [T]

            return taus, Z