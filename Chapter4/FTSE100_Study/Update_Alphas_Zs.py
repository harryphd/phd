import numpy as np
import math
import random
from scipy.stats import dirichlet

class Update_Alpha_Z:
    def __init__(self, seed=42):
        self.seed = seed

    def Counts(self, taus, GPDParams, Z, oldZ, j):
        inds = np.unique(Z[:, taus])

        if oldZ[j]:
            inds = oldZ[j]
        for i in inds:
            if GPDParams[i][1] == 0:
                inds.remove(i)

        countinds = [0] * len(inds)
        ind_params = [0] * len(inds)

        for ii in range(len(inds)):
            countinds[ii] = list(Z[:, taus]).count(inds[ii])
            ind_params[ii] = GPDParams[inds[ii]]

        return countinds, ind_params, inds

    def Update_Alpha_Iterator(self, pri_delt, taus, GPDParams, Z, tot_alp, oldZ, j):
        countinds, ind_params, cur_inds = Update_Alpha_Z.Counts(self, taus, GPDParams, Z, oldZ, j)
        newcounts = [x + pri_delt for x in countinds]
        alp = dirichlet.rvs(newcounts)[0].tolist()
        for i in range(len(cur_inds)):
            for k in range(max(cur_inds)+1):
                if k == cur_inds[i]:
                    tot_alp[k] = alp[i]
        return alp, countinds, ind_params, tot_alp, cur_inds

    def dGPD_weight(self, alpy, xi, sig, dat):

        totali=0

        if (1 + xi * (np.max(dat) / sig)) <= 0:
            return -999999
        else:
            for m in range(len(dat)):
                logged = (1 + xi * (dat[m] / sig))
                totali = totali + np.log(logged)
            total = totali * (-1 / xi - 1) - np.log(sig) * len(dat)
            total = total + np.log(alpy)
        return total


    def Update_Z_Iterator(self, taus, Z, K, GPDParams, pri_delt, DataMat, oldZ):
        tot_alp = [0] * K * (len(taus)-1)
        for j in range(len(taus) - 1):
            removed_inds = []
            alp, countinds, ind_params, tot_alp, cur_inds = Update_Alpha_Z.Update_Alpha_Iterator(self, pri_delt, taus[j], GPDParams, Z, tot_alp, oldZ, j)

            for i in range(len(cur_inds)):
                tot_alp[cur_inds[i]] = alp[i]

            for k in range(K):
                denom = 0
                new_weights = []
                maxval = []
                dat_j = DataMat[k, taus[j]:taus[j + 1]]
                dat_j = dat_j[dat_j>0]
                timewindowlikli = []
                for l in range(len(alp)):
                    timewindowlikli.append(Update_Alpha_Z.dGPD_weight(self, alp[l], ind_params[l][0], ind_params[l][1], dat_j))
                    maxval.append(timewindowlikli[l])

                const = abs(max(maxval))

                for l in range(len(alp)):
                    denomval = timewindowlikli[l] + const
                    if denomval > 708:
                        denom = np.exp(
                            708)
                    else:
                        denom = denom + np.exp(denomval)

                for l in range(len(alp)):
                    denomval = timewindowlikli[l] + const

                    if denomval > 708:
                        new_weights.append(np.exp(708) / denom)
                    else:
                        new_weights.append(np.exp(denomval) / denom)


                D_num = random.choices(cur_inds, new_weights)[0]
                Z[k, taus[j]:taus[j + 1]] = [D_num] * (taus[j + 1] - taus[j])

            if oldZ[j] == np.unique(Z[:, taus[j]]).tolist():
                oldZ[j] = []

            if len(alp) != len(np.unique(Z[:, taus[j]])):
                removed_inds.append([item for item in cur_inds if item not in np.unique(Z[:, taus[j]]).tolist()])
                oldZ[j] = sorted(np.unique(Z[:, taus[j]]).tolist() + removed_inds[0])


        return Z, tot_alp, oldZ
