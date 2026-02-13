import numpy as np
import math


class Update_GPD:
    def __init__(self, seed=42):
        self.seed = seed

    def prior_gpd(self, xi_, sig_, nu, kappa, alpha, beta):
        return (np.sqrt(kappa/(2 * np.pi))) * np.exp(-1 / 2 * kappa * (xi_ - nu) ** 2) * beta ** alpha / math.gamma(alpha) * (sig_) ** (alpha - 1) * np.exp(-beta*sig_)

    def posterior_gpd(self, taus, Z, GPDparams, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, y):
        inds = np.unique(Z[:, taus[0]])
        ys = []

        for i in inds:
            rows = np.where(Z[:, taus[0]] == i)[0]

            data_in_rows = []
            for j in rows:
                data_in_rows.append(y[j, taus[0]:taus[1]].tolist())

            flat_row = [item for sublist in data_in_rows for item in sublist]
            ys.append(flat_row)

        reject = 0
        total = 0
        for i in range(len(inds)):
            totali = 0
            (xi, sig) = GPDparams[inds[i]]
            if (1 + xi * (np.max(ys[i]) / sig)) < 0:
                return -9999999
            else:
                for j in range(len(ys[i])):
                    totali = totali + np.log((1 + xi * (ys[i][j] / sig)))
                total = total + totali * (-1 / xi - 1) - np.log(sig) * len(ys[i])
                total = total - (1 / (2 * xi_sig ** 2) * (xi) ** 2) + (1 / scale_sig * np.sqrt(np.pi * 2)) * np.exp(
                    (-1 / 2 * (sig / scale_sig) ** 2)) + np.log(
                    Update_GPD.prior_gpd(self, xi, sig, nu=pri_nu, kappa=pri_kappa, alpha=pri_alpha, beta=pri_beta))
        return total


    def acceptance_ratio_gpd(self, taus, Z, GPDparams_old, GPDparams_new, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, y):
        if Update_GPD.posterior_gpd(self, taus, Z, GPDparams_new, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, y) - Update_GPD.posterior_gpd(self, taus, Z, GPDparams_old, xi_sig,
                                                                                        scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, y) > 709:
            return 10000000
        else:
            return np.exp(
            Update_GPD.posterior_gpd(self, taus, Z, GPDparams_new, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, y) - Update_GPD.posterior_gpd(self, taus, Z, GPDparams_old, xi_sig,
                                                                                        scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, y))

    def Update_GPD_Iterator(self, taus, Z, GPDParams, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, DataMat):

        updated = list(0 for i in range(len(taus)-1))
        for j in range(len(taus)-1):
            inds = np.unique(Z[:, taus[j]])

            propedGPD = GPDParams.copy()
            for k in inds:
                propedGPD[k] = (GPDParams[k][0] + np.random.normal(0, xi_sig),
                                GPDParams[k][1] * np.exp(np.random.normal(0, scale_sig)))

            r = Update_GPD.acceptance_ratio_gpd(self, [taus[j],taus[j+1]], Z, GPDParams, propedGPD, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, DataMat)

            if math.isnan(r):
                r = 0
            rstart = min(1, r)
            U = np.random.uniform(0, 1, 1)
            updated[j] = 0
            if U < rstart:
                GPDParams = propedGPD.copy()
                updated[j] = 1

        return GPDParams, updated