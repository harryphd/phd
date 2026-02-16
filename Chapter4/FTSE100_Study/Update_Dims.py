import numpy as np
import random
import math
from scipy.stats import beta, expon
import scipy.special as sc
from Update_Alphas_Zs import Update_Alpha_Z
AlpZGen = Update_Alpha_Z()



class Update_Dim:

    def __init__(self, seed=42):
        self.seed = seed


    def accept_split(self, wjstar, xijstar, sigjstar):
        ### j* split into j1 and j2
        u1 = beta.rvs(2, 2, size=1)[0]
        u2 = expon.rvs(scale=0.025, size=1)[0]
        u3 = beta.rvs(6, 8, size=1)[0]

        ### Get split variables

        wj1, wj2, xij1, xij2, sigj1, sigj2 = Update_Dim.split(self, wjstar, xijstar, sigjstar, u1, u2, u3)

        return wj1, wj2, xij1, xij2, sigj1, sigj2, u1, u2, u3


    def accept_merge(self, wj1, wj2, xij1, xij2, sigj1, sigj2):
        ### j1 and j2 merged into j*. j* occupies the state of j1
        u1 = beta.rvs(2, 2, size=1)[0]
        u2 = expon.rvs(scale=0.025, size=1)[0]
        u3 = beta.rvs(6, 8, size=1)[0]

        ### Get merge variables

        wjstar, xijstar, sigjstar = Update_Dim.merge(self, wj1, wj2, xij1, xij2, sigj1, sigj2)

        return wjstar, xijstar, sigjstar, u1, u2, u3

    def split(self, wjstar, xijstar, sigjstar, u1, u2, u3):

        # wjs
        wj1 = wjstar * u1
        wj2 = wjstar * (1 - u1)

        # xijs
        xij1 = xijstar - u2
        xij2 = xijstar + u2

        # sigjs
        sigj1 = sigjstar * (1 + u3)
        sigj2 = sigjstar * (1 - u3)

        return wj1, wj2, xij1, xij2, sigj1, sigj2


    def merge(self, wj1, wj2, xij1, xij2, sigj1, sigj2):
        # wjstar
        wjstar = wj1 + wj2

        # xijstar
        xijstar = wj1 / wjstar * xij1 + wj2 / wjstar * xij2

        # sigjstar
        sigjstar = wj1 / wjstar * (sigj1) + wj2 / wjstar * (sigj2)

        return wjstar, xijstar, sigjstar


    def Palloc(self, inds, taus, Data, wj1, wj2, xij1, xij2, sigj1, sigj2):
        ### define Pi1 and Pi2

        wjs = [wj1, wj2]
        xis = [xij1, xij2]
        sigs = [sigj1, sigj2]

        Ks1 = []
        Ks2 = []
        Palloc_vec = []

        for k in inds:
            denom = 0
            new_weights = []
            maxval = []
            dat_j = Data[k, taus[0]:taus[1]]
            dat_j = dat_j[dat_j > 0]
            timewindowlikli = []
            for l in range(len(wjs)):
                timewindowlikli.append(AlpZGen.dGPD_weight(wjs[l], xis[l], sigs[l], dat_j))
                maxval.append(timewindowlikli[l])


            const = abs(max(maxval))

            for l in range(len(wjs)):
                denomval = timewindowlikli[l] + const
                if denomval > 708:
                    denom = np.exp(
                        708)
                else:
                    denom = denom + np.exp(
                        denomval)

            for l in range(len(wjs)):
                denomval = timewindowlikli[l] + const
                if denomval > 708:
                    new_weights.append(np.exp(708) / denom)
                else:
                    new_weights.append(
                        np.exp(denomval) / denom)

            D_num = random.choices([0, 1], new_weights)[0]

            if D_num == 0:
                Ks1.append(k)
                Palloc_vec.append(new_weights[0])
            else:
                Ks2.append(k)
                Palloc_vec.append(new_weights[1])

        Palloc_val = np.prod(Palloc_vec)

        return Ks1, Ks2, Palloc_val

    def Rval(self, Data, taus, inds, xijstar, xi1, xi2, sigjstar, sig1, sig2, wjstar, w1, w2, \
             delta, Ks1, Ks2, Palloc_val, nu, kappa, alpha_val, beta_val, u1, u2, u3, ck, sk):

        if len(Ks1) == 0 or len(Ks2) == 0:
            R = 0
        elif xi1 < 0 or xi2 < 0 or xijstar < 0:
            R = 0
        else:
            dat = []
            for k in Ks1:
                dat.append(Data[k, taus[0]:taus[1]])
            data = [item for sublist in dat for item in sublist]
            data = [x for x in data if x > 0]
            totali1 = AlpZGen.dGPD_weight(w1, xi1, sig1, data)
            dat = []
            for k in Ks2:
                dat.append(Data[k, taus[0]:taus[1]])
            data = [item for sublist in dat for item in sublist]
            data = [x for x in data if x > 0]
            totali2 = AlpZGen.dGPD_weight(w2, xi2, sig2, data)
            dat = []
            for k in inds:
                dat.append(Data[k, taus[0]:taus[1]])
            data = [item for sublist in dat for item in sublist]
            data = [x for x in data if x > 0]
            totalis = AlpZGen.dGPD_weight(wjstar, xijstar, sigjstar, data)

            if totalis == 0:
                R = -1
            else:
                likli = (totali1+totali2) - totalis
                if likli > 700:
                    likli = 700
                R = np.exp( likli ) * (
                        w1 ** (delta - 1 + len(Ks1)) * w2 ** (delta - 1 + len(Ks2))) / \
                (wjstar ** (delta - 1 + len(inds)) * sc.beta(delta, len(inds * delta))) * \
                np.sqrt(kappa / 2 * np.pi) * np.exp(
                -1 / 2 * kappa * ((xi1 - nu) ** 2 + (xi2 - nu) ** 2 - (xijstar - nu) ** 2)) * \
                beta_val ** alpha_val / math.gamma(alpha_val) * (sig1 ** (alpha_val - 1) * sig2 ** (alpha_val - 1) / sigjstar ** (alpha_val - 1)) * np.exp(
                -beta_val * (sig1 + sig2 - sigjstar)) * \
                ck / (sk * Palloc_val) * (
                            beta.pdf(u1, 2, 2) * expon.pdf(u2, scale=0.025) * beta.pdf(u3, 6, 8)) ** (-1) * \
                4 * wjstar * sigjstar
        return R


    def RJprop(self, Data, GPDparams, taus, Z, K, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, tot_alp):

        ### split(0) or merge(1)
        if len(np.unique(Z[:, taus[0]])) == 1:
            sm_num = 0
            ck = 0.5
            sk = 1
        elif len(np.unique(Z[:, taus[0]])) == K:
            sm_num = 1
            ck = 1
            sk = 0.5
        elif len(np.unique(Z[:, taus[0]])) == K-1:
            sm_num = random.choices([0, 1], [0.5, 0.5])[0]
            ck = 1
            sk = 0.5
        else:
            sm_num = random.choices([0, 1], [0.5, 0.5])[0]
            ck = 0.5
            sk = 0.5

        ### split
        if sm_num == 0:
            ### choose component
            comp = random.choices(np.unique(Z[:, taus[0]]))[0]
            rows = np.where(Z[:, taus[0]] == comp)[0]
            weight = tot_alp[comp]
            (xijst, sigjst) = GPDparams[comp]

            ### find split params
            wj1, wj2, xij1, xij2, sigj1, sigj2, u1, u2, u3 = Update_Dim.accept_split(self, wjstar=weight, xijstar=xijst, sigjstar=sigjst)

            ### find new cluster alloc
            Ks1, Ks2, Palloc_val = Update_Dim.Palloc(self, inds=rows, taus=taus, Data=Data, wj1=wj1, wj2=wj2, xij1=xij1, xij2=xij2,
                                          sigj1=sigj1, sigj2=sigj2)

            ### find r
            r_val = Update_Dim.Rval(self, Data=Data, taus=taus, inds=rows, xijstar=xijst, xi1=xij1, xi2=xij2, sigjstar=sigjst, \
                         sig1=sigj1, sig2=sigj2, wjstar=weight, w1=wj1, w2=wj2, delta=pri_delt, Ks1=Ks1, Ks2=Ks2, \
                         Palloc_val=Palloc_val, nu=pri_nu, kappa=pri_kappa, alpha_val=pri_alpha, beta_val=pri_beta, \
                         u1=u1, u2=u2, u3=u3, ck=ck, sk=sk)

            if r_val == -1:
                rstart = 0
            else:
                rstart = min(1, r_val)
            U = np.random.uniform(0, 1, 1)
            if U < rstart:
                return 0, 1, (comp, Ks1, Ks2), (wj1, wj2, xij1, xij2, sigj1, sigj2)
            else:
                return 0, 0, comp, ()

        ### merge
        if sm_num == 1:
            ### choose component
            comp1 = random.choice(np.unique(Z[:, taus[0]]))
            rows1 = np.where(Z[:, taus[0]] == comp1)[0]
            comp2 = random.choice(np.unique(Z[:, taus[0]]))
            while comp2 == comp1:
                comp2 = random.choice(np.unique(Z[:, taus[0]]))
            rows2 = np.where(Z[:, taus[0]] == comp2)[0]
            wj1 = tot_alp[comp1]
            wj2 = tot_alp[comp2]
            (xij1, sigj1) = GPDparams[comp1]
            (xij2, sigj2) = GPDparams[comp2]

            ### find split params
            wjstar, xijst, sigjst, u1, u2, u3 = Update_Dim.accept_merge(self, wj1=wj1, wj2=wj2, xij1=xij1, xij2=xij2, sigj1=sigj1,
                                                             sigj2=sigj2)
            both_rows = np.concatenate((rows1, rows2), axis=None)
            ### find new cluster alloc
            Ks1, Ks2, Palloc_val = Update_Dim.Palloc(self, inds=both_rows, taus=taus, Data=Data, wj1=wj1, wj2=wj2, xij1=xij1,
                                          xij2=xij2, sigj1=sigj1, sigj2=sigj2)

            ### find r
            r_val = Update_Dim.Rval(self, Data=Data, taus=taus, inds=both_rows, xijstar=xijst, xi1=xij1, xi2=xij2, sigjstar=sigjst, \
                         sig1=sigj1, sig2=sigj2, wjstar=wjstar, w1=wj1, w2=wj2, delta=pri_delt, Ks1=Ks1, Ks2=Ks2, \
                         Palloc_val=Palloc_val, nu=pri_nu, kappa=pri_kappa, alpha_val=pri_alpha, beta_val=pri_beta, \
                         u1=u1, u2=u2, u3=u3, ck=ck, sk=sk)

            # print(r_val)

            if r_val == -1:
                rstart = 0
            elif r_val == 0:
                rstart = 1
            else:
                rstart = min(1, r_val ** (-1))
            U = np.random.uniform(0, 1, 1)
            if U < rstart:
                return 1, 1, (comp1, comp2, Ks1, Ks2), (wjstar, xijst, sigjst)
            else:
                return 1, 0, (comp1, comp2), ()


    def Update_Dim_Iterator(self, Data, GPDparams, taus, Z, K, T, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, tot_alp):
        updated_dims = list(0 for i in range(len(taus)-1))
        for j in range(len(taus) - 1):
            updated_dims[j] = 0
            type, accept, components, params = Update_Dim.RJprop(self, Data, GPDparams, [taus[j],taus[j+1]], Z, K, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, tot_alp)

            if type == 0 and accept == 1:
                updated_dims[j] = 1
                newind = min([x for x in range(K*j,K*(1+j)) if x not in np.unique(Z[:, taus[j]]).tolist()])
                tot_alp[components[0]] = params[0]
                tot_alp[newind] = params[1]

                GPDparams[components[0]] = (params[2], params[4])
                GPDparams[newind] = (params[3], params[5])

                for i in range(K):
                    for k in range(T):
                        if Z[i,k] == components[0] and i in components[2]:
                            Z[i,k] = newind

            if type == 1 and accept == 1:
                updated_dims[j] = 1
                tot_alp[components[0]] = params[0]
                tot_alp[components[1]] = 0

                GPDparams[components[0]] = (params[1], params[2])
                GPDparams[components[1]] = (0, 0)

                for i in range(K):
                    for k in range(T):
                        if Z[i,k] == components[1] and i in components[2]:
                            Z[i,k] = components[0]
                        if Z[i,k] == components[1] and i in components[3]:
                            Z[i,k] = components[0]

        return GPDparams, Z, tot_alp, updated_dims


