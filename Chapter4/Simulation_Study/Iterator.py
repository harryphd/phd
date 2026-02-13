from Update_Taus_New import Update_Tau
from Update_GPDs import Update_GPD
from Update_Alphas_Zs import Update_Alpha_Z
from Update_Dims import  Update_Dim

TauGen = Update_Tau()
GPDGen = Update_GPD()
AlpZGen = Update_Alpha_Z()
DimGen = Update_Dim()


class Iteration:
    def __init__(self, seed=42):
        self.seed = seed

    def Iterate(self, GPDParams, taus, Z, T, K, xi_sig, scale_sig, pri_delt, pri_nu, pri_kappa, pri_alpha, pri_beta, DataMat, oldZ):


        taus, Z = TauGen.Update_Tau_Iterator(taus, GPDParams, Z, T, K, DataMat)

        GPDParams, updated_GPD = GPDGen.Update_GPD_Iterator(taus, Z, GPDParams, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, DataMat)


        Z, tot_alp, oldZ = AlpZGen.Update_Z_Iterator(taus, Z, K, GPDParams, pri_delt, DataMat, oldZ)

        GPDParams, Z, tot_alp, updated_Dim = DimGen.Update_Dim_Iterator(DataMat, GPDParams, taus, Z, K, T, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, tot_alp)

        return taus, GPDParams, Z, tot_alp, updated_GPD, oldZ, updated_Dim