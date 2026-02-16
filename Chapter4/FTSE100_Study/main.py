import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import genpareto
import os
import csv

from DataGeneration import DataGenerator
Data_generator = DataGenerator()

from Iterator import Iteration
Iter = Iteration()


def save_list_to_file(file_name, my_list):
    with open(file_name, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')


def Data_Zs():

    xi_sig = 0.05
    scale_sig = 0.05

    #### Priors xi ~ N(nu, kappa^-1),  sig ~ Gamma(alpha, beta)
    pri_nu = 0.1
    pri_kappa = 2
    pri_alpha = 3
    pri_beta = 40
    pri_delt = 8

    #Number of time windows
    NumTaus = 3

    FinData = pd.read_csv('FinanceData2.csv')

    FinData['Date'] = pd.to_datetime(FinData['Date'])

    FinData_noTime = FinData.drop(columns=['Date'])
    FinData_noTime = FinData_noTime.transpose()

    K = FinData_noTime.shape[0]
    T = FinData_noTime.shape[1]

    DataMat, Init_Zs, tau_prop = Data_generator.extremeFinData(K=K, T=T, Data = FinData_noTime, NumTaus=NumTaus, threshold=0)

    GPDParams = [(0,0)] * K * NumTaus

    for i in range(len(GPDParams)):
        GPDParams[i] = (0.2, 2)


    alp_tot = [0] * K * NumTaus

    for i in range(NumTaus):
        alp_tot[0 + i*K] = 0.5
        alp_tot[1 + i * K] = 0.5


    return DataMat, Init_Zs, K, T, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, tau_prop, GPDParams, alp_tot, NumTaus

def return_acc_samples(input_list):
    # Convert the input list to a numpy array
    input_array = np.array(input_list)

    # Calculate the differences between consecutive elements
    differences = np.diff(input_array)

    # Include the first element from the input list in the output, as np.diff reduces the size by 1
    output_list = [input_list[0]]

    # Append values from the input list where differences are not equal to 0
    output_list.extend(input_list[i + 1] for i, diff in enumerate(differences) if diff != 0)

    return output_list



if __name__ == '__main__':

    for sim_index in range(1, 6):  # Sim_Final_7 to Sim_Final_11
        Sim_Num = f"Sim_Final_{sim_index}"
        output_dir = f"Data/{Sim_Num}"
        os.makedirs(output_dir, exist_ok=True)


        Datamat, Z, K, T, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, taus, GPDParams, alp_tot, NumTaus = Data_Zs()

        # Save observed exceedance matrix for PPC plotting later
        np.save("DataMat.npy", Datamat)

        print(Z)

        LABEL_FONTSIZE = 22
        TICK_FONTSIZE = 18
        TITLE_FONTSIZE = 22
        LEGEND_FONTSIZE = 18  # not used, but kept for consistency

        plt.rcParams.update({
            "font.size": TICK_FONTSIZE,  # base font
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        LINEWIDTH = 2.5

        # -------------------------
        # Time windows
        # -------------------------
        t1, t2 = 2118, 4128
        T = Datamat.shape[1]

        # -------------------------

        companies = [
            "Sage", "Vodafone", "Elecosoft", "BT", "GB Group",
            "HSBC", "Barclays", "Schroders", "NatWest", "St James's Place",
            "Tesco", "Sainsbury's", "BP", "Rolls Royce Holding", "BAE Systems",
        ]

        ROW_IDX = {
            "Sage": 0,
            "Vodafone": 1,
            "Elecosoft": 2,
            "BT": 3,
            "GB Group": 4,
            "HSBC": 5,
            "Barclays": 6,
            "Schroders": 7,
            "NatWest": 8,
            "St James's Place": 9,
            "Tesco": 10,
            "Sainsbury's": 11,
            "BP": 12,
            "Rolls Royce Holding": 13,
            "BAE Systems": 14,
        }

        fig, axes = plt.subplots(5, 3, figsize=(18, 18), sharex=False)
        axes = axes.ravel()

        for i, name in enumerate(companies):
            ax = axes[i]
            r = ROW_IDX[name]

            ax.plot(range(0, t1), Datamat[r, 0:t1], color="r", linewidth=LINEWIDTH)
            ax.plot(range(t1, t2), Datamat[r, t1:t2], color="g", linewidth=LINEWIDTH)
            ax.plot(range(t2, T), Datamat[r, t2:T], color="b", linewidth=LINEWIDTH)

            ax.set_title(name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Negative Log-returns")
            ax.grid(True, alpha=0.25)

        # In case you ever pass fewer than 15 series, hide unused axes (safe-guard)
        for j in range(len(companies), len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        plt.savefig('all_moving_average.png')
        plt.show()

        Num_Iter = 50000
        burnin = 25000

        n_save_z = 2000
        data_Zs = np.zeros((n_save_z, K, T))

        gpd_traces = {("xi", r, w): [] for r in range(K) for w in range(NumTaus)}
        gpd_traces.update({("sig", r, w): [] for r in range(K) for w in range(NumTaus)})

        components = []
        tau1 = []
        tau2 = []
        oldZ = [[] for _ in range(NumTaus)]

        for i in tqdm(range(Num_Iter)):
            taus, GPDParams, Z, alp_tot, updated_GPD, oldZ, updated_Dim, loglike = Iter.Iterate(
                GPDParams=GPDParams,
                taus=taus,
                Z=Z,
                T=T,
                K=K,
                xi_sig=xi_sig,
                scale_sig=scale_sig,
                pri_delt=pri_delt,
                pri_nu=pri_nu,
                pri_kappa=pri_kappa,
                pri_alpha=pri_alpha,
                pri_beta=pri_beta,
                DataMat=Datamat,
                oldZ=oldZ
            )

            if i > burnin:
                components.append(np.unique(Z).tolist())

                for r in range(K):
                    for w in range(NumTaus):
                        comp = Z[r, taus[w]]
                        gpd_traces[("xi", r, w)].append(GPDParams[comp][0])
                        gpd_traces[("sig", r, w)].append(GPDParams[comp][1])

                if NumTaus >= 2:
                    tau1.append(taus[1])
                if NumTaus >= 3:
                    tau2.append(taus[2])

                if i >= Num_Iter - n_save_z:
                    data_Zs[i - (Num_Iter - n_save_z)] = Z

        # -------------------------
        # Save outputs
        # -------------------------

        for r in range(K):
            for w in range(NumTaus):
                series_id = r + 1
                window_id = w + 1

                xi_path = f"Data/{Sim_Num}/gpdparams{series_id}xi{window_id}.txt"
                sig_path = f"Data/{Sim_Num}/gpdparams{series_id}sig{window_id}.txt"

                save_list_to_file(xi_path, gpd_traces[("xi", r, w)])
                save_list_to_file(sig_path, gpd_traces[("sig", r, w)])

        # Save tau traces
        if len(tau1) > 0:
            save_list_to_file(f"Data/{Sim_Num}/tau1.txt", tau1)
        if len(tau2) > 0:
            save_list_to_file(f"Data/{Sim_Num}/tau2.txt", tau2)

        # Save last 2000 Z draws
        np.save(f"Data/{Sim_Num}/data_Zs.npy", data_Zs)
