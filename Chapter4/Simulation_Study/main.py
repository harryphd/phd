import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from DataGeneration import DataGenerator
from Iterator import Iteration

# Initialize external modules
data_generator = DataGenerator()
iter_obj = Iteration()


def Data_Zs():
    """Generate data and simulation parameters."""
    K = 25
    T = 1000
    # Define change points
    Taus = [280, 600, T]
    xi_sig = 0.1
    scale_sig = 0.2

    # Priors: xi ~ N(nu, kappa^-1), sig ~ Gamma(alpha, beta)
    pri_nu = 0.15
    pri_kappa = 2
    pri_alpha = 3
    pri_beta = 1
    pri_delt = 8

    NumTaus = 3

    # Input shapes and scales for three time windows
    InputShapes = [
        [0.15, 0.15, 0.05, 0.15, 0.05, 0.15, 0.15, 0.05, 0.15, 0.05],
        [0.1, 0.2, 0.1, 0.03, 0.03, 0.1, 0.03, 0.2, 0.03, 0.2],
        [0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1]
    ]
    InputScales = [
        [1, 1, 3, 1, 3, 1, 1, 3, 1, 3],
        [2.5, 1, 2.5, 1.5, 1.5, 2.5, 1.5, 1, 1.5, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    ]


    DataMat, Init_Zs, tau_prop = data_generator.generate_data(
        Taus=Taus, K=K, T=T, Shapes=InputShapes, Scales=InputScales, NumTaus=NumTaus
    )

    # Initialize GPD parameters for each series/time-window
    GPDParams = [(0, 0)] * (K * NumTaus)
    GPDParams[0] = (0.6, 1)
    GPDParams[1] = (0.2, 5)
    GPDParams[25] = (0.4, 5)
    GPDParams[26] = (0.1, 1)
    GPDParams[50] = (0.2, 5)
    GPDParams[51] = (0.4, 1)

    # Initialize alpha values (only two nonzero values per time window)
    alp_tot = [0] * (K * NumTaus)
    for i in range(NumTaus):
        alp_tot[0 + i * K] = 0.5
        alp_tot[1 + i * K] = 0.5

    return DataMat, Init_Zs, K, T, xi_sig, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, Taus, GPDParams, alp_tot, NumTaus


def plot_initial_time_series(DataMat, Taus, T):
    """Plot the simulated time series (first 6 series)."""
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(hspace=0.5)
    num_plots = 6  # plot first six series

    for i in range(num_plots):
        plt.subplot(3, 2, i + 1)
        plt.plot(range(0, Taus[0]), DataMat[i, :Taus[0]], color='r')
        plt.plot(range(Taus[0], Taus[1]), DataMat[i, Taus[0]:Taus[1]], color='g')
        plt.plot(range(Taus[1], T), DataMat[i, Taus[1]:T], color='b')
        plt.title(f"Time Series {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Simulated Data")
    plt.tight_layout()
    # plt.show()


def return_acc_samples(input_list):
    """
    Return a list of accepted samples where a change occurred.
    This function keeps the first sample then only adds a value if it differs from the previous one.
    """
    input_array = np.array(input_list)
    differences = np.diff(input_array)
    output_list = [input_list[0]]
    output_list.extend(item for diff, item in zip(differences, input_list[1:]) if diff != 0)
    return output_list


def save_list_to_file(file_name, my_list):
    """Save a list of items to a text file (one item per line)."""
    with open(file_name, 'w') as file:
        for item in my_list:
            file.write(f"{item}\n")


def run_simulation():
    # Generate data and parameters
    (DataMat, Z, K, T, xi_sig, scale_sig, pri_nu, pri_kappa,
     pri_alpha, pri_beta, pri_delt, Taus, GPDParams, alp_tot, NumTaus) = Data_Zs()

    # Optional: print initial cluster assignments
    print("Initial Z:", Z)

    # Plot the simulated time series
    plot_initial_time_series(DataMat, Taus, T)

    # Set MCMC parameters
    num_iter = 80000
    burnin = 40000

    # ---- NEW: settings for saving 2000 Z's after 5000 iterations ----
    save_after_iter = 3000  # start recording after this many iterations
    num_z_to_save = 2000  # number of Z samples to store
    data_Zs = None  # will become (2000, K, T)
    z_fill_idx = 0
    data_zs_saved = False

    # Where to save

    data_zs_path = "data_Zs.npy"
    # ---------------------------------------------------------------

    # Containers for storing results:
    # For each series (0 to K-1) and each time-window (0 to NumTaus-1) we keep lists for xi and sigma
    gpd_samples = {
        series: {
            tau_idx: {'xi': [], 'sig': []} for tau_idx in range(NumTaus)
        } for series in range(K)
    }
    # Store tau values (for tau[1] and tau[2]) over iterations
    tau_samples = {1: [], 2: []}
    # List to store the unique cluster components from Z at each iteration (optional)
    components = []
    # oldZ is maintained across iterations (one for each time window)
    oldZ = [[] for _ in range(NumTaus)]

    # Main iteration loop
    for it in tqdm(range(num_iter), desc="MCMC iterations"):
        Taus, GPDParams, Z, alp_tot, updated_GPD, oldZ, updated_Dim = iter_obj.Iterate(
            GPDParams=GPDParams,
            taus=Taus,
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
            DataMat=DataMat,
            oldZ=oldZ
        )

        # ---- NEW: collect 2000 Z's starting AFTER iteration 5000 ----
        if (it > save_after_iter) and (not data_zs_saved):
            if data_Zs is None:
                data_Zs = np.zeros((num_z_to_save, K, T), dtype=Z.dtype)

            data_Zs[z_fill_idx] = Z  # store current Z
            z_fill_idx += 1

            # Save immediately once filled
            if z_fill_idx >= num_z_to_save:
                np.save(data_zs_path, data_Zs)
                data_zs_saved = True
        # ------------------------------------------------------------

        # Save current tau values (here index 1 and 2 of Taus)
        tau_samples[1].append(Taus[1])
        tau_samples[2].append(Taus[2])

        # After burn-in, store GPD parameter samples for each series and time-window
        if it > burnin:
            components.append(np.unique(Z).tolist())
            for series in range(K):
                for tau_idx in range(NumTaus):
                    # Use Z[series, Taus[tau_idx]] as the index for the GPDParams list
                    idx = Z[series, Taus[tau_idx]]
                    xi_val, sig_val = GPDParams[idx]
                    gpd_samples[series][tau_idx]['xi'].append(xi_val)
                    gpd_samples[series][tau_idx]['sig'].append(sig_val)

    # Get accepted samples by filtering out duplicate consecutive values
    accepted_gpd_samples = {
        series: {
            tau_idx: {
                'xi': return_acc_samples(gpd_samples[series][tau_idx]['xi']),
                'sig': return_acc_samples(gpd_samples[series][tau_idx]['sig'])
            }
            for tau_idx in range(NumTaus)
        }
        for series in range(K)
    }
    accepted_tau1 = return_acc_samples(tau_samples[1])
    accepted_tau2 = return_acc_samples(tau_samples[2])

    return gpd_samples, accepted_gpd_samples, tau_samples, accepted_tau1, accepted_tau2, components, K, NumTaus


def save_simulation_results(sim_num, gpd_samples, tau_samples):
    """Save GPD parameters and tau values to files.

    Files are saved in a directory Data/sim_num/ with filenames including the series and tau indices.
    """
    base_dir = os.path.join("Data", sim_num)
    os.makedirs(base_dir, exist_ok=True)
    K = len(gpd_samples)
    NumTaus = len(gpd_samples[0])

    for series in range(K):
        for tau_idx in range(NumTaus):
            xi_file = os.path.join(base_dir, f"gpdparams_series{series + 1}_tau{tau_idx + 1}_xi.txt")
            sig_file = os.path.join(base_dir, f"gpdparams_series{series + 1}_tau{tau_idx + 1}_sig.txt")
            save_list_to_file(xi_file, gpd_samples[series][tau_idx]['xi'])
            save_list_to_file(sig_file, gpd_samples[series][tau_idx]['sig'])

    # Save tau samples
    save_list_to_file(os.path.join(base_dir, "tau1.txt"), tau_samples[1])
    save_list_to_file(os.path.join(base_dir, "tau2.txt"), tau_samples[2])


def plot_results(accepted_gpd_samples, accepted_tau1, accepted_tau2):
    """Plot a few histograms for selected parameters and tau values."""
    # Example: Plot histograms for series 1 (index 0) and series 3 (index 2), time-window 1 (tau index 0)
    # Adjust the “simulated parameter” vertical lines as needed.
    plt.figure()
    plt.hist(accepted_gpd_samples[0][0]['xi'], bins=20)
    plt.axvline(np.mean(accepted_gpd_samples[0][0]['xi']), color='k', linestyle='dashed', linewidth=1,
                label='Mean of Samples')
    plt.axvline(0.47455972, color='r', linestyle='dashed', linewidth=1, label='Simulated Parameter')
    plt.xlabel('Posterior Samples')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title("Series 1, Tau 1 - xi")

    plt.figure()
    plt.hist(accepted_gpd_samples[0][0]['sig'], bins=20)
    plt.axvline(np.mean(accepted_gpd_samples[0][0]['sig']), color='k', linestyle='dashed', linewidth=1,
                label='Mean of Samples')
    plt.axvline(0.91438162, color='r', linestyle='dashed', linewidth=1, label='Simulated Parameter')
    plt.xlabel('Posterior Samples')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title("Series 1, Tau 1 - sigma")

    plt.figure()
    plt.hist(accepted_gpd_samples[2][0]['xi'], bins=20)  # Series 3, Tau 1
    plt.axvline(np.mean(accepted_gpd_samples[2][0]['xi']), color='k', linestyle='dashed', linewidth=1,
                label='Mean of Samples')
    plt.axvline(0.11870523, color='r', linestyle='dashed', linewidth=1, label='Simulated Parameter')
    plt.xlabel('Posterior Samples')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title("Series 3, Tau 1 - xi")

    plt.figure()
    plt.hist(accepted_gpd_samples[2][0]['sig'], bins=20)  # Series 3, Tau 1
    plt.axvline(np.mean(accepted_gpd_samples[2][0]['sig']), color='k', linestyle='dashed', linewidth=1,
                label='Mean of Samples')
    plt.axvline(4.4805496, color='r', linestyle='dashed', linewidth=1, label='Simulated Parameter')
    plt.xlabel('Posterior Samples')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title("Series 3, Tau 1 - sigma")

    plt.figure()
    plt.hist(accepted_tau1, bins=20)
    plt.axvline(np.mean(accepted_tau1), color='k', linestyle='dashed', linewidth=1, label='Mean of Samples')
    plt.axvline(280, color='r', linestyle='dashed', linewidth=1, label='Simulated Parameter')
    plt.xlabel('Posterior Samples')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title("Tau1 Samples")

    plt.figure()
    plt.hist(accepted_tau2, bins=20)
    plt.axvline(np.mean(accepted_tau2), color='k', linestyle='dashed', linewidth=1, label='Mean of Samples')
    plt.axvline(600, color='r', linestyle='dashed', linewidth=1, label='Simulated Parameter')
    plt.xlabel('Posterior Samples')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.title("Tau2 Samples")


    K = len(accepted_gpd_samples)
    NumTaus = len(accepted_gpd_samples[0])
    component_counts = [0] * (K * NumTaus)

    plt.figure()
    plt.bar(range(K * NumTaus), component_counts)
    plt.xlabel("Component")
    plt.ylabel("Count")
    plt.title("Component Counts")
    # plt.show()



def main():
    base_sim_num = "Sim_"
    num_runs = 1
    xi_sig_values = np.linspace(0.08, 0.1, num_runs)  # Vary xi_sig between 0.055 and 0.065
    pri_kappas = [2]

    for i in range(num_runs):
        sim_num = f"{base_sim_num}{i + 80}"
        pri_kappa = pri_kappas[i]

        # Modify the Data_Zs function to accept xi_sig as an argument
        def Data_Zs_mod(xi_sig_val):
            """Generate data and simulation parameters with varying xi_sig."""
            K = 10
            T = 1000
            Taus = [280, 600, T]
            scale_sig = 0.25

            pri_nu = 0.1
            # pri_kappa = 2
            pri_alpha = 3
            pri_beta = 1
            pri_delt = 10

            NumTaus = 3
            InputShapes = [
                [0.15, 0.15, 0.05, 0.15, 0.05, 0.15, 0.15, 0.05, 0.15, 0.05],
                [0.1, 0.2, 0.1, 0.03, 0.03, 0.1, 0.03, 0.2, 0.03, 0.2],
                [0.15, 0.03, 0.15, 0.03, 0.15, 0.03, 0.15, 0.03, 0.15, 0.03]
            ]
            InputScales = [
                [1, 1, 3, 1, 3, 1, 1, 3, 1, 3],
                [2.5, 1, 2.5, 1.5, 1.5, 2.5, 1.5, 1, 1.5, 1],
                [1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1]
            ]

            DataMat, Init_Zs, tau_prop = data_generator.generate_data(
                Taus=Taus, K=K, T=T, Shapes=InputShapes, Scales=InputScales, NumTaus=NumTaus
            )

            GPDParams = [(0, 0)] * (K * NumTaus)
            GPDParams[0] = (0.6, 1)
            GPDParams[1] = (0.2, 5)
            GPDParams[10] = (0.4, 5)
            GPDParams[11] = (0.1, 1)
            GPDParams[20] = (0.2, 5)
            GPDParams[21] = (0.4, 1)

            alp_tot = [0] * (K * NumTaus)
            for j in range(NumTaus):
                alp_tot[0 + j * K] = 0.5
                alp_tot[1 + j * K] = 0.5

            # -------------------------
            # Styling (match your Malaysia/Hill plot)
            # -------------------------
            LABEL_FONTSIZE = 22
            TICK_FONTSIZE = 18
            TITLE_FONTSIZE = 22
            LEGEND_FONTSIZE = 18  # not used (no legend), but kept for consistency

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
            # True clusters (given)
            # -------------------------
            clusters = {
                "w1": {1: [1, 2, 4, 6, 7, 9], 2: [3, 5, 8, 10]},
                "w2": {1: [1, 3, 6], 2: [2, 8, 10], 3: [4, 5, 7, 9]},
                "w3": {1: [1, 3, 5, 7, 9], 2: [2, 4, 6, 8, 10]},
            }

            def invert_membership(cluster_dict, K=10):
                """Return array series_to_cluster[0..K-1] = cluster_id (ints)."""
                out = np.zeros(K, dtype=int)
                for cid, members in cluster_dict.items():
                    for s in members:
                        out[s - 1] = cid
                return out

            K = 10
            w1_id = invert_membership(clusters["w1"], K=K)
            w2_id = invert_membership(clusters["w2"], K=K)
            w3_id = invert_membership(clusters["w3"], K=K)

            # -------------------------
            colors_w1 = {1: "tab:blue", 2: "tab:orange"}
            colors_w2 = {1: "tab:green", 2: "tab:red", 3: "tab:purple"}
            colors_w3 = {1: "tab:brown", 2: "tab:pink"}

            # -------------------------
            # Plot simulated data (DataMat shape assumed (10,1000))
            # -------------------------
            taus = [280, 600, 1000]  # (0,280), (280,600), (600,1000)

            fig = plt.figure(figsize=(12, 14))
            plt.subplots_adjust(hspace=0.7, wspace=0.35)

            for s in range(K):
                ax = plt.subplot(5, 2, s + 1)

                # Window 1 segment
                ax.plot(
                    range(0, taus[0]),
                    DataMat[s, 0:taus[0]],
                    linewidth=LINEWIDTH,
                    color=colors_w1[int(w1_id[s])]
                )

                # Window 2 segment
                ax.plot(
                    range(taus[0], taus[1]),
                    DataMat[s, taus[0]:taus[1]],
                    linewidth=LINEWIDTH,
                    color=colors_w2[int(w2_id[s])]
                )

                # Window 3 segment
                ax.plot(
                    range(taus[1], taus[2]),
                    DataMat[s, taus[1]:taus[2]],
                    linewidth=LINEWIDTH,
                    color=colors_w3[int(w3_id[s])]
                )

                ax.set_title(f"Time Series {s + 1}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Simulated Data")
                ax.grid(True, alpha=0.25)

            fig.tight_layout()
            plt.show()

            return DataMat, Init_Zs, K, T, xi_sig_val, scale_sig, pri_nu, pri_kappa, pri_alpha, pri_beta, pri_delt, Taus, GPDParams, alp_tot, NumTaus

        global Data_Zs  # Override the original function
        Data_Zs = lambda: Data_Zs_mod(xi_sig_values[i])



        # Run the simulation
        (gpd_samples,
         accepted_gpd_samples,
         tau_samples,
         accepted_tau1,
         accepted_tau2,
         components,
         K,
         NumTaus) = run_simulation()

        # Save results with a unique identifier
        save_simulation_results(sim_num, gpd_samples, tau_samples)

        # Plot results
        plot_results(accepted_gpd_samples, accepted_tau1, accepted_tau2)


if __name__ == '__main__':
    main()

