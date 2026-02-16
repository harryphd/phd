import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from scipy.stats import genpareto, norm, gamma
from scipy.stats.mstats import mquantiles
from arch import arch_model


class DataGenerator:

    def __init__(self, seed=47):
        self.seed = seed


    def moving_80(self, data):
        # Set the window size for the moving quantile (60-day window)
        window_size = 60

        # Initialize a list to store the moving quantiles for each row
        moving_quantiles_list = []

        # Calculate the moving 80th percentile for each row
        for i in range(data.shape[0]):
            row = data[i, :]
            quantiles = []
            for j in range(len(row)):
                if j < window_size // 2 or j >= len(row) - window_size // 2:
                    # Take top 2 maximum points in the first and last 30 points
                    if j < window_size // 2:
                        subset = row[:window_size]
                    else:
                        subset = row[-window_size:]
                    top_2_max = np.partition(subset, -2)[-2:]
                    quantile_val = np.nanpercentile(top_2_max, 80)
                else:
                    quantile_val = np.nanpercentile(
                        row[j - window_size // 2: j + window_size // 2 + 1], 80
                    )
                quantiles.append(quantile_val)
            moving_quantiles_list.append(quantiles)

        # Convert the list of moving quantiles to a numpy array
        moving_quantiles_array = np.array(moving_quantiles_list)

        # Compute the exceedances (X - Q80) where X > Q80, else NaN
        exceedances = data - moving_quantiles_array
        exceedances[data <= moving_quantiles_array] = np.nan

        return exceedances

    def moving_quantile_threshold(self, data: np.ndarray, q: float = 0.8, window_size: int = 60):
        """
        Compute a moving quantile threshold u_{k,t} for each series k at each time t.

        This generalises moving_95_3, including the same edge handling:
        for the first/last half-window, it uses the top-2 values in the edge window
        and takes the q-quantile of those two points (so it stays "extreme-focused" at the edges).

        Returns
        -------
        u : array, shape (K, T)
            Moving thresholds for each series across time.
        """
        data = np.asarray(data, dtype=float)
        K, T = data.shape
        half = window_size // 2

        u = np.zeros((K, T), dtype=float)

        for i in range(K):
            row = data[i, :]
            for t in range(T):
                if t < half:
                    subset = row[:window_size]
                    # top-2 max points
                    top_2 = np.partition(subset, -2)[-2:]
                    u[i, t] = np.nanpercentile(top_2, q * 100.0)
                elif t >= T - half:
                    subset = row[-window_size:]
                    top_2 = np.partition(subset, -2)[-2:]
                    u[i, t] = np.nanpercentile(top_2, q * 100.0)
                else:
                    subset = row[t - half: t + half + 1]
                    u[i, t] = np.nanpercentile(subset, q * 100.0)

        return u

    def exceedances_from_threshold(self, data: np.ndarray, u: np.ndarray):
        """
        Convert raw data into exceedances (data - u) when data > u, else NaN.
        """
        data = np.asarray(data, dtype=float)
        u = np.asarray(u, dtype=float)
        exc = data - u
        exc[data <= u] = np.nan
        return exc

    def fixed_quantile_threshold(self, data: np.ndarray, q: float = 0.8):
        """
        Fixed threshold per series: u_k is the q-quantile of series k across time.
        Returns u with shape (K, 1) for convenience.
        """
        data = np.asarray(data, dtype=float)
        u_k = np.nanpercentile(data, q * 100.0, axis=1)  # shape (K,)
        return u_k.reshape(-1, 1)

    def build_exceedances(self, data: np.ndarray, method: str, q: float, window_size: int = None):
        """
        Unified entry point for building exceedances used by the MCMC.

        Parameters
        ----------
        method : {"moving", "fixed"}
        q      : quantile in (0,1) e.g. 0.8 or 0.9
        window_size : only used for method="moving"

        Returns
        -------
        exc : array (K, T)
            Exceedance matrix (data - u) where data > u else NaN.
        u   : array
            Thresholds used:
              - moving: (K, T)
              - fixed : (K, 1)
        """
        if method not in {"moving", "fixed"}:
            raise ValueError("method must be 'moving' or 'fixed'")

        if method == "moving":
            if window_size is None:
                raise ValueError("window_size is required for method='moving'")
            u = self.moving_quantile_threshold(data, q=q, window_size=window_size)  # (K,T)
            exc = self.exceedances_from_threshold(data, u)
            return exc, u

        # fixed
        u = self.fixed_quantile_threshold(data, q=q)  # (K,1)
        # broadcast to (K,T)
        u_bt = np.repeat(u, data.shape[1], axis=1)
        exc = self.exceedances_from_threshold(data, u_bt)
        return exc, u


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
        for i in range(K):
            y = []
            for k in range(NumTau):

                GeneratedShapes[i, k] = norm.rvs(loc=Shapes[k][i], scale=0.03, random_state=self.seed + i + 8 * k)
                GeneratedScales[i, k] = gamma.rvs(100, scale=1 / Scales[k][i], random_state=self.seed + i + 8 * k)

                y1 = genpareto.rvs(GeneratedShapes[i, k], loc=0, scale=GeneratedScales[i, k],
                                   size=TimeWindowVec[k + 1] - TimeWindowVec[k], random_state=self.seed + i).tolist()
                y = y + y1

                Z_prop[i, tau_prop[k]:tau_prop[k + 1]] = random.choice([k*K, k*K+1])

            DataMat[i, :] = y

        print(GeneratedShapes)
        print(GeneratedScales)

        return DataMat, Z_prop, tau_prop


    def extremeFinData(self, K, T, Data, NumTaus, threshold):

        # if NumTau != len(Shapes):
        #     raise ValueError("Number of time windows and dimension of input shape and scale parameters are mismatched.")

        Z_prop = np.zeros(shape=(K, T), dtype='uint8')

        tau_prop = [int((T / NumTaus) * i) for i in range(NumTaus)] + [T]

        for i in range(K):
            for k in range(NumTaus):
                Z_prop[i, tau_prop[k]:tau_prop[k + 1]] = random.choice([k*K, k*K+1])

        FinData_noTime = Data.values

        FinData_noTime = self.moving_80(FinData_noTime)


        return FinData_noTime, Z_prop, tau_prop





if __name__ == '__main__':
    NumTaus = 3

    FinData = pd.read_csv('FinanceData2.csv')
    # FinData = FinData.iloc[:, :3]
    row_mask = (FinData.index >= 1443) & (FinData.index <= 3732)
    FinData['Date'] = pd.to_datetime(FinData['Date'])
    FinCapped = FinData[row_mask]
    FinData_noTime = FinCapped.drop(columns=['Date'])
    FinData_noTime = FinData_noTime.transpose()


    K = FinData_noTime.shape[0]
    T = FinData_noTime.shape[1]

    Data_generator = DataGenerator()

    DataMat, Init_Zs, tau_prop = Data_generator.extremeFinData(K=K, T=T, Data = FinData_noTime, NumTaus=NumTaus, threshold=0)

    print(np.count_nonzero(~np.isnan(DataMat[0, :])))

