import time 
import numpy as np
import matplotlib.pyplot as plt

from efficient_frontier import Portfolio
import utilities as ut

start = time.time()

n_asset = 200  # number of asset in our portfolio (max 431 for return_2.npy)

# 2022.

list_tk_2022 = np.load('data/indici_16_19.npy', allow_pickle='TRUE')
list_tk_2022 = list_tk_2022[0:n_asset]

return_1d_2022 = np.load('data/return_16_19.npy', allow_pickle='TRUE')
return_1d_2022 = return_1d_2022[0:n_asset, :]

cross_corr_2022 = np.load('data/cross_correlation_16_19.npy', allow_pickle='TRUE')
cross_corr_2022 = cross_corr_2022[0:n_asset, 0:n_asset]

filt_corr_2022 = ut.c_filtering(cross_corr_2022, return_1d_2022)

sigma_2022 = np.array([np.sqrt(np.mean(return_1d_2022[i, :]**2)-np.mean(return_1d_2022[i, :])**2) for i in range(n_asset)])

cross_corr_2022 = ut.q_operator(cross_corr_2022, sigma_2022)
filt_corr_2022  = ut.q_operator(filt_corr_2022, sigma_2022)


# 2023.

list_tk_2023 = np.load('data/indici_20_23.npy', allow_pickle='TRUE')
list_tk_2023 = list_tk_2023[0:n_asset]

return_1d_2023 = np.load('data/return_20_23.npy', allow_pickle='TRUE')
return_1d_2023 = return_1d_2023[0:n_asset, :]

cross_corr_2023 = np.load('data/cross_correlation_20_23.npy', allow_pickle='TRUE')
cross_corr_2023 = cross_corr_2023[0:n_asset, 0:n_asset]

sigma_2023 = np.array([np.sqrt(np.mean(return_1d_2023[i, :]**2)-np.mean(return_1d_2023[i, :])**2) for i in range(n_asset)])

cross_corr_2023 = ut.q_operator(cross_corr_2023, sigma_2023)



# Portfolio with cross_correlation.

portfolio_2023_real = Portfolio(return_1d_2023, cross_corr_2023, list_tk_2023)
portfolio_2023_pred = Portfolio(return_1d_2023, cross_corr_2022, list_tk_2023)

vol_pred, ret_pred, _ = portfolio_2023_pred.optimization(20)
vol_real, ret_real, _ = portfolio_2023_real.optimization(20)

# Portfolio with filtered correlation.

portfolio_2023_pred_filt = Portfolio(return_1d_2023, filt_corr_2022, list_tk_2023)

vol_pred_filt, ret_pred_filt, _ = portfolio_2023_pred_filt.optimization(20)

# Elapsed time.

mins = (time.time()-start)//60
sec  = (time.time()-start) % 60

print(f"Elapsed time: {mins} min {sec:.2f} sec")

#==============================================================================
# Plot Efficiet frontier
#==============================================================================

# Plot cross.

plt.figure(1)
plt.plot(vol_pred, ret_pred, "r-", label="Predicted")
plt.plot(vol_real, ret_real, "b-", label="Realized")
plt.plot(vol_pred_filt, ret_pred_filt, "y--", label="Predicted filtered")

plt.title("Portfolio optimization comparison", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)

plt.legend(loc='best')
plt.grid()

plt.show()   



