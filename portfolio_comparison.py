import time 
import numpy as np
import matplotlib.pyplot as plt
from efficient_frontier import Portfolio

import utilities as ut

start = time.time()

n_asset    = 50   # number of asset in our portfolio (max 431 for return_2.npy)

# 2022.

list_tk_2022 = np.load('data/indici_2022.npy', allow_pickle='TRUE')
list_tk_2022 = list_tk_2022[0:n_asset]

return_1d_2022 = np.load('data/return_2022.npy', allow_pickle='TRUE')
return_1d_2022 = return_1d_2022[0:n_asset, :]

cross_corr_2022 = np.load('data/cross_correlation_2022.npy', allow_pickle='TRUE')
cross_corr_2022 = cross_corr_2022[0:n_asset, 0:n_asset]

norm_corr_2022 = np.load('data/norm_correlation_2022.npy', allow_pickle='TRUE')
norm_corr_2022 = norm_corr_2022[0:n_asset, 0:n_asset]

filt_corr_2022 = ut.c_filtering(cross_corr_2022, return_1d_2022)

sigma_2022 = np.zeros(n_asset)

for i in range(n_asset):
    sigma_2022[i] = np.sqrt(np.mean(return_1d_2022[i, :]**2)-np.mean(return_1d_2022[i, :])**2)

cross_corr_2022 = ut.q_operator(cross_corr_2022, sigma_2022) 
norm_corr_2022 = ut.q_operator(norm_corr_2022, sigma_2022)
filt_corr_2022 = ut.q_operator(filt_corr_2022, sigma_2022)

# 2023.

list_tk_2023 = np.load('data/indici_2023.npy', allow_pickle='TRUE')
list_tk_2023 = list_tk_2023[0:n_asset]

return_1d_2023 = np.load('data/return_2023.npy', allow_pickle='TRUE')
return_1d_2023 = return_1d_2023[0:n_asset, :]

cross_corr_2023 = np.load('data/cross_correlation_2023.npy', allow_pickle='TRUE')
cross_corr_2023 = cross_corr_2023[0:n_asset, 0:n_asset]

norm_corr_2023 = np.load('data/norm_correlation_2023.npy', allow_pickle='TRUE')
norm_corr_2023 = norm_corr_2023[0:n_asset, 0:n_asset]

filt_corr_2023 = ut.c_filtering(cross_corr_2023, return_1d_2023)


sigma_2023 = np.zeros(n_asset)

for i in range(n_asset):
    sigma_2023[i] = np.sqrt(np.mean(return_1d_2023[i, :]**2)-np.mean(return_1d_2023[i, :])**2)

cross_corr_2023 = ut.q_operator(cross_corr_2023, sigma_2023)
norm_corr_2023 = ut.q_operator(norm_corr_2023, sigma_2023)
filt_corr_2023 = ut.q_operator(filt_corr_2023, sigma_2023)


# Portfolio with cross_correlation.

portfolio_2023_real = Portfolio(return_1d_2023, cross_corr_2023, list_tk_2023)
portfolio_2023_pred = Portfolio(return_1d_2023, cross_corr_2022, list_tk_2023)

vol_pred, ret_pred, _ = portfolio_2023_pred.optimization(20)
vol_real, ret_real, _ = portfolio_2023_real.optimization(20)

# Portfolio with norm_correlation.

portfolio_2023_real_norm = Portfolio(return_1d_2023, norm_corr_2023, list_tk_2023)
portfolio_2023_pred_norm = Portfolio(return_1d_2023, norm_corr_2022, list_tk_2023)

vol_pred_norm, ret_pred_norm, _ = portfolio_2023_pred_norm.optimization(20)
vol_real_norm, ret_real_norm, _ = portfolio_2023_real_norm.optimization(20)

# Portfolio with filtered correlation.

portfolio_2023_real_filt = Portfolio(return_1d_2023, filt_corr_2023, list_tk_2023)
portfolio_2023_pred_filt = Portfolio(return_1d_2023, filt_corr_2022, list_tk_2023)

vol_pred_filt, ret_pred_filt, _ = portfolio_2023_pred_filt.optimization(20)
vol_real_filt, ret_real_filt, _ = portfolio_2023_real_filt.optimization(20)


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


plt.title("Portfolio optimization with cross correlation", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)
plt.legend(loc='best')

# Plot norm.

plt.figure(2)
plt.plot(vol_pred_norm, ret_pred_norm, "r-", label="Predicted")
plt.plot(vol_real_norm, ret_real_norm, "b-", label="Realized")


plt.title("Portfolio optimization with normalized correlation", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)
plt.legend(loc='best')

# Plot filtered.

plt.figure(3)
plt.plot(vol_pred_filt, ret_pred_filt, "r-", label="Predicted")
plt.plot(vol_real_filt, ret_real_filt, "b-", label="Realized")


plt.title("Portfolio optimization with filtered correlation", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)
plt.legend(loc='best')

# Plot cross and filt.

plt.figure(4)
plt.plot(vol_pred_filt, ret_pred_filt, "r-", label="Predicted")
plt.plot(vol_real, ret_real, "b-", label="Realized")


plt.title("Portfolio optimization with cross realized and filtered prediction", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)
plt.legend(loc='best')

plt.show()   



