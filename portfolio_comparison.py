"""
Code to meake a comparison between portfolios
"""
import time 
import numpy as np
import matplotlib.pyplot as plt

from efficient_frontier import Portfolio
import utilities as ut

start = time.time()

#==============================================================================
# Load data
#==============================================================================

n_asset     = 200  # number of asset in our portfolio (max 431 for 2022.npy)
target_name = "2022"
input_name  = "2023"

# --------------------------- 2022 ---------------------------

list_tk_input    = np.load(f'data/indici_{input_name}.npy', allow_pickle='TRUE')
list_tk_input    = list_tk_input[0:n_asset]

return_1d_input  = np.load(f'data/return_{input_name}.npy', allow_pickle='TRUE')
return_1d_input  = return_1d_input[0:n_asset, :]

cross_corr_input = np.load(f'data/cross_correlation_{input_name}.npy', allow_pickle='TRUE')
cross_corr_input = cross_corr_input[0:n_asset, 0:n_asset]

filtr_corr_input = ut.c_filtering(cross_corr_input, return_1d_input)

sigma_input = np.array([np.sqrt(np.mean(return_1d_input[i, :]**2)-np.mean(return_1d_input[i, :])**2) for i in range(n_asset)])

cross_corr_input = ut.q_operator(cross_corr_input, sigma_input)
filtr_corr_input = ut.q_operator(filtr_corr_input, sigma_input)

# --------------------------- 2023 ---------------------------

list_tk_target    = np.load(f'data/indici_{target_name}.npy', allow_pickle='TRUE')
list_tk_target    = list_tk_target[0:n_asset]

return_1d_target  = np.load(f'data/return_{target_name}.npy', allow_pickle='TRUE')
return_1d_target  = return_1d_target[0:n_asset, :]

cross_corr_target = np.load(f'data/cross_correlation_{target_name}.npy', allow_pickle='TRUE')
cross_corr_target = cross_corr_target[0:n_asset, 0:n_asset]

sigma_target = np.array([np.sqrt(np.mean(return_1d_target[i, :]**2)-np.mean(return_1d_target[i, :])**2) for i in range(n_asset)])

cross_corr_target = ut.q_operator(cross_corr_target, sigma_target)

#==============================================================================
# Optimizzation
#==============================================================================

# Portfolio with cross_correlation.

portfolio_real = Portfolio(return_1d_target, cross_corr_target, list_tk_target)
portfolio_pred = Portfolio(return_1d_target, cross_corr_input,  list_tk_input)

vol_pred, ret_pred, _ = portfolio_pred.optimization(20)
vol_real, ret_real, _ = portfolio_real.optimization(20)

# Portfolio with filtered correlation.

portfolio_pred_filtr = Portfolio(return_1d_target, filtr_corr_input, list_tk_input)

vol_pred_filtr, ret_pred_filtr, _ = portfolio_pred_filtr.optimization(20)

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
plt.plot(vol_pred_filtr, ret_pred_filtr, "y--", label="Predicted filtered")

plt.title("Portfolio optimization comparison", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)

plt.legend(loc='best')
plt.grid()

plt.show()   
