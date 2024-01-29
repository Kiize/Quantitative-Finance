import time 
import numpy as np
import matplotlib.pyplot as plt
from efficient_frontier import Portfolio

start = time.time()

n_asset    = 434   # number of asset in our portfolio (max 431 for return_2.npy)

# 2022.

list_tk_2022 = np.load('data/indici_2022.npy', allow_pickle='TRUE')
list_tk_2022 = list_tk_2022[0:n_asset]

return_1d_2022 = np.load('data/return_2022.npy', allow_pickle='TRUE')
return_1d_2022 = return_1d_2022[0:n_asset, :]

cross_corr_2022 = np.load('data/cross_correlation_2022.npy', allow_pickle='TRUE')
cross_corr_2022 = cross_corr_2022[0:n_asset, 0:n_asset]

#cross_corr_2022 = np.cov(return_1d_2022)


# 2023.

list_tk_2023 = np.load('data/indici_2023.npy', allow_pickle='TRUE')
list_tk_2023 = list_tk_2023[0:n_asset]

return_1d_2023 = np.load('data/return_2023.npy', allow_pickle='TRUE')
return_1d_2023 = return_1d_2023[0:n_asset, :]

cross_corr_2023 = np.load('data/cross_correlation_2023.npy', allow_pickle='TRUE')
cross_corr_2023 = cross_corr_2023[0:n_asset, 0:n_asset]

#cross_corr_2023 = np.cov(return_1d_2023)

portfolio_2023_real = Portfolio(return_1d_2023, cross_corr_2023, list_tk_2023)
portfolio_2023_pred = Portfolio(return_1d_2023, cross_corr_2022, list_tk_2023)

vol_pred, ret_pred, _ = portfolio_2023_pred.optimization(20)
vol_real, ret_real, _ = portfolio_2023_real.optimization(20)

#data = data[data > 0].dropna(how='all')

#data.to_csv('data/portfolios.csv', index=True)

mins = (time.time()-start)//60
sec  = (time.time()-start) % 60

print(f"Elapsed time: {mins} min {sec:.2f} sec")

#==============================================================================
# Plot Efficiet frontier
#==============================================================================


plt.figure(1)
plt.plot(vol_pred, ret_pred, "r-", label="Predicted")
plt.plot(vol_real, ret_real, "b-", label="Realized")


plt.title("Portfolio optimization", fontsize=10)
plt.xlabel('Volatility %', fontsize=10)
plt.ylabel('Return %', fontsize=10)
plt.legend(loc='best')
plt.show()   



