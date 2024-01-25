"""
Random Portfolio simulation
"""
import numpy as np
import matplotlib.pyplot as plt

from build_dataset import load

np.random.seed(69420)

#=============================================================
#====================== pre-loaded data ======================
#=============================================================
"""
n_asset    = 4      # number of asset in our portfolio (max 431 for return_2.npy)
n_days     = 1006   # number of day (max 1005 for return_2.npy)
days_in_yr = 252    # day of activity

return_1d  = np.load('data/return_2.npy', allow_pickle='TRUE')
return_1d  = return_1d[0:n_asset, 0:n_days]
cross_corr = np.cov(return_1d)
"""
#=============================================================
#========================= load data =========================
#=============================================================

days_in_yr = 252    # day of activity
start_date = "2020-01-01"
end_date   = "2023-12-31"

list_tk = ["^GSPC", "AAPL", "^IXIC", "EBAY"]
history = load(start_date, end_date, list_ticker=list_tk)

return_1d = []
for i, ticker in enumerate(list_tk):
    # Compute the normalized return to 1 day
    adj_close = history[ticker]['Adj Close']
    return_1d.append(adj_close.pct_change()[1:])

return_1d  = np.array(return_1d)
cross_corr = np.cov(return_1d)

#=============================================================
#========================= Portfolio =========================
#=============================================================

def random_weights(n):
    '''
    Produces random weights
    
    Parameters
    ----------
    n : int
        size of portfolio, number of asset
    
    Return
    ------
    w : 1darray
        uniform random weights
    '''
    w = np.random.rand(n) # All w_i > 0
    w = w / sum(w)        # sum(W) must be 1
    return w
    

def portfolio(weights, returns, ret_cov):
    '''
    Returns the mean and standard deviation of returns for a portfolio
    
    Parameter
    ---------
    weights : 1darray
        weights of portfolio
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    
    Returns
    -------
    mean_prt : float
        mean of portfolio return
    devs_prt : float
        standard deviation of portfolio 
    '''
    mean_ret = np.mean(returns, axis=1)
    mean_prt = sum( weights * mean_ret) * days_in_yr
    devs_prt = np.sqrt((weights.T @ ret_cov @ weights) * days_in_yr)
    
    return mean_prt, devs_prt

#=============================================================
#=========================== Plot ============================
#=============================================================

n_portfolios = 20000

ret = np.zeros(n_portfolios)
vol = np.zeros(n_portfolios)
for i in range(n_portfolios):
     w = random_weights(cross_corr.shape[0])
     ret[i], vol[i] = portfolio(w, return_1d, cross_corr)

sr = ret/vol # sharpe ratio

plt.figure(1)
plt.scatter(vol*100, ret*100, c=sr, cmap='plasma')
plt.colorbar(label="sharpe ratio")
plt.xlabel('Volatility %')
plt.ylabel('Return %')
plt.title('Randomly generated portfolios')
plt.show()
