import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from build_dataset import load

#==============================================================================
# Portfolio creation
#==============================================================================

def portfolio(w, returns, ret_cov):
    '''
    Returns the mean and standard deviation of returns for a portfolio
    
    Parameter
    ---------
    w : 1darray
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
    mean_prt = sum( w * mean_ret) * days_in_yr
    devs_prt = np.sqrt((w.T @ ret_cov @ w) * days_in_yr)
    
    return mean_prt, devs_prt


def negative_sharpe_ratio(w, returns, ret_cov, risk_free_rate=0.):
    '''
    Function to compute the sharpe ratio, and we want to maximize it
    so we add a minus and then minimize th negative sharpe ratio.
    
    Parameter
    ---------
    w : 1darray
        weights of portfolio
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    risk_free_rate : float, optional, default 0
        The risk-free rate is the rate of return on an investment
        that has a zero chance of loss. It means the investment 
        is so safe that there is no risk associated with it.
    
    Returns
    -------
    sharpe_ratio : float
        It measures the additional return above the
        return of the risk-free asset per unit of volatility
    '''
    ret_prt, vol_prt = portfolio(w, returns, ret_cov)
    sharpe_ratio = (ret_prt - risk_free_rate)/vol_prt 

    return -sharpe_ratio


def max_sharpe_ratio(returns, ret_cov, risk_free_rate=0.):
    '''
    Function that maximize the sharpe ratio,
    to find the optimal weights for our portfolio
    
    Parameter
    ---------
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    risk_free_rate : float, optional, default 0
        The risk-free rate is the rate of return on an investment
        that has a zero chance of loss. It means the investment 
        is so safe that there is no risk associated with it.
    
    Returns
    -------
    result : dict
        result of minimizzation
    '''
    # Number of asset in oru portfolio
    n = returns.shape[0]
    # Extra argumento to pass to the objective function
    args = (returns, ret_cov, risk_free_rate)
    # Unitarity constraint
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.})
    # All weights must be between 0 and 1, they are percentages
    bounds = tuple((0, 1) for _ in range(n))
    # Initial guess
    init = n * [1/n]
    # minimizzation
    result = opt.minimize(negative_sharpe_ratio, init, args=args,
                          bounds=bounds, constraints=constraints)

    return result


def portfolio_vol(w, returns, ret_cov):
    return portfolio(w, returns, ret_cov)[1]


def min_vol(returns, ret_cov):

    # Number of asset in oru portfolio
    n = returns.shape[0]
    # Extra argumento to pass to the objective function
    args = (returns, ret_cov)
    # Unitarity constraint
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.})
    # All weights must be between 0 and 1, they are percentages
    bounds = tuple((0, 1) for _ in range(n))
    # Initial guess
    init = n * [1/n]
    # minimizzation
    result = opt.minimize(portfolio_vol, init, args=args,
                          bounds=bounds, constraints=constraints)

    return result

def portfolio_ret(w, returns, ret_cov):
    return portfolio(w, returns, ret_cov)[0]


def frontier(returns, ret_cov, target):

    # Number of asset in oru portfolio
    n = returns.shape[0]
    # Extra argumento to pass to the objective function
    args = (returns, ret_cov)
    # Unitarity constraint
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                   {'type': 'eq', 'fun': lambda w: portfolio_ret(w, returns, ret_cov) - target})
    # All weights must be between 0 and 1, they are percentages
    bounds = tuple((0, 1) for _ in range(n))
    # Initial guess
    init = n * [1/n]
    # minimizzation
    result = opt.minimize(portfolio_vol, init, args=args,
                          bounds=bounds, constraints=constraints)

    return result


def get_result(returns, ret_cov, N=20, risk_free_rate=0.):
    '''
    Function tu compute the return of the portfolio
    and save in a pandas dataframe the result.
    
    Parameter
    ---------
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    risk_free_rate : float, optional, default 0
        The risk-free rate is the rate of return on an investment
        that has a zero chance of loss. It means the investment 
        is so safe that there is no risk associated with it.
    
    Returns
    -------
    ret_prt_sr : float
        return of the portfolio for maximum sharpe ratio
    vol_prt_sr : float
        volatility of the portfolio for maximum sharpe ratio
    sr_data : pandas DataFrame
        DataFrame with optimal weights and associated tiker
    '''
    # Maximization
    result_sr = max_sharpe_ratio(returns, ret_cov, risk_free_rate)
    # Compute the return of our portfolio
    ret_prt_sr, vol_prt_sr = portfolio(result_sr['x'], returns, ret_cov)

    # Minimization
    result_mv = min_vol(returns, ret_cov)
    # Compute the return of our portfolio
    ret_prt_mv, vol_prt_mv = portfolio(result_mv['x'], returns, ret_cov)

    # Frontier
    frontier_vol = []
    data = pd.DataFrame(index=list_tk, columns=[f"{i}" for i in range(N)])
    target = np.linspace(ret_prt_mv, ret_prt_sr, N)

    for i,t in enumerate(target):
        tmp = frontier(returns, ret_cov, t)
        frontier_vol.append(tmp['fun'])
        data[f"{i}"] = [round(i*100, 1) for i in tmp['x']]

    return 100*np.array(frontier_vol), 100*target, data

if __name__ == '__main__':
    #==============================================================================
    # Pre-loaded data
    #==============================================================================
    """
    n_asset    = 100    # number of asset in our portfolio (max 431 for return_2.npy)
    n_days     = 252  # number of day (max 1005 for return_2.npy)
    days_in_yr = 252    # day of activity


    list_tk = np.load('data/indici_2.npy', allow_pickle='TRUE')
    list_tk = list_tk[0:n_asset]


    return_1d = np.load('data/return_2.npy', allow_pickle='TRUE')
    return_1d = return_1d[0:n_asset, 0:n_days]


    cross_corr = np.cov(return_1d)
    #"""
    #==============================================================================
    # Load data
    #==============================================================================
    #"""
    days_in_yr  = 252    # day of activity
    start_date  = "2020-01-01"
    end_date    = "2023-12-31"

    list_tk = ["^GSPC", "AAPL", "^IXIC", "EBAY"]
    history = load(start_date, end_date, list_ticker=list_tk)

    return_1d = []
    for i, ticker in enumerate(list_tk):
        # Compute the normalized return to 1 day
        adj_close = history[ticker]['Adj Close']
        return_1d.append(adj_close.pct_change()[1:])

    return_1d  = np.array(return_1d)
    cross_corr = np.cov(return_1d)
    #"""

    #==============================================================================
    # -------------------
    #==============================================================================

    frontier_vol, frontier_ret, data = get_result(return_1d, cross_corr)

    print(data)
    data.to_csv('data/portfolios.csv', index=True)

    #==============================================================================
    # Plot Efficiet frontier
    #==============================================================================

    # Create random portfolio for comparison
    n_portfolios = 20000
    ret = np.zeros(n_portfolios)
    vol = np.zeros(n_portfolios)
    for i in range(n_portfolios):
        w = np.random.rand(cross_corr.shape[0]) # All w_i > 0
        w = w / sum(w)        # sum(W) must be 1
        ret[i], vol[i] = portfolio(w, return_1d, cross_corr)

    sr = ret/vol # sharpe ratio

    plt.figure(1)
    plt.plot(frontier_vol, frontier_ret, "r-", label="Frontier")


    plt.scatter(vol*100, ret*100, c=sr, cmap='plasma')
    plt.colorbar(label="sharpe ratio")
    plt.xlabel('Volatility %')
    plt.ylabel('Return %')
    plt.legend(loc='best')
    plt.show()







    
