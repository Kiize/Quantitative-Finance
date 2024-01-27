"""
The purpose of the code is to calculate the efficient frontier in
the profit vs risk plan using the Markowitz theory of portfolio optimization.
To do this, we first compute, always via optimisation, the portfolio associated
with the maximum Sharpe ratio and then the portfolio associated with the minimum
volatility.
These two portfolios set the minimum and the maximum return of the efficient frontier.
So we now minimize volatility with fixed returns. 
"""
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

days_in_yr = 252 # day of activity

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
    init = n * [1 / n]
    # minimizzation
    result = opt.minimize(negative_sharpe_ratio, init, args=args,
                          bounds=bounds, constraints=constraints)
    return result


def portfolio_ret(w, returns, ret_cov):
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
    '''
    return portfolio(w, returns, ret_cov)[0]


def portfolio_vol(w, returns, ret_cov):
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
    devs_prt : float
        standard deviation of portfolio 
    '''
    return portfolio(w, returns, ret_cov)[1]


def min_vol(returns, ret_cov):
    '''
    Function that minimize the portfolio's volatility,
    to find the optimal weights.
    
    Parameter
    ---------
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    
    Returns
    -------
    result : dict
        result of minimizzation
    '''
    # Number of asset in oru portfolio
    n = returns.shape[0]
    # Extra argumento to pass to the objective function
    args = (returns, ret_cov)
    # Unitarity constraint
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.})
    # All weights must be between 0 and 1, they are percentages
    bounds = tuple((0, 1) for _ in range(n))
    # Initial guess
    init = n * [1 / n]
    # minimizzation
    result = opt.minimize(portfolio_vol, init, args=args,
                          bounds=bounds, constraints=constraints)
    return result


def frontier(returns, ret_cov, target):
    '''
    Function that minimize the portfolio's volatility,
    to find the optimal weights.
    
    Parameter
    ---------
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    target : float
        expected return
    
    Returns
    -------
    result : dict
        result of minimizzation
    '''
    # Number of asset in oru portfolio
    n = returns.shape[0]
    # Extra argumento to pass to the objective function
    args = (returns, ret_cov)
    # Unitarity constraint and return constraint
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                   {'type': 'eq', 'fun': lambda w: portfolio_ret(w, returns, ret_cov) - target})
    # All weights must be between 0 and 1, they are percentages
    bounds = tuple((0, 1) for _ in range(n))
    # Initial guess
    init = n * [1 / n]
    # minimizzation
    result = opt.minimize(portfolio_vol, init, args=args,
                          bounds=bounds, constraints=constraints)
    return result


def get_result(returns, ret_cov, ticker, N_P=20, risk_free_rate=0.):
    '''
    Function tu compute the return of the portfolio
    and save in a pandas dataframe the result.
    
    Parameter
    ---------
    returns : 2darray
        matrix of return N asset for L time
    ret_cov : 2darray
        corss correlation matrix N x N
    ticker : list
        list of ticker of our portfolio
    N_P : int, optinal, default 20
        number of portfolios on the efficient frontier
    risk_free_rate : float, optional, default 0
        The risk-free rate is the rate of return on an investment
        that has a zero chance of loss. It means the investment 
        is so safe that there is no risk associated with it.
    
    Returns
    -------
    frontier_vol : 1darray
        result of optimizzation for volatility (x-axis)
    target : 1darray
        return that we want for our potfolio (y-axis)
    data : pandas DataFrame
        DataFrame with optimal weights and associated ticker
    '''
    
    print("Start computation...")
    # Maximization of sharpe ratio
    result_sr = max_sharpe_ratio(returns, ret_cov, risk_free_rate)
    # Compute the return of our portfolio
    ret_prt_sr, vol_prt_sr = portfolio(result_sr['x'], returns, ret_cov)
    print("Maximization of sharpe ratio done")

    # Minimization of volatility
    result_mv = min_vol(returns, ret_cov)
    # Compute the return of our portfolio
    ret_prt_mv, vol_prt_mv = portfolio(result_mv['x'], returns, ret_cov)
    print("Minimization of volatility done")

    # Frontier
    frontier_vol = []
    target       = np.linspace(ret_prt_mv, ret_prt_sr, N_P)
    
    # Dataframe to store all porfolios
    data = pd.DataFrame(index=ticker, columns=[f"{i}" for i in range(N_P)])
    
    print("Start frontier computation...")
    # computation of frontier
    for i, t in enumerate(target):
        tmp = frontier(returns, ret_cov, t)
        frontier_vol.append(tmp['fun'])
        data[f"{i}"] = [round(i*100, 2) for i in tmp['x']]
        print(f"{(i+1)/N_P * 100:.2f} % \r", end="")
    
    # percentage return
    return 100*np.array(frontier_vol), 100*target, data


if __name__ == '__main__':

    from build_dataset import load

    #==============================================================================
    # Pre-loaded data
    #==============================================================================
    """
    n_asset    = 50   # number of asset in our portfolio (max 431 for return_2.npy)
    n_days     = 1006   # number of day (max 1005 for return_2.npy)
    
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
        # Compute the return to 1 day
        adj_close = history[ticker]['Adj Close']
        return_1d.append(adj_close.pct_change()[1:])
        # normalizzation
        #sig = np.sqrt(np.mean(return_1d[i]**2)-np.mean(return_1d[i])**2)
        #return_1d[i] = (return_1d[i] - np.mean(return_1d[i]))/sig 

    return_1d  = np.array(return_1d)
    cross_corr = np.cov(return_1d)
    #"""

    #==============================================================================
    # Computation
    #==============================================================================

    frontier_vol, frontier_ret, data = get_result(return_1d, cross_corr, list_tk)
    data = data[data > 0].dropna(how='all')
    
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
        w = w / sum(w)                          # sum(W) must be 1
        ret[i], vol[i] = portfolio(w, return_1d, cross_corr)

    sr = ret/vol # sharpe ratio

    plt.figure(1)
    plt.plot(frontier_vol,     frontier_ret,     "r-", label="Efficent frontier")
    plt.plot(frontier_vol[0],  frontier_ret[0],  "bo", label="Minimum volatility")
    plt.plot(frontier_vol[-1], frontier_ret[-1], "ko", label="Maximum sharpe ratio")
    
    plt.scatter(vol*100, ret*100, c=sr, cmap='plasma')
    plt.colorbar(label="sharpe ratio")
    
    plt.title("Portfolio optimizzation", fontsize=10)
    plt.xlabel('Volatility %', fontsize=10)
    plt.ylabel('Return %', fontsize=10)
    plt.legend(loc='best')
    plt.show()   
