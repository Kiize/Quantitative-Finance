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


class Portfolio:
    '''
    Class to create optimal portfolio according
    Markowitz theory of portfolio optimization
    
    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt

    from build_dataset import load
    from efficient_portfolio import Portfolio

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

    return_1d  = np.array(return_1d)
    cross_corr = np.cov(return_1d)

    portfolio = Portfolio(return_1d, cross_corr, list_tk, days_in_yr=days_in_yr)
    frontier_vol, frontier_ret, data = portfolio.optimization(20)
    data = data[data > 0].dropna(how='all')

    data.to_csv('data/portfolios.csv', index=True)

    plt.figure(1)
    plt.plot(frontier_vol,     frontier_ret,     "r-", label="Efficent frontier")
    plt.plot(frontier_vol[0],  frontier_ret[0],  "bo", label="Minimum volatility")
    plt.plot(frontier_vol[-1], frontier_ret[-1], "ko", label="Maximum sharpe ratio")
    plt.title("Portfolio optimizzation", fontsize=10)
    plt.xlabel('Volatility %', fontsize=10)
    plt.ylabel('Return %', fontsize=10)
    plt.legend(loc='best')
    plt.show()
    '''

    def __init__(self, returns, ret_cov, ticker, risk_free_rate=0, days_in_yr=252):
        '''
        returns : 2darray
            matrix of return N asset for L time
        ret_cov : 2darray
            corss correlation matrix N x N
        ticker : list
            list of ticker we want include in our portfolio
        risk_free_rate : float, optional, default 0
            The risk-free rate is the rate of return on an investment
            that has a zero chance of loss. It means the investment 
            is so safe that there is no risk associated with it.
        days_in_yr : int, optionale, default 252
            days of financial market activity
        '''
        self.returns        = returns
        self.ret_cov        = ret_cov
        self.risk_free_rate = risk_free_rate
        self.days_in_yr     = days_in_yr
        self.ticker         = ticker
 

    def ret_portfolio(self, w):
        '''
        Function to compute the return of the porfolio.
        
        Parameter
        ---------
        w : 1darray
            weights of portfolio

        Return
        ------
        ret_port : float
            return of the portfolio
        '''
        ret_port = np.mean(self.returns, axis=1)
        ret_port = sum( w * ret_port) * self.days_in_yr
        return ret_port


    def vol_portfolio(self, w):
        '''
        Function to compute the volatility of the porfolio.
        
        Parameter
        ---------
        w : 1darray
            weights of portfolio
        
        Return
        ------
        vol_port : float
            volatility of the portfolio
        '''
        vol_port = np.sqrt((w.T @ self.ret_cov @ w) * self.days_in_yr)
        return vol_port


    def tot_portfolio(self, w):
        '''
        Function to compute the whole portfolio information
        
        Parameter
        ---------
        w : 1darray
            weights of portfolio
        
        Return
        ------
        mu : float
            return of the portfolio
        std : float
            volatility of the portfolio
        '''
        mu  = self.ret_portfolio(w)
        std = self.vol_portfolio(w)
        return mu, std


    def negative_sharpe_ratio(self, w):
        '''
        Function to compute the sharpe ratio, and we want to maximize it
        so we add a minus and then minimize th negative sharpe ratio.

        Parameter
        ---------
        w : 1darray
            weights of portfolio

        Returns
        -------
        sharpe_ratio : float
            It measures the additional return above the
            return of the risk-free asset per unit of volatility
        '''

        ret_prt, vol_prt = self.tot_portfolio(w)
        sharpe_ratio = (ret_prt - self.risk_free_rate)/vol_prt 

        return -sharpe_ratio


    def minimum(self, f, constraints, args=()):
        '''
        Function to compute the optimal weights via optimizzatio.
        Although it is possible to change the constraints,
        here we consider the bounds and the starting point
        he same for every situation.

        Parameters
        ----------
        f : callable
            function to minimize, 
            portfolio's volatility or negative sharpe ratio
        constraints : tuple of dictionarys
            all contraints for minimization, e.g. unitary weights
            and fixed portfolio's return
        args : tuple
            extra argument to pass at f

        Return
        ------
        result : dict
            result of minimization
        '''
        # Number of asset in our portfolio
        n = self.returns.shape[0]
        # All weights must be between 0 and 1, they are percentages
        bounds = tuple((0, 1) for _ in range(n))
        # Initial guess
        init = n * [1 / n]
        # minimizzation
        result = opt.minimize(f, init, args=args, bounds=bounds,
                              constraints=constraints)
        return result


    def optimization(self, N_P):
        '''
        Function to compute all efficient frontier

        Parameter
        ---------
        N_P : int
            number of points, i.e. nuber of portfolios

        Returns
        -------
        frontier_vol : 1darray
            result of optimizzation for volatility (x-axis)
        target : 1darray
            return that we want for our potfolio (y-axis)
        data : pandas DataFrame
            DataFrame with optimal weights and associated ticker
        '''
        # Frontier array
        frontier_vol = np.zeros(N_P)
        # Dataframe to store all porfolios
        data = pd.DataFrame(index=self.ticker, columns=[f"{i}" for i in range(N_P)])
        # Unitarity constraint
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.})

        # Maximization of sharpe ratio
        result_sr = self.minimum(self.negative_sharpe_ratio, constraints)
        ret_prt_sr, vol_prt_sr = self.tot_portfolio(result_sr['x'])
        # Minimization of volatility
        result_mv = self.minimum(self.vol_portfolio, constraints)
        ret_prt_mv, vol_prt_mv = self.tot_portfolio(result_mv['x'])

        # Store data
        frontier_vol[ 0] = vol_prt_mv 
        frontier_vol[-1] = vol_prt_sr
        data["0"]        = [round(i*100, 2) for i in result_sr['x']]
        data[f"{N_P-1}"] = [round(i*100, 2) for i in result_mv['x']]

        # Frontier's computation
        target = np.linspace(ret_prt_mv, ret_prt_sr, N_P)

        for i, t in enumerate(target[1:-1]):

            # Unitarity constraint and return constraint
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                           {'type': 'eq', 'fun': lambda w: self.ret_portfolio(w) - t})
            tmp = self.minimum(self.vol_portfolio, constraints)
            # Store data
            frontier_vol[i+1] = tmp['fun']
            data[f"{i+1}"] = [round(j*100, 2) for j in tmp['x']]
            print(f"{(i+1)/(N_P-2) * 100:.2f} % \r", end='')

        frontier_vol, target = 100*frontier_vol, 100*target

        return frontier_vol, target, data

if __name__ == '__main__':

    from build_dataset import load

    #==============================================================================
    # Pre-loaded data
    #==============================================================================
    """
    n_asset    = 5   # number of asset in our portfolio (max 431 for return_2.npy)
    n_days     = 1006   # number of day (max 1005 for return_2.npy)
    days_in_yr = 252
    
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

    return_1d  = np.array(return_1d)
    cross_corr = np.cov(return_1d)
    #"""

    #==============================================================================
    # Computation
    #==============================================================================

    portfolio = Portfolio(return_1d, cross_corr, list_tk, days_in_yr=days_in_yr)
    frontier_vol, frontier_ret, data = portfolio.optimization(20)
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
        ret[i] = sum( w * np.mean(return_1d, axis=1)) * days_in_yr
        vol[i] = np.sqrt((w.T @ cross_corr @ w) * days_in_yr)

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
