import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Random portfolio.

n_asset = 100
n_days = 1000
days_in_yr = 252

#ass_index = n_asset * ['A']
ass_index = np.load('data/indici.npy', allow_pickle='TRUE')
ass_index = ass_index[0:n_asset]


return1d = np.load('data/return.npy', allow_pickle='TRUE')
return1d = return1d[0:n_asset, 0:n_days]

#return1d = np.random.randn(n_asset, n_days)
crosscor = np.cov(return1d)

def portfolio(w, returns, ret_cov):
    mean_ret = np.mean(returns, axis=1)
    mean_prt = sum(w * mean_ret) * days_in_yr
    devs_prt = np.sqrt( (w.T @ ret_cov @ w) * days_in_yr)

    return mean_prt, devs_prt

def negativeSharpeRatio(w, returns, ret_cov, risk_free_rate = 0.):
    mean_prt, devs_prt = portfolio(w, returns, ret_cov)
    shape_ratio = (mean_prt - risk_free_rate)/devs_prt 

    return -shape_ratio

def maxSharpeRatio(returns, ret_cov, risk_free_rate = 0.):
    n = returns.shape[0]
    args = (returns, ret_cov, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0,1) for _ in range(n))

    init = n * [1/n]

    result = opt.minimize(negativeSharpeRatio, init, args=args, bounds=bounds, constraints=constraints)

    return result

def get_result(returns, ret_cov, risk_free_rate=0.):
    result = maxSharpeRatio(returns, ret_cov, risk_free_rate)

    mean_prt_sr, devs_prt_sr = portfolio(result['x'], returns, ret_cov)
    mean_prt_sr, devs_prt_sr = 100 * mean_prt_sr, 100 * devs_prt_sr

    sr_data = pd.DataFrame(result['x'], index=ass_index, columns=['allocation'])

    sr_data.allocation = [round(i*100, 0) for i in sr_data.allocation]

    return mean_prt_sr, devs_prt_sr, sr_data

mean_prt_sr, devs_prt_sr, sr_data = get_result(return1d, crosscor)

sr_data = sr_data[sr_data['allocation'] > 0.]
print(sr_data)









    