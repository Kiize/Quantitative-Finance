import yfinance_cache as yf
import pandas as pd
import numpy as np
import powerlaw

"""
We study the behavior of the exponent \alpha by considering a two years span and moving this time window by two months, thus constructing a time series.
The data are stored in a .csv file and will be analyzed in time_series_alpha.py
"""

start_yr = 1989
start_m = 1
end_yr = start_yr + 2

data = []

while(end_yr < 2025):
    start = f"{start_yr}-{start_m}-01"
    end = f"{end_yr}-{start_m}-01"
    tmp_data = yf.download(tickers = "^GSPC", start = start, end = end, interval = "1d", adjust_splits=True, adjust_divs=True)
    daily_closes = tmp_data['Close']
    log_ret = np.log(daily_closes) - np.log(daily_closes.shift(1))
    log_ret = log_ret.dropna()


    results = powerlaw.Fit(np.abs(log_ret.values).ravel())
    data.append([start, end, results.power_law.alpha])

    if start_m + 2 < 13:
        start_m += 2  
    else:
        start_yr += 1
        end_yr += 1
        start_m = (start_m + 2)%12

    
df = pd.DataFrame(data)
df.columns = ["Start date", "End date", "alpha"]
df.to_csv("alpha_arr.csv", index=False, encoding="utf-8")