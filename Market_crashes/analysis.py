import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import batched
from scipy.stats import norm
from scipy.optimize import curve_fit
import powerlaw

"""
We study the log returns of the S&P 500 index from 1990 to 2025, focusing on their behavior during the market crashes of 2000, 2008, 2020.
"""
start = "1990-01-01"
end = "2025-01-01"

data = yf.download(tickers = "^GSPC", start = start, end = end, interval = "1d", auto_adjust=True)
daily_closes = data['Close']
log_ret = np.log(daily_closes) - np.log(daily_closes.shift(1))
log_ret = log_ret.dropna()
crash_dates = ["1997-01-01", "2002-01-01", "2007-01-01", "2009-01-01", "2020-01-01", "2021-01-01"]

mean_gauss = np.mean(log_ret.values)
std_gauss = np.std(log_ret.values)

# Log-ret: we plot the log returns of our index

fig = plt.figure()
plt.title(r"Log return $r(t)$")
plt.plot(log_ret.index, log_ret.values)
for crash0, crash1 in batched(crash_dates, n=2):
    plt.axvspan(crash0, crash1, alpha=0.4)

plt.savefig("log_ret_SP500.png")

# Abs log-ret: we plot the absolute value of the log returns

fig2 = plt.figure()
plt.title(r"Abs log return $|r(t)|$")
plt.plot(log_ret.index, np.abs(log_ret.values))
for crash0, crash1 in batched(crash_dates, n=2):
    plt.axvspan(crash0, crash1, alpha=0.4)

plt.savefig("abs_log_ret_SP500.png")

# Hist: we plot an histogram of the log returns and compare them with a gaussian distribution with the same mean and variance

fig3 = plt.figure()
plt.title("Histogram log return")
plt.hist(log_ret.values, bins=200, density=True, label='S&P 500 Log Returns', alpha=0.7)
xmin, xmax = -0.1, 0.1
x = np.linspace(xmin, xmax, 100)
plt.plot(x, norm.pdf(x, mean_gauss, std_gauss), color = "r")
#plt.savefig("hist_log_ret_SP500.png")

# Power law tails: we fit the power law tails of our distribution using the powerlaw package.

fig4 = plt.figure()
x_sort = np.abs(log_ret.values)
x_sort.ravel().sort()
y_ccdf = 1.0 - (np.arange(len(x_sort)) / len(x_sort))
fit_threshold = 0.01
x_sort = x_sort[x_sort > fit_threshold]
y_ccdf = y_ccdf[len(y_ccdf) - len(x_sort):]
plt.xscale("log")
plt.yscale("log")

results = powerlaw.Fit(np.abs(log_ret.values).ravel())
print(results.power_law.alpha)
print(results.power_law.xmin)

fig_fit = results.plot_ccdf(marker="o", linewidth=0)
results.power_law.plot_ccdf(color="r", ax = fig_fit)
plt.savefig("fit_log_ret_SP500.png") 

# Autocorrelations: we plot the autocorrelations over n_days of our log returns

to_corr = log_ret.values.ravel()
corr = np.correlate(to_corr, to_corr, "full")
n_days = 10
corr = corr[corr.size//2:corr.size//2 + n_days] # simmetria
corr = corr/corr[0]    # Normalizzazione
fig_corr = plt.figure()
plt.title(f"Autocorrelations for {n_days} days")
plt.plot(corr, "o")
plt.savefig("autocorr.png")

# Volatility: we plot the autocorrelations over n_days_vol of our absolute log returns

to_vol = np.abs(log_ret.values).ravel()
vol = np.correlate(to_vol, to_vol, "full")
n_days_vol = 250
vol = vol[vol.size//2:vol.size//2 + n_days_vol]
fig_vol = plt.figure()
plt.title(f"Autocorrelations of volatility for {n_days_vol} days")
plt.plot(vol, "o")
plt.savefig("volatility.png")

plt.show()