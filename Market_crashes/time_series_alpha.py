import yfinance_cache as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import batched

"""
We analyze the time series of \alpha focusing in particular on its behavior during the market crashes.
"""

df = pd.read_csv("alpha_arr.csv", encoding="utf-8", header=0)
print(df)

start = "1990-12-01"
end = "2024-11-01"

data = yf.download(tickers = "^GSPC", start = start, end = end, interval = "1d", adjust_splits=True, adjust_divs=True)
crash_dates = ["1997-01-01", "2002-01-01", "2007-01-01", "2009-01-01", "2020-01-01", "2021-01-01"]

fig, (ax1, ax2) = plt.subplots(
    nrows=2, 
    ncols=1, 
    sharex=True 
    #figsize=(12, 8),
    # Use gridspec_kw to control the relative height of the plots
    #gridspec_kw={'height_ratios': [2, 1]} 
)

ax1.plot(data.index, data['Close'], label='S&P 500 Index')
ax1.set_yscale('log') # Log scale is best for price charts
ax1.set_ylabel('S&P 500')
ax1.set_title(r'S&P 500 Index and Rolling Tail Exponent $\alpha$')

ax2.plot(pd.to_datetime(df["End date"]), df["alpha"], label=r'Tail Exponent $\alpha(t)$', color='r', marker='.')
ax2.set_ylabel(r'Tail Exponent $\alpha$')
ax2.set_xlabel('Date')
ax2.set_ylim([2, 10])

for crash0, crash1 in batched(crash_dates, n=2):
    ax1.axvspan(crash0, crash1, alpha=0.4)
    ax2.axvspan(crash0, crash1, alpha=0.4)

plt.savefig("rolling_tail_exponent.png")
plt.show()