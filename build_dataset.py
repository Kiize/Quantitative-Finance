"""
Code to download stock histories to create the dataset to analyze.
"""
import time
import numpy as np
import datapackage
import pandas as pd
import yfinance as yf

start = time.time()

start_story = '2020-01-01' # Start date for stock histories
end_story   = '2023-12-31' # End   date for stock histories

# Information on S&P500 stocks
#DATA_URL = r"https://datahub.io/core/s-and-p-500-companies/datapackage.json"
DATA_URL = r"https://pkgstore.datahub.io/core/s-and-p-500-companies-financials/3/datapackage.json"
package = datapackage.Package(DATA_URL)
resources = package.resources


# Only the data written in the table are of interest,
# which contain: symbol, name and securities sector
for resource in resources:
    if resource.tabular:
        dataset = pd.read_csv(resource.descriptor['path'])

# Now dataset contains symbols, names and sectors of stocks.
# We reorder it in alphabetical order of the symbol and readjust
# the index that labels it
dataset = dataset.sort_values('Symbol').reset_index(drop=True)

# Save info on a csv
dataset.to_csv('data/dataset_infromation.csv', index=False)

# Dictionary which will contain the histories of the various titles
history = {}
dataset_len = len(dataset)

for index, row in dataset.iterrows():
    # Loop over the entire data set and read the symbol
    ticker = row['Symbol']
    print(f"[{(index + 1)}/{dataset_len}] download of: {ticker}")

    # Download the history of the title, in the time range of interest
    hist = yf.download(ticker, start=start_story, end=end_story, interval='1d', progress=False)

    # Check 
    if hist.empty:
        print(f"The history of {ticker} is not available")
    else:
        history[ticker] = hist

# Save
np.save("data/dataset.npy", history)

mins = (time.time()-start)//60
sec = (time.time()-start) % 60
print(f"Elapsed time: {mins} min {sec:.2f} sec")
