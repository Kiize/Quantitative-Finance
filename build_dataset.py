"""
Code to download stock histories to create the dataset to analyze.
for SP500
https://pkgstore.datahub.io/core/s-and-p-500-companies/10/datapackage.json
"""
import time
import numpy as np
import datapackage
import pandas as pd
import yfinance as yf

def load(start_date, end_date, list_ticker=None, json_link=None, save=False):

    if list_ticker is None and json_link is None:
        err_msg = "You must pass a list of tiker or \
                   \na link to json file with all information"
        raise Exception(err_msg)
    
    if list_ticker :
        # Dictionary which will contain the histories of the various titles
        history = {}
        ticker_len = len(list_ticker)

        for index, ticker in enumerate(list_ticker):
            # Loop over the entire data set and read the symbol
            ticker = row['Symbol']
            print(f"[{(index + 1)}/{dataset_len}] download of: {ticker}")

            # Download the history of the title, in the time range of interest
            hist = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)

            # Check 
            if hist.empty:
                print(f"The history of {ticker} is not available")
            else:
                history[ticker] = hist

        # Save
        if save:
            np.save("data/dataset.npy", history)
        else:
            return histy
    
    if json_link is not None:
        
        package = datapackage.Package(json_link)
        resources = package.resources


        # Only the data written in the table are of interest,
        # which contain: symbol, name and sector
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
            hist = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)

            # Check 
            if hist.empty:
                print(f"The history of {ticker} is not available")
            else:
                history[ticker] = hist

        # Save
        np.save("data/dataset.npy", history)

    

if __main__ = "__main__":
    start = time.time()
    load()
    mins = (time.time()-start)//60
    sec  = (time.time()-start) % 60
    print(f"Elapsed time: {mins} min {sec:.2f} sec")
    

