"""
Code to download stock histories to create the dataset to analyze.
For SP500 data is possible to use: 
https://pkgstore.datahub.io/core/s-and-p-500-companies/10/datapackage.json
"""
import time
import numpy as np
import datapackage
import pandas as pd
import yfinance as yf


def load(start_date, end_date, list_ticker=None, json_link=None,
         save=False, path_data=None, path_info=None):
    '''
    Function to load history of titles via yfinance.
    It's possibile to pass a list of ticker or a link to a .json file
    with all information; for example for all SP500 is possible to use:
    https://pkgstore.datahub.io/core/s-and-p-500-companies/10/datapackage.json

    It's also possible to save the data in some file; in this case the function
    will return None; If we don't want to save the data the function will return
    the all data in a dictionary.

    One between list_ticker or json_link is mandatory

    Parameters
    ----------
    start_date : string
        start of the title's history, in the form: "yyyy-mm-dd"
    end_date : string
        end of the title's history, in the form: "yyyy-mm-dd"
    list_ticker : list, optional, default None
        list of ticker that we want
    json_link : string, optional, default None
        link to json file wiht necessary information
    save : bool, optional, default False
        set True to save data in .npy file
    path_data : string, optional, default None
        path for save data; mandatory for save=True
    path_info : string, optional, default None
        path to save some other information that are in the file .json

    Return
    ------
    history : dict
        dictionary with all the desired title's history
    '''

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
            print(f"[{(index + 1)}/{ticker_len}] download of: {ticker}")

            # Download the history of the title, in the time range of interest
            hist = yf.download(ticker, start=start_date, end=end_date,
                               interval='1d', progress=False)

            # Check 
            if hist.empty:
                print(f"The history of {ticker} is not available")
            else:
                history[ticker] = hist

        # Save
        if save:
            np.save(path_data, history)
        else:
            return history

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
        dataset.to_csv(f'{path_info}.csv', index=False)

        # Dictionary which will contain the histories of the various titles
        history = {}
        dataset_len = len(dataset)

        for index, row in dataset.iterrows():
            # Loop over the entire data set and read the symbol
            ticker = row['Symbol']
            print(f"[{(index + 1)}/{dataset_len}] download of: {ticker}")

            # Download the history of the title, in the time range of interest
            hist = yf.download(ticker, start=start_date, end=end_date,
                               interval='1d', progress=False)

            # Check 
            if hist.empty:
                print(f"The history of {ticker} is not available")
            else:
                history[ticker] = hist

        # Save
        if save:
            np.save(path_data, history)
        else:
            return history


if __name__ == "__main__":

    sp500 = r"https://pkgstore.datahub.io/core/s-and-p-500-companies/10/datapackage.json"
    path_file = r"data/16_19"
    path_info = r"data/16_19_information"

    start = time.time()
    start_date = "2016-01-01"
    end_date   = "2019-12-31"

    load(start_date, end_date, json_link=sp500, 
         save=True, path_data=path_file, path_info=path_info)

    mins = (time.time()-start)//60
    sec  = (time.time()-start) % 60

    print(f"Elapsed time: {mins} min {sec:.2f} sec")

