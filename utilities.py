"""
Code with some usefull functions
"""

import numpy as np

def dens_prob(L, N):
    """
    Function that returns the distribution of
    random matrix's eigenvalues in the limit
    of large N and L.
    A random matrix R is defined as: R = 1/L A.T@A
    where A is np.random.randn(N,L).

    Parameters
    ----------
    L, N : int
       parameters of matrix 

    Returns
    -------
    l, P : array
        l is an array for plot and P is the distribution
    
    Example
    -------
    x, y = dens_prob(1000, 500)
    plt.plot(x, y)
    
    """
    q = L/N

    #limiti del supporto
    l1 = 1 + 1/q - 2*np.sqrt(1/q)
    l2 = 1 + 1/q + 2*np.sqrt(1/q)

    l = np.linspace(l1, l2, 1000)
    P = (q/(2*np.pi) * np.sqrt((l2-l)*(l-l1)))/l

    return l, P


def corr(x, y):
    """
    Function to compute temporal corralation between x and y

    Parameters
    ----------
    x : 1darray
        temporal series
    y : 1darray
        temporal series

    Returns
    ----------
    corr : floar
        correlation
    """
    
    sigmaxy = ((x - np.mean(x))*(y -  np.mean(y))).sum()
    sigmax  = np.sqrt(((x - np.mean(x))**2).sum())
    sigmay  = np.sqrt(((y - np.mean(y))**2).sum())
    corr    = sigmaxy/(sigmax*sigmay)

    return corr


def pro(retunorm, eigvec, k):
    """
    Function to compute the projection of temporal series
    over the k-th eigenvalue of cross-correlation

    Parameters
    ----------
    retunorm : 2darray
        matrix with all return (normalized)
    eigvec : 2darray
        matrix of eigvectors of cross-correlation
    k : int
        index of eigvectors

    Returns
    ----------
    pro : 1darray
        array containing the projection of the temporal
        series of the titles on the k-th eigenvector
    """

    N, L = retunorm.shape
    pro = np.zeros(L)
    for j in range(L):
        a = 0
        for i in range(N):
            a += retunorm[i, j]*eigvec[i, k]
        pro[j] = a

    pro = pro/np.sqrt(np.mean(pro**2)-np.mean(pro)**2)

    return pro


def I(eigvec):
    """
    Function to compute the inverse participation ratio

    Parameters
    ----------
    eigvec : 2darray
        matrix of eigvectors of cross-correlation
    """
    
    N = eigvec.shape[0]
    I = np.zeros(N)

    for i in range(N):
        a = 0
        for j in range(N):
            a += v[j, i]**4
        I[i] = a

    return I


def find_max(x, k):
    """
    Function that finds the indices of the first k maxima,
    we want not only positive but also negative peaks,
    doing abs(x) everything translates into finding the highest maxima

    Parameters
    ----------
    x : 1darray
        array of which to find the maxima
    k : int
        Number of indices we want find

    Returns
    ----------
    index : 1darray
        the array of indices of the k maxima, sorted in decreasing order
    """

    # To avoid problem we make a copy
    v = x.copy()
    v = np.abs(v)
    index = np.zeros(k)
    
    # First maximum
    v_m = np.max(v)
    ind = np.where(v == v_m)[0][0]
    index[0] = ind
    
    # I remove the newly found element
    # so I can iterate the procedure
    v = np.delete(v, ind)

    for i in range(1, k):
        v_m = np.max(v)
        ind = np.where(v == v_m)[0][0]
        if ind < index[i-1]:
            index[i] = ind
        else:
            index[i] = ind + i
        v = np.delete(v, ind)

    return index


def find_info(data_info, ticker):
    """
    Function which, given the titles data frame,
    finds the symbol and returns the information of
    the associated title (contained in the data frame)
    in the form of a table with latex syntax.

    Parameters
    ----------
    data_info : pandas data frame
        table containing symbols, names and sectors
    ticker : list
        list of tickers to trace in the data frame
        to obtain the name of the title and sector
    """

    print('\hline')
    print(r'Symbol & Name & Sector \\')
    
    for tk in ticker: # loop over titles

        # I find the symbol and keep the index
        ind = np.where(data_info['Symbol']==tk)[0][0]

        # I find the name and sector of the associated index
        name = data_info['Name'][ind]
        sect = data_info['Sector'][ind]

        print('\hline')
        print(rf'{tk} & {name} &  {sect} \\')

    print('\hline')
