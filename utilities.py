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
    pro  = np.zeros(L)

    for j in range(L):
        a = 0
        for i in range(N):
            a += retunorm[i, j]*eigvec[i, k]
        pro[j] = a

    # normalizzation
    pro = pro/np.sqrt(np.mean(pro**2)-np.mean(pro)**2)

    return pro


def q_operator(C, sigma):
    """
    Function to reweight the correlation matrix with volatilities.
    This is because the cross correlation is calculated with normalized returns.

    Parameters
    ----------
    C : 2darray
        cross correlation matrix
    sigma : 1d array
        volatility of each index

    Return
    ------
    Q : 2darray
        Q = \sum_i \sum_j C_ij s_i s_j
    """
    n = len(sigma)

    Q = [[C[i, j]*sigma[i]*sigma[j] for i in range(n)] for j in range(n)]
    Q = np.array(Q)

    return Q    


def c_filtering(C, return_arr):
    """
    Function to compute the filtered correlation matrix
    using the information of the greater of the greater eigenvalues.

    C : 2darray
        cross correlation matrix
    return_arr : 2darray
        matrix of all return

    Returns
    -------
    C_filtered : 2darray
        filtered matrix with the information of bigger eigenvalues
    """
    # C filtering.
    n_asset = len(return_arr[:,0])

    eigval, eigvec = np.linalg.eig(C)
    eigvals = np.sort(eigval)   
    eigvec = eigvec[:, eigval.argsort()]

    q = len(return_arr[0,:])/n_asset

    #limiti del supporto
    #l1 = 1 + 1/q - 2*np.sqrt(1/q)
    l2 = 1 + 1/q + 2*np.sqrt(1/q)

    tmp = eigvals[eigvals > l2]
    tmp_eig = np.append(np.zeros(n_asset - len(tmp)), tmp)
    tmp_eig_inv = np.append(np.zeros(n_asset - len(tmp)), 1/tmp)
    lambda_eig = np.diag(tmp_eig)
    lambda_eig_inv = np.diag(tmp_eig_inv)

    C_filtered = lambda_eig @ eigvec @ lambda_eig_inv

    # We keep the trace conserved.
    for i in range(n_asset):
        C_filtered[i, i] = 1

    return C_filtered


