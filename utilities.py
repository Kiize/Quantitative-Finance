"""
Codice contenente tutte le funzioni necessarie
"""

import numpy as np

def dens_prob(L, N):
    """
    funzione che restituisce la distribuzione
    degli autovalori di una matrice random
    nel limite di N ed L infiniti.

    Parameters
    ----------
    L, N : int
       Parametri della matrice

    Returns
    ----------
    dic : dict
        dizionario contenete : {'supporto': l, 'pdf': P}
        l : 1darray
            array del supporto della distribuzione
        P : 1darray
            array che contiene La distribuzione
    """
    q = L/N

    #limiti del supporto
    l1 = 1 + 1/q - 2*np.sqrt(1/q)
    l2 = 1 + 1/q + 2*np.sqrt(1/q)

    l = np.linspace(l1, l2, 1000)
    P = (q/(2*np.pi) * np.sqrt((l2-l)*(l-l1)))/l

    dic = {'supporto': l, 'pdf': P}
    return dic


def corr(x, y):
    """
    funzione che dati due array x(t) y(t)
    ne calcola la correlazione temporale

    Parameters
    ----------
    x : 1darray
        array contenente l'andamento temporale di x
    y : 1darray
        array contenente l'andamento temporale di y

    Returns
    ----------
    corr : floar
        correlazione dei due array
    """
    sigmaxy = ((x - np.mean(x))*(y -  np.mean(y))).sum()
    sigmax = np.sqrt(((x - np.mean(x))**2).sum())
    sigmay = np.sqrt(((y - np.mean(y))**2).sum())
    corr = sigmaxy/(sigmax*sigmay)

    return corr


def pro(retunorm, eigvec, k, L, N):
    """
    Calcolo della proiezione delle serie
    temporali sul k-esimo autovettore

    Parameters
    ----------
    retunorm : 2darray
        matrice contenenete i ritorni degli indici
    eigvec : 2darray
        matrice contenete gli autovettori
    k : int
        numero che labella gli autovettori
    L : int
        lunghezza serie temporale
    N : int
        numero di titoli

    Returns
    ----------
    pro : 1darray
        array contenente la proiezione delle serie
        temporali dei titoli sull'autovettore k-esimo
    """

    pro = np.zeros(L)
    for j in range(L):
        a = 0
        for i in range(N):
            a += retunorm[i, j]*eigvec[i, k]
        pro[j] = a

    pro = pro/np.sqrt(np.mean(pro**2)-np.mean(pro)**2)

    return pro


def I(N, v):
    """
    funzione che restituisce
    l'inverse participation ratio

    Parameters
    ----------
    N : int
        numero dei titloi
    v : 2darray
        matrice che contiene gli autovettori
    """

    I = np.zeros(N)

    for i in range(N):
        a = 0
        #sommo le componenti degli array alla quarta
        for j in range(N):
            a += v[j, i]**4
        I[i] = a

    return I


def find_max(x, k):
    """
    funzione che trova gli indici dei primi k massimi,
    in realtà si vogliono i picchi più ampi sia positivi
    che negativi, facendo abs(x) tutto si traduce nel
    trovare i massimi maggiori

    Parameters
    ----------
    x : 1darray
        array di cui trovare i massimi
    k : int
        numero di massimi da trovare

    Returns
    ----------
    index : 1darray
        array degli indici dei k massimi,
        oridnati in maniera decrescente
    """

    #uso la funzione copy per non modificare l'array originarioi
    v = x.copy()
    v = np.abs(v)
    index = np.zeros(k)
    #trovo il primo massimo e conservo l'indice
    max = np.max(v)
    ind = np.where(v == max)[0][0]
    index[0] = ind
    #tolgo l'elemento trovato per poter iterare
    v = np.delete(v, ind)

    for i in range(1, k):
        max = np.max(v)
        ind = np.where(v == max)[0][0]
        if ind < index[i-1]:
            index[i] = ind
        else:
            index[i] = ind + i
        v = np.delete(v, ind)

    return index


def find_info(data_info, ticker):
    """
    funzione che dato il dataframe dei titoli
    trova il simbolo e restituisce le informazioni
    del titolo contenute del dataframe in forma
    di una tabella con la sintassi di latex
    in modo da poter copiare il risultato su latex

    Parameters
    ----------
    data_info : pandas data frame
        tabella contenenete simbolli, nomi e settori
    ticker : list
        lista dei ticker da rintracciare nel dataframe
        per ricavare nome del titolo e settore
    """

    #stampo su shell con sintassi latex
    print('\hline')
    print(r'Symbol & Name & Sector \\')
    #ciclo su tutti i tioli
    for tk in ticker:

        #trovo il simbolo e conservo l'indice
        ind = np.where(data_info['Symbol']==tk)[0][0]

        #trovo nome e settore del relativo indice
        name = data_info['Name'][ind]
        sect = data_info['Sector'][ind]

        #stampo su shell con sintassi latex
        print('\hline')
        print(rf'{tk} & {name} &  {sect} \\')
    print('\hline')

def q_operator(C, sigma):
    n = len(sigma)
    Q = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Q[i, j] = C[i, j]*sigma[i]*sigma[j]

    return Q    

def c_filtering(C, return_arr):
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

    C_filtered = lambda_eig @ C @ lambda_eig_inv

    # We keep the trace conserved.
        
    for i in range(n_asset):
        C_filtered[i, i] = 1

    return C_filtered


