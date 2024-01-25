"""
Code to compute the cross-correlation matrix
"""

import time
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

start = time.time()

start_story = "2020-01-01" # Start date for stock histories
end_story   = "2023-12-31" # End   date for stock histories

# Read SP500
print('read SP500 history')
SP500 = yf.download('^GSPC', start=start_story, end=end_story, interval='1d', progress=False)
L = len(SP500)

# Read the data created via: build_datset.py
history = np.load(r"data/dataset_2.npy",allow_pickle='TRUE').item()

# Some titles may have been 'born' during the period of time
# considered and are therefore excluded from the analysis
index = [] # list that will contain the titles to be analyzed

for i, ticker in enumerate(history.keys()):
    
    open_t = history[ticker]['Open']
    
    # If the story is of a different length than that of the
    # SP500 (in principle it can only be shorter)
    # I exclude the title from the analysis
    if len(open_t) != L:
        print(f'The history of {ticker} is too short, it will be excluded')
    else:
        index.append(ticker) # I keep the remaining titles


N = len(index)

print(f'Number of index: {N}, length of time intervall: {L-1}')

# Some important matrix
return_1d   = np.zeros((N, L-1))
return_norm = np.zeros((N, L-1))
cross_corr  = np.zeros((N, N))

# matrix for shuffle
ret_norm_shuff   = np.zeros((N, L-1))
cross_corr_shuff = np.zeros((N, N))



for i, ticker in enumerate(index):

    # Compute the normalized return to 1 day
    adj_close = history[ticker]['Adj Close']

    return_1d[i, :] = adj_close.pct_change()[1:]
    
    sigma = np.sqrt(np.mean(return_1d[i, :]**2)-np.mean(return_1d[i, :])**2)
    return_norm[i, :] = (return_1d[i, :] - np.mean(return_1d[i, :]))/sigma

    # Like befor but with a shuffle to destroy correlation
    adj_close_shuff = np.random.choice(return_1d[i, :], size=L-1, replace=False)
    sigma = np.sqrt(np.mean(adj_close_shuff**2)-np.mean(adj_close_shuff)**2)
    ret_norm_shuff[i, :] = (adj_close_shuff - np.mean(adj_close_shuff))/sigma
    
    print(f"{(i+1)/N * 100:.2f} % \r", end='')


# Compute the cross correlation matrix
cross_corr       = np.cov(return_norm)
cross_corr_shuff = np.cov(ret_norm_shuff)

np.save("data/cross_correlation_2.npy", cross_corr)
np.save("data/normalized_return_2.npy", return_norm)
np.save("data/return_2.npy", return_1d)
np.save("data/indici_2.npy", index)

mins = (time.time()-start)//60
sec  = (time.time()-start) % 60

print(f"Elapsed time: {mins} min {sec:.2f} sec")

#==============================================================================
# Some pretty plot
#==============================================================================

# For a true random matrix
q   = L/N
l1  = 1 + 1/q - 2*np.sqrt(1/q)
l2  = 1 + 1/q + 2*np.sqrt(1/q)
l   = np.linspace(l1, l2, 1000)
pdf = (q/(2*np.pi) * np.sqrt((l2 - l)*(l - l1)))/l


plt.figure(1)
x = np.reshape(cross_corr, N*N)
plt.title('Distribution of cross-correlation coefficients',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
#plt.yscale('log')
plt.grid()
plt.hist(x, int(np.sqrt(N*N-1)), density=True)


plt.figure(2)
y = np.linalg.eigvalsh(cross_corr)
plt.title('Distribution of the eigenvalues of $C_{ij}$',fontsize=15)
plt.xlabel('$\lambda$ of $C_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.grid()
plt.yscale('log')
plt.plot(l, pdf, 'k')
plt.hist(y, N, density=True)


plt.figure(3)
x = np.reshape(cross_corr_shuff, N*N)
plt.title('Distribution of cross-correlation coefficients\n after shuffle',fontsize=15)
plt.xlabel('cross correlation $C_{ij}$',fontsize=15)
plt.ylabel('P($C_{ij}$)',fontsize=15)
plt.hist(x, int(np.sqrt(N*N-1)), density=True)
#plt.yscale('log')
plt.grid()


plt.figure(4)
y = np.linalg.eigvalsh(cross_corr_shuff)
plt.title('Distribution of the eigenvalues of $C_{ij}$ after shuffle',fontsize=15)
plt.xlabel('$\lambda$ of $C_{ij}$', fontsize=15)
plt.ylabel('P($\lambda$)', fontsize=15)
plt.hist(y, int(np.sqrt(N-1)), density=True)
plt.plot(l, pdf, 'k')
plt.grid()

plt.show()
